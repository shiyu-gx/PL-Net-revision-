import argparse
import copy
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from src.unet import UNet
from src.utils.EarthworkDataset import EarthworkDataset
from src.utils.dice_score import dice_loss
from src.segment_anything import sam_model_registry, SamPredictor
from src.utils.loss import TCSLoss


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def resolve_path(root_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (root_dir / path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args) -> UNet:
    model = UNet(
        n_channels=args.in_channels,
        n_classes=args.classes,
        bilinear=args.bilinear,
        use_sam=args.use_sam,
        fusion_method=args.fusion_method,
    )
    return model.to(memory_format=torch.channels_last)


def load_model_weights(model: UNet, checkpoint_path: Path, device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict.pop("mask_values", None)
    model.load_state_dict(state_dict)
    return model


def create_wandb_run(args, fold_idx: int):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError as e:
        raise ImportError("wandb is not installed, but --use_wandb was enabled.") from e

    run = wandb.init(
        project=args.wandb_project,
        name=f"{args.exp_name}_fold_{fold_idx}",
        resume="allow",
        anonymous="allow" if args.wandb_anonymous else "never",
        config=vars(args),
    )
    return run


def safe_wandb_log(run, payload: dict):
    if run is None:
        return
    try:
        run.log(payload)
    except Exception:
        pass


def build_sam_predictor(args, device: torch.device, root_dir: Path):
    if not args.use_sam:
        return None

    sam_checkpoint = resolve_path(root_dir, args.sam_checkpoint)
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(sam_checkpoint)).to(device)
    predictor = SamPredictor(sam)
    return predictor


def prepare_sam_logits(
    image_pads,
    elev_valid_masks,
    valid_masks,
    predictor,
    device,
    save_debug=False,
    debug_dir: Path | None = None,
):
    """
    Build SAM logits in two-channel logit form: [background_logit, foreground_logit].
    """
    image_pads = image_pads.to(device=device, dtype=torch.uint8)
    batch_size, height, width = image_pads.shape[0], image_pads.shape[1], image_pads.shape[2]
    sam_logits_pred = torch.zeros((batch_size, 2, height, width), device=device)

    for idx in range(batch_size):
        elev_valid_mask_256 = F.interpolate(
            elev_valid_masks[idx].float().unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode="nearest",
        )
        elev_valid_mask_256 = elev_valid_mask_256.squeeze(0).detach().cpu().numpy() * 255

        predictor.set_image(image_pads[idx].detach().cpu().numpy())
        sam_logits, _, _ = predictor.predict(
            mask_input=elev_valid_mask_256,
            multimask_output=False,
            return_logits=True,
        )

        if save_debug and debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            sam_pred_mask = sam_logits > 0.0
            sam_binary_mask = (sam_pred_mask.astype(np.uint8) * 255)
            cv2.imwrite(str(debug_dir / f"sam_pred_fold_debug_{idx}.png"), sam_binary_mask.squeeze())

        sam_prob_foreground = torch.sigmoid(torch.tensor(sam_logits, device=device))
        sam_prob_background = 1.0 - sam_prob_foreground
        sam_probs = torch.cat([sam_prob_background, sam_prob_foreground], dim=0)
        sam_probs_logits = torch.log(sam_probs + 1e-8)

        sam_probs_logits = torch.where(
            valid_masks[idx].unsqueeze(0) > 0,
            sam_probs_logits,
            torch.zeros_like(sam_probs_logits),
        )
        sam_logits_pred[idx] = sam_probs_logits

    return sam_logits_pred


def compute_segmentation_loss(
    args,
    model,
    criterion,
    tcs_loss_fn,
    masks_pred,
    true_masks,
    use_sam=False,
    unet_logits=None,
    sam_logits_pred=None,
):
    if model.n_classes == 1:
        loss = criterion(masks_pred.squeeze(1), true_masks.float())
        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
        return loss

    if use_sam and args.loss_function == "tcs":
        return tcs_loss_fn(masks_pred, unet_logits, true_masks, sam_logits_pred)

    if args.loss_function == "cross_entropy":
        return criterion(masks_pred, true_masks)

    if args.loss_function == "dice":
        return dice_loss(
            F.softmax(masks_pred, dim=1).float(),
            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True,
        )

    if args.loss_function == "cross_entropy_dice":
        loss = criterion(masks_pred, true_masks)
        loss += dice_loss(
            F.softmax(masks_pred, dim=1).float(),
            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True,
        )
        return loss

    raise ValueError(f"Unknown loss function: {args.loss_function}")


def train_one_epoch(
    args,
    model,
    train_loader,
    optimizer,
    grad_scaler,
    criterion,
    tcs_loss_fn,
    predictor,
    device,
    epoch,
    epochs,
    run=None,
):
    model.train()
    epoch_loss = 0.0
    global_examples = 0

    autocast_device = device.type if device.type != "mps" else "cpu"

    with tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
        for step, batch in enumerate(train_loader):
            images = batch["image_dem_norm"].to(
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last,
            )
            true_masks = batch["true_mask"].to(device=device, dtype=torch.long)
            valid_masks = batch["valid_mask"].to(device=device, dtype=torch.bool)
            image_pads = batch["image_pad"]
            elev_valid_masks = batch["elev_valid_mask"]

            assert images.shape[1] == model.n_channels, (
                f"Network expects {model.n_channels} input channels, "
                f"but got {images.shape[1]}."
            )

            sam_logits_pred = None
            if args.use_sam:
                sam_logits_pred = prepare_sam_logits(
                    image_pads=image_pads,
                    elev_valid_masks=elev_valid_masks,
                    valid_masks=valid_masks,
                    predictor=predictor,
                    device=device,
                    save_debug=args.save_debug and epoch == 1 and step == 0,
                    debug_dir=Path(args.debug_dir) if args.debug_dir else None,
                )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=autocast_device, enabled=args.amp):
                if args.use_sam:
                    masks_pred, unet_logits = model(images, sam_logits_pred)
                else:
                    masks_pred = model(images)
                    unet_logits = None

                masks_pred = torch.where(
                    valid_masks.unsqueeze(1),
                    masks_pred,
                    torch.zeros_like(masks_pred),
                )

                loss = compute_segmentation_loss(
                    args=args,
                    model=model,
                    criterion=criterion,
                    tcs_loss_fn=tcs_loss_fn,
                    masks_pred=masks_pred,
                    true_masks=true_masks,
                    use_sam=args.use_sam,
                    unet_logits=unet_logits,
                    sam_logits_pred=sam_logits_pred,
                )

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            batch_size = images.shape[0]
            global_examples += batch_size
            epoch_loss += loss.item() * batch_size

            pbar.update(batch_size)
            pbar.set_postfix(**{"loss (batch)": f"{loss.item():.4f}"})

            safe_wandb_log(
                run,
                {
                    "train/loss_step": loss.item(),
                    "train/epoch": epoch,
                },
            )

    mean_epoch_loss = epoch_loss / max(global_examples, 1)
    return mean_epoch_loss


def save_fold_artifacts(
    output_dir: Path,
    fold_idx: int,
    train_idx,
    val_idx,
    model: UNet,
):
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    with open(fold_dir / "fold_split.txt", "w", encoding="utf-8") as f:
        f.write(f"Train indices: {train_idx.tolist()}\n")
        f.write(f"Validation indices: {val_idx.tolist()}\n")

    state_dict = model.state_dict()
    state_dict["mask_values"] = [0, 1]
    torch.save(state_dict, fold_dir / "checkpoint_epoch.pth")


def get_args():
    parser = argparse.ArgumentParser(description="Train PL-Net on RGB/DEM earthwork data")

    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", "-l", dest="lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--load", type=str, default="", help="Optional pretrained checkpoint path")
    parser.add_argument("--target_size", "-s", type=int, default=256, help="Input resize target size")
    parser.add_argument("--validation", dest="val", type=float, default=10.0, help="Reserved for compatibility")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes")
    parser.add_argument("--evaluate", action="store_true", help="Run validation at the end of each epoch")

    parser.add_argument("--exp_name", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--in_channels", type=int, default=3, help="3 for RGB, 4 for RGB+DEM")
    parser.add_argument("--use_sam", type=str2bool, default=False, help="Whether to use SAM")
    parser.add_argument("--sam_model_type", type=str, default="vit_l", help="SAM model type")
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="model_weights/SAM/sam_vit_l_0b3195.pth",
        help="Path to SAM checkpoint",
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="add",
        choices=["add", "csaf", "none"],
        help="Feature fusion method",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "dice", "cross_entropy_dice", "tcs"],
        help="Loss function",
    )
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for final output GT loss in TCS")
    parser.add_argument("--lambda2", type=float, default=0.5, help="Weight for UNet intermediate GT loss in TCS")
    parser.add_argument("--lambda3", type=float, default=0.5, help="Weight for final output SAM loss in TCS")
    parser.add_argument("--threshold_m", type=float, default=5.91, help="Elevation threshold in meters")

    parser.add_argument("--data_root", type=str, default="data/Earthwork", help="Dataset root directory")
    parser.add_argument("--device", type=str, default="auto", help='Device: "auto", "cpu", "cuda", "cuda:0", etc.')
    parser.add_argument("--n_splits", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")

    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Optimizer weight decay")
    parser.add_argument("--momentum", type=float, default=0.999, help="RMSprop momentum")
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="Gradient clipping max norm")

    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="Earthwork", help="W&B project name")
    parser.add_argument("--wandb_anonymous", action="store_true", help="Use anonymous W&B mode")

    parser.add_argument("--save_debug", action="store_true", help="Save a small amount of SAM debug output")
    parser.add_argument("--debug_dir", type=str, default="debug", help="Directory for debug images")

    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    set_seed(args.seed)

    root_dir = Path(__file__).resolve().parent
    data_root = resolve_path(root_dir, args.data_root)
    output_dir = root_dir / args.exp_name

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logging.info(f"Using device {device}")

    img_dir = data_root / "imgs"
    mask_dir = data_root / "masks"
    dem_dir = data_root / "dems"

    try:
        dataset = EarthworkDataset(
            images_dir=img_dir,
            mask_dir=mask_dir,
            dem_dir=dem_dir,
            target_size=args.target_size,
            in_channels=args.in_channels,
            threshold_m=args.threshold_m,
        )
    except (AssertionError, RuntimeError, IndexError) as e:
        logging.error(
            "Error while loading the dataset. "
            "Make sure images, masks, and DEMs are present under data_root."
        )
        raise e

    predictor = build_sam_predictor(args, device, root_dir)

    pretrained_state_dict = None
    if args.load:
        load_path = resolve_path(root_dir, args.load)
        pretrained_state_dict = torch.load(load_path, map_location=device)
        pretrained_state_dict.pop("mask_values", None)
        logging.info(f"Loaded initial weights from {load_path}")

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = list(kf.split(dataset))

    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Number of folds: {args.n_splits}")
    logging.info(f"Experiment directory: {output_dir}")

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logging.info(
            f"Fold {fold_idx + 1}/{args.n_splits}: "
            f"training size={len(train_idx)}, validation size={len(val_idx)}"
        )

        model = build_model(args)
        if pretrained_state_dict is not None:
            model.load_state_dict(copy.deepcopy(pretrained_state_dict))

        logging.info(
            f"Network:\n"
            f"\t{model.n_channels} input channels\n"
            f"\t{model.n_classes} output channels (classes)\n"
            f"\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling"
        )

        model.to(device=device)

        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

        loader_args = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            foreach=True,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        tcs_loss_fn = TCSLoss(
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3,
            criterion=criterion,
            dice_loss=dice_loss,
        )

        run = create_wandb_run(args, fold_idx)

        best_val_score = None

        for epoch in range(1, args.epochs + 1):
            mean_train_loss = train_one_epoch(
                args=args,
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                criterion=criterion,
                tcs_loss_fn=tcs_loss_fn,
                predictor=predictor,
                device=device,
                epoch=epoch,
                epochs=args.epochs,
                run=run,
            )

            payload = {
                "train/loss_epoch": mean_train_loss,
                "train/epoch": epoch,
                "train/lr": optimizer.param_groups[0]["lr"],
            }

            if args.evaluate:
                if args.use_sam:
                    val_score = evaluate(model, val_loader, device, args.amp, predictor)
                else:
                    val_score = evaluate(model, val_loader, device, args.amp)

                scheduler.step(val_score)
                payload["val/dice"] = float(val_score)
                best_val_score = float(val_score) if best_val_score is None else max(best_val_score, float(val_score))

                logging.info(
                    f"Fold {fold_idx} | Epoch {epoch}/{args.epochs} | "
                    f"train_loss={mean_train_loss:.6f} | val_dice={float(val_score):.6f}"
                )
            else:
                logging.info(
                    f"Fold {fold_idx} | Epoch {epoch}/{args.epochs} | "
                    f"train_loss={mean_train_loss:.6f}"
                )

            safe_wandb_log(run, payload)

        save_fold_artifacts(
            output_dir=output_dir,
            fold_idx=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            model=model,
        )
        logging.info(f"Saved fold_{fold_idx} checkpoint to {output_dir / f'fold_{fold_idx}'}")

        if run is not None:
            try:
                if best_val_score is not None:
                    run.summary["best_val_dice"] = best_val_score
                run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "CUDA OutOfMemoryError detected. Try reducing batch size, image size, or enabling --amp."
        )
        torch.cuda.empty_cache()
        sys.exit(1)
