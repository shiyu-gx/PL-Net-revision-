import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.hipify.hipify_python import str2bool
from tqdm import tqdm

import wandb
from evaluate import evaluate
from src.unet import UNet
from src.utils.EarthworkDataset import EarthworkDataset
from src.utils.dice_score import dice_loss
from src.segment_anything import sam_model_registry, SamPredictor
from src.utils.loss import TCSLoss


root_dir = Path(__file__).resolve().parent

# Public-repo-friendly relative paths
dir_img = root_dir / Path("data/Earthwork/imgs/")
dir_mask = root_dir / Path("data/Earthwork/masks/")
dir_dem = root_dir / Path("data/Earthwork/dems/")


def train_model(
    args,
    model,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    target_size: int = 1024,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    use_sam: bool = True,
    sam_model_type: str = "vit_l",
    sam_checkpoint: str = "model_weights/SAM/" + "sam_vit_l_0b3195.pth",
):
    # 1. Create dataset
    try:
        dataset = EarthworkDataset(
            images_dir=dir_img,
            mask_dir=dir_mask,
            dem_dir=dir_dem,
            target_size=target_size,
            in_channels=args.in_channels,
            threshold_m=args.threshold_m,
        )
    except (AssertionError, RuntimeError, IndexError):
        logging.error(
            "Error while loading the dataset. "
            "Make sure that you have the images, masks and dems in the correct folders."
        )
        sys.exit(1)

    # Load SAM model if use_sam is True
    if use_sam:
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device)
        predictor = SamPredictor(sam)

    # 2. Perform 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(kf.split(dataset))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

        # 3. Create data loaders
        loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

        logging.info(
            f"Fold {fold_idx + 1}: Training size: {len(train_set)}, "
            f"Validation size: {len(val_set)}"
        )

        # Initialize logging
        experiment = wandb.init(project="Earthwork", resume="allow", anonymous="must")
        experiment.config.update(
            dict(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_percent=val_percent,
                save_checkpoint=save_checkpoint,
                target_size=target_size,
                amp=amp,
            )
        )

        logging.info(
            f"""Starting training:
            Epochs:            {epochs}
            Batch size:        {batch_size}
            Learning rate:     {learning_rate}
            Training size:     {len(train_set)}
            Validation size:   {len(val_set)}
            Checkpoints:       {save_checkpoint}
            Device:            {device.type}
            Image target size: {target_size}
            Mixed Precision:   {amp}
        """
        )

        # 4. Set up optimizer, scheduler, AMP scaler, and loss
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=True,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=5
        )  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        tcs_loss = TCSLoss(
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3,
            criterion=criterion,
            dice_loss=dice_loss,
        )
        global_step = 0

        # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0

            with tqdm(total=len(train_set), desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
                for batch in train_loader:
                    images = batch["image_dem_norm"]
                    true_masks = batch["true_mask"]
                    valid_masks = batch["valid_mask"]
                    image_pads = batch["image_pad"]
                    elev_valid_masks = batch["elev_valid_mask"]

                    images = images.to(
                        device=device,
                        dtype=torch.float32,
                        memory_format=torch.channels_last,
                    )
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                    valid_masks = valid_masks.to(device=device, dtype=torch.long)

                    assert images.shape[1] == model.n_channels, (
                        f"Network has been defined with {model.n_channels} input channels, "
                        f"but loaded images have {images.shape[1]} channels. Please check that "
                        "the images are loaded correctly."
                    )

                    if use_sam:
                        with torch.no_grad():
                            image_pads = image_pads.to(device=device, dtype=torch.uint8)
                            sam_logits_pred = torch.zeros(
                                (image_pads.shape[0], 2, image_pads.shape[1], image_pads.shape[2]),
                                device=device,
                            )  # [B, 2, H, W]

                            for idx in range(image_pads.shape[0]):
                                elev_valid_mask_256 = F.interpolate(
                                    elev_valid_masks[0].float().unsqueeze(0).unsqueeze(0),
                                    size=(256, 256),
                                    mode="nearest",
                                )
                                elev_valid_mask_256 = (
                                    elev_valid_mask_256.squeeze(0).detach().cpu().numpy() * 255
                                )

                                predictor.set_image(image_pads[idx].detach().cpu().numpy())
                                sam_logits, _, _ = predictor.predict(
                                    mask_input=elev_valid_mask_256,
                                    multimask_output=False,
                                    return_logits=True,
                                )  # shape: (1, H, W)

                                sam_prob_foreground = torch.sigmoid(
                                    torch.tensor(sam_logits, device=device)
                                )
                                sam_prob_background = 1.0 - sam_prob_foreground
                                sam_probs = torch.cat(
                                    [sam_prob_background, sam_prob_foreground], dim=0
                                )  # (2, H, W)
                                sam_probs_logits = torch.log(sam_probs + 1e-8)

                                sam_probs_logits = torch.where(
                                    valid_masks[idx] > 0,
                                    sam_probs_logits,
                                    torch.zeros_like(sam_probs_logits),
                                )

                                sam_logits_pred[idx] = sam_probs_logits

                    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                        if use_sam:
                            masks_pred, unet_logits = model(images, sam_logits_pred)
                        else:
                            masks_pred = model(images)

                        masks_pred = torch.where(
                            valid_masks.unsqueeze(1) > 0,
                            masks_pred,
                            torch.zeros_like(masks_pred),
                        )

                        if model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(
                                torch.sigmoid(masks_pred.squeeze(1)),
                                true_masks.float(),
                                multiclass=False,
                            )
                        else:
                            if use_sam:
                                if args.loss_function == "tcs":
                                    loss = tcs_loss(masks_pred, unet_logits, true_masks, sam_logits_pred)
                                elif args.loss_function == "cross_entropy":
                                    loss = criterion(masks_pred, true_masks)
                                elif args.loss_function == "dice":
                                    loss = dice_loss(
                                        F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, model.n_classes)
                                        .permute(0, 3, 1, 2)
                                        .float(),
                                        multiclass=True,
                                    )
                                elif args.loss_function == "cross_entropy_dice":
                                    loss = criterion(masks_pred, true_masks)
                                    loss += dice_loss(
                                        F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, model.n_classes)
                                        .permute(0, 3, 1, 2)
                                        .float(),
                                        multiclass=True,
                                    )
                                else:
                                    raise ValueError(f"Unknown loss function: {args.loss_function}")
                            else:
                                if args.loss_function == "cross_entropy":
                                    loss = criterion(masks_pred, true_masks)
                                elif args.loss_function == "dice":
                                    loss = dice_loss(
                                        F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, model.n_classes)
                                        .permute(0, 3, 1, 2)
                                        .float(),
                                        multiclass=True,
                                    )
                                elif args.loss_function == "cross_entropy_dice":
                                    loss = criterion(masks_pred, true_masks)
                                    loss += dice_loss(
                                        F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, model.n_classes)
                                        .permute(0, 3, 1, 2)
                                        .float(),
                                        multiclass=True,
                                    )
                                else:
                                    raise ValueError(f"Unknown loss function: {args.loss_function}")

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()

                    experiment.log(
                        {
                            "train loss": loss.item(),
                            "step": global_step,
                            "epoch": epoch,
                        }
                    )
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                    # Evaluation round
                    if args.evaluate:
                        division_step = 2
                        if division_step > 0 and global_step % division_step == 0:
                            histograms = {}
                            for tag, value in model.named_parameters():
                                tag = tag.replace("/", ".")
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
                                if value.grad is not None and not (
                                    torch.isinf(value.grad) | torch.isnan(value.grad)
                                ).any():
                                    histograms["Gradients/" + tag] = wandb.Histogram(
                                        value.grad.data.cpu()
                                    )

                            if use_sam:
                                val_score = evaluate(model, val_loader, device, amp, predictor)
                            else:
                                val_score = evaluate(model, val_loader, device, amp)

                            scheduler.step(loss)
                            logging.info(f"Validation Dice score: {val_score}")

                            try:
                                experiment.log(
                                    {
                                        "learning rate": optimizer.param_groups[0]["lr"],
                                        "validation Dice": val_score,
                                        "images": wandb.Image(images[0].cpu()),
                                        "masks": {
                                            "true": wandb.Image(true_masks[0].float().cpu()),
                                            "pred": wandb.Image(
                                                masks_pred.argmax(dim=1)[0].float().cpu()
                                            ),
                                        },
                                        "step": global_step,
                                        "epoch": epoch,
                                        **histograms,
                                    }
                                )
                            except Exception:
                                pass

            if epoch % epochs == 0:
                # Save only the last epoch model for each fold
                dir_checkpoint = root_dir / Path(f"{args.exp_name}/")
                fold_checkpoint_dir = dir_checkpoint / f"fold_{fold_idx}"
                fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)

                with open(fold_checkpoint_dir / "fold_split.txt", "w") as f:
                    f.write(f"Train indices: {train_idx.tolist()}\n")
                    f.write(f"Validation indices: {val_idx.tolist()}\n")

                state_dict = model.state_dict()
                state_dict["mask_values"] = 2
                torch.save(state_dict, str(fold_checkpoint_dir / "checkpoint_epoch.pth"))
                logging.info(f"Checkpoint {epoch} saved!")


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--epochs", "-e", metavar="E", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--batch-size", "-b", dest="batch_size", metavar="B", type=int, default=1, help="Batch size"
    )
    parser.add_argument(
        "--learning-rate", "-l", metavar="LR", type=float, default=1e-5,
        help="Learning rate", dest="lr"
    )
    parser.add_argument("--load", "-f", type=str, default=False, help="Load model from a .pth file")
    parser.add_argument("--target_size", "-s", type=int, default=256, help="Resize target size of the images")
    parser.add_argument(
        "--validation", "-v", dest="val", type=float, default=10.0,
        help="Percent of the data that is used as validation (0-100)"
    )
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes")
    parser.add_argument("--evaluate", "-val", default=False, help="Evaluate during training")

    # New Params
    parser.add_argument("--exp_name", type=str, default="baseline", help="Name of the experiment for logging purposes")
    parser.add_argument(
        "--in_channels", type=int, default=3,
        help="Number of input channels to the model, e.g., 4 for RGB + DEM, 3 for RGB"
    )
    parser.add_argument("--use_sam", type=str2bool, default=False, help="Whether to use SAM model for segmentation")
    parser.add_argument(
        "--sam_model_type", type=str, default="vit_l",
        help="Type of SAM model to use (e.g., vit_l, vit_b, vit_h)"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="model_weights/SAM/" + "sam_vit_l_0b3195.pth",
        help="Path to SAM model checkpoint"
    )
    parser.add_argument(
        "--fusion_method", type=str, default="add", choices=["add", "csaf", "none"],
        help="Method to fuse SAM and UNet features (e.g., csaf, add)"
    )
    parser.add_argument(
        "--loss_function", type=str, default="cross_entropy",
        choices=["cross_entropy", "dice", "cross_entropy_dice", "tcs"],
        help="Loss function to use during training"
    )
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for final output GT loss in TCS Loss")
    parser.add_argument("--lambda2", type=float, default=0.5, help="Weight for UNet intermediate GT loss in TCS Loss")
    parser.add_argument("--lambda3", type=float, default=0.5, help="Weight for final output SAM loss in TCS Loss")
    parser.add_argument(
        "--threshold_m", type=float, default=5.91,
        help="Elevation threshold in meters for generating SAM masks"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(
        n_channels=args.in_channels,
        n_classes=args.classes,
        bilinear=args.bilinear,
        use_sam=args.use_sam,
        fusion_method=args.fusion_method,
    )
    model = model.to(memory_format=torch.channels_last)

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
    )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict["mask_values"]
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {args.load}")

    model.to(device=device)

    try:
        train_model(
            args=args,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            target_size=args.target_size,
            val_percent=args.val / 100,
            amp=args.amp,
            use_sam=args.use_sam,
            sam_model_type=args.sam_model_type,
            sam_checkpoint=args.sam_checkpoint,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training. "
            "Consider enabling AMP (--amp) for fast and memory efficient training"
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            args=args,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            target_size=args.target_size,
            val_percent=args.val / 100,
            amp=args.amp,
            use_sam=args.use_sam,
            sam_model_type=args.sam_model_type,
            sam_checkpoint=args.sam_checkpoint,
        )
