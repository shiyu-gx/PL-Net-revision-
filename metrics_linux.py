import argparse
import ast
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.EarthworkDataset import EarthworkDataset
from src.unet import UNet
from src.segment_anything import sam_model_registry, SamPredictor
from src.utils.dice_score import multiclass_dice_coeff


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


def compute_metrics(mask_pred, mask_true):
    """Compute Dice, mIoU, and Pixel Accuracy."""
    mask_true_one_hot = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
    mask_pred_one_hot = F.one_hot(mask_pred, 2).permute(0, 3, 1, 2).float()
    dice = multiclass_dice_coeff(
        mask_pred_one_hot[:, 1:],
        mask_true_one_hot[:, 1:],
        reduce_batch_first=False
    )

    y_true = mask_true.cpu().numpy()
    y_pred = mask_pred.cpu().numpy()

    pixel_accuracy = np.mean(y_pred == y_true)

    iou_list = []
    for c in range(2):
        intersection = np.sum((y_true == c) & (y_pred == c))
        union = np.sum((y_true == c) | (y_pred == c))
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        iou_list.append(iou)

    mean_iou = np.mean(iou_list)
    return dice.item(), mean_iou, pixel_accuracy


def clean_mask(mask, min_component_area=1000, max_aspect_ratio=5.0):
    """Remove small or extremely elongated connected components."""
    mask_np = mask[0].detach().cpu().numpy().astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

    for i in range(1, len(stats)):
        area = stats[i][cv2.CC_STAT_AREA]
        width = max(stats[i][cv2.CC_STAT_WIDTH], 1)
        height = max(stats[i][cv2.CC_STAT_HEIGHT], 1)
        aspect_ratio = max(width / height, height / width)

        if area < min_component_area or aspect_ratio > max_aspect_ratio:
            mask_np[labels == i] = 0

    return mask_np


def compute_volume(depth, dem_scale, pixel_size_m=0.03, depth_size_m=0.03):
    """Compute excavation volume in cubic meters."""
    pixel_area_m2 = (dem_scale * pixel_size_m) ** 2
    depth_m = depth * depth_size_m
    volume_m3 = np.sum(depth_m) * pixel_area_m2
    print(f"Excavation volume: {volume_m3:.2f} cubic meters")
    return volume_m3


def predict_img(
    net,
    data,
    sam_predictor,
    device,
    fold_dir,
    results_root,
    exp_name,
    min_component_area,
    max_aspect_ratio,
    pixel_size_m,
    depth_size_m,
    save_debug=False,
):
    net.eval()

    image_name = str(data["image_name"][0])
    save_dir = results_root / exp_name / fold_dir / image_name
    save_dir.mkdir(parents=True, exist_ok=True)

    image = data["image_dem_norm"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    mask_true_orig = data["true_mask"].to(device=device, dtype=torch.long)
    valid_mask = data["valid_mask"].to(device=device, dtype=torch.bool)
    image_pad = data["image_pad"]
    elev_valid_mask = data["elev_valid_mask"]

    unlabeled = (mask_true_orig != 1).unsqueeze(1)

    if sam_predictor is not None:
        with torch.no_grad():
            image_pad = image_pad.to(device=device, dtype=torch.uint8)
            batch_size, height, width = image_pad.shape[0], image_pad.shape[1], image_pad.shape[2]
            sam_logits_pred = torch.zeros((batch_size, 2, height, width), device=device)

            if save_debug:
                cv2.imwrite(str(save_dir / "image_pad.png"), image_pad[0].detach().cpu().numpy())
                cv2.imwrite(str(save_dir / "mask_true.png"), mask_true_orig[0].detach().cpu().numpy() * 255)

            for idx in range(batch_size):
                elev_valid_mask_256 = F.interpolate(
                    elev_valid_mask[idx].float().unsqueeze(0).unsqueeze(0),
                    size=(256, 256),
                    mode="nearest",
                )
                elev_valid_mask_256 = elev_valid_mask_256.squeeze(0).detach().cpu().numpy() * 255

                sam_predictor.set_image(image_pad[idx].detach().cpu().numpy())
                sam_logits, _, _ = sam_predictor.predict(
                    mask_input=elev_valid_mask_256,
                    multimask_output=False,
                    return_logits=True,
                )

                if save_debug:
                    sam_pred_mask = sam_logits > 0.0
                    sam_binary_mask = (sam_pred_mask.astype(np.uint8) * 255)
                    cv2.imwrite(
                        str(save_dir / f"{image_name}_metrics_sam_pred_mask.png"),
                        sam_binary_mask.squeeze()
                    )
                    sam_unlabeled_mask = (
                        sam_binary_mask.squeeze()
                        * unlabeled[idx].squeeze().detach().cpu().numpy().astype(np.uint8)
                    )
                    cv2.imwrite(
                        str(save_dir / f"{image_name}_metrics_sam_unlabeled_mask.png"),
                        sam_unlabeled_mask
                    )

                sam_prob_foreground = torch.sigmoid(torch.tensor(sam_logits, device=device))
                sam_prob_background = 1.0 - sam_prob_foreground
                sam_probs = torch.cat([sam_prob_background, sam_prob_foreground], dim=0)
                sam_probs_logits = torch.log(sam_probs + 1e-8)
                sam_probs_logits = torch.where(
                    valid_mask[idx] > 0,
                    sam_probs_logits,
                    torch.zeros_like(sam_probs_logits)
                )
                sam_logits_pred[idx] = sam_probs_logits
    else:
        sam_logits_pred = None

    with torch.no_grad():
        if sam_predictor is not None:
            mask_pred_logits, unet_logits = net(image, sam_logits_pred)

            unet_logits = torch.where(
                valid_mask.unsqueeze(1) > 0,
                unet_logits,
                torch.zeros_like(unet_logits)
            )
            unet_pred = unet_logits.argmax(dim=1)

            mask_pred_logits = torch.where(
                valid_mask.unsqueeze(1) > 0,
                mask_pred_logits,
                torch.zeros_like(mask_pred_logits)
            )
            mask_pred = mask_pred_logits.argmax(dim=1)

            dice_score, mean_iou, pixel_accuracy = compute_metrics(mask_pred, mask_true_orig)
            print(f"{image_name} dice score: {dice_score:.4f}, mIoU: {mean_iou:.4f}, Pixel Acc: {pixel_accuracy:.4f}")

            cleaned_unet_np = clean_mask(
                unet_pred,
                min_component_area=min_component_area,
                max_aspect_ratio=max_aspect_ratio
            )
            cleaned_unet = torch.from_numpy(cleaned_unet_np).unsqueeze(0).long().to(device)

            cleaned_mask_np = clean_mask(
                mask_pred,
                min_component_area=min_component_area,
                max_aspect_ratio=max_aspect_ratio
            )
            cleaned_mask = torch.from_numpy(cleaned_mask_np).unsqueeze(0).long().to(device)

            dice_score_clean, mean_iou_clean, pixel_accuracy_clean = compute_metrics(cleaned_mask, mask_true_orig)
            print(
                f"{image_name} (cleaned) dice score: {dice_score_clean:.4f}, "
                f"mIoU: {mean_iou_clean:.4f}, Pixel Acc: {pixel_accuracy_clean:.4f}"
            )

            dem = data["dem_pad"][0].cpu().numpy()
            dem_scale = data["dem_scale"][0].item()
            dem_excavation = dem[cleaned_mask_np > 0]
            dem_excavation_pic = dem * cleaned_mask_np
            cv2.imwrite(str(save_dir / f"{image_name}_dem_excavation.png"), dem_excavation_pic)
            volume = compute_volume(
                dem_excavation,
                dem_scale,
                pixel_size_m=pixel_size_m,
                depth_size_m=depth_size_m
            )

            with open(save_dir / f"{image_name}_metrics.txt", "w", encoding="utf-8") as f:
                f.write(f"Dice Score: {dice_score:.4f}\n")
                f.write(f"Mean IoU: {mean_iou:.4f}\n")
                f.write(f"Pixel Accuracy: {pixel_accuracy:.4f}\n")
                f.write(f"Dice Score (cleaned): {dice_score_clean:.4f}\n")
                f.write(f"Mean IoU (cleaned): {mean_iou_clean:.4f}\n")
                f.write(f"Pixel Accuracy (cleaned): {pixel_accuracy_clean:.4f}\n")
                f.write(f"Excavation Volume (cubic meters): {volume:.2f}\n")

            cv2.imwrite(str(save_dir / f"{image_name}_mask_pred.png"), cleaned_mask_np.astype(np.uint8) * 255)
            cv2.imwrite(str(save_dir / f"{image_name}_unet_pred.png"), cleaned_unet_np.astype(np.uint8) * 255)

            if save_debug:
                unlabeled_mask_np = (
                    mask_pred.squeeze().detach().cpu().numpy()
                    * unlabeled[0].squeeze().detach().cpu().numpy().astype(np.uint8)
                    * 255
                )
                cv2.imwrite(str(save_dir / f"{image_name}_unlabeled_mask_pred.png"), unlabeled_mask_np)

        else:
            mask_pred_logits = net(image)
            mask_pred_logits = torch.where(
                valid_mask.unsqueeze(1) > 0,
                mask_pred_logits,
                torch.zeros_like(mask_pred_logits)
            )
            mask_pred = mask_pred_logits.argmax(dim=1)

            dice_score, mean_iou, pixel_accuracy = compute_metrics(mask_pred, mask_true_orig)
            print(f"{image_name} dice score: {dice_score:.4f}, mIoU: {mean_iou:.4f}, Pixel Acc: {pixel_accuracy:.4f}")

            cleaned_mask_np = clean_mask(
                mask_pred,
                min_component_area=min_component_area,
                max_aspect_ratio=max_aspect_ratio
            )
            cleaned_mask = torch.from_numpy(cleaned_mask_np).unsqueeze(0).long().to(device)

            dice_score_clean, mean_iou_clean, pixel_accuracy_clean = compute_metrics(cleaned_mask, mask_true_orig)
            print(
                f"{image_name} (cleaned) dice score: {dice_score_clean:.4f}, "
                f"mIoU: {mean_iou_clean:.4f}, Pixel Acc: {pixel_accuracy_clean:.4f}"
            )

            dem = data["dem_pad"][0].cpu().numpy()
            dem_scale = data["dem_scale"][0].item()
            dem_excavation = dem[cleaned_mask_np > 0]
            dem_excavation_pic = dem * cleaned_mask_np
            cv2.imwrite(str(save_dir / f"{image_name}_dem_excavation.png"), dem_excavation_pic)
            volume = compute_volume(
                dem_excavation,
                dem_scale,
                pixel_size_m=pixel_size_m,
                depth_size_m=depth_size_m
            )

            with open(save_dir / f"{image_name}_metrics.txt", "w", encoding="utf-8") as f:
                f.write(f"Dice Score: {dice_score:.4f}\n")
                f.write(f"Mean IoU: {mean_iou:.4f}\n")
                f.write(f"Pixel Accuracy: {pixel_accuracy:.4f}\n")
                f.write(f"Dice Score (cleaned): {dice_score_clean:.4f}\n")
                f.write(f"Mean IoU (cleaned): {mean_iou_clean:.4f}\n")
                f.write(f"Pixel Accuracy (cleaned): {pixel_accuracy_clean:.4f}\n")
                f.write(f"Excavation Volume (cubic meters): {volume:.2f}\n")

            cv2.imwrite(str(save_dir / f"{image_name}_mask_pred.png"), cleaned_mask_np.astype(np.uint8) * 255)

            if save_debug:
                unlabeled_mask_np = (
                    mask_pred.squeeze().detach().cpu().numpy()
                    * unlabeled[0].squeeze().detach().cpu().numpy().astype(np.uint8)
                    * 255
                )
                cv2.imwrite(str(save_dir / f"{image_name}_unlabeled_mask_pred.png"), unlabeled_mask_np)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation metrics and excavation volume")

    parser.add_argument("--exp_name", type=str, default="baseline", help="Experiment directory name")
    parser.add_argument("--target_size", "-s", type=int, default=1024, help="Input image target size")
    parser.add_argument("--in_channels", type=int, default=3, help="Input channels: 3 for RGB, 4 for RGB+DEM")
    parser.add_argument("--use_sam", type=str2bool, default=False, help="Whether to use SAM")
    parser.add_argument("--sam_model_type", type=str, default="vit_l", help="SAM model type")
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="model_weights/SAM/sam_vit_l_0b3195.pth",
        help="Path to the SAM checkpoint"
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="add",
        choices=["add", "csaf", "none"],
        help="SAM and UNet fusion method"
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "dice", "cross_entropy_dice", "tcs"],
        help="Loss function name, kept for experiment consistency"
    )
    parser.add_argument("--threshold_m", type=float, default=5.91, help="Elevation threshold in meters")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes")

    parser.add_argument("--data_root", type=str, default="data/Earthwork", help="Dataset root directory")
    parser.add_argument("--results_root", type=str, default="logs/results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="auto", help='Device: "auto", "cpu", "cuda", "cuda:0", etc.')

    parser.add_argument("--min_component_area", type=int, default=1000, help="Minimum connected component area")
    parser.add_argument("--max_aspect_ratio", type=float, default=5.0, help="Maximum connected component aspect ratio")
    parser.add_argument("--pixel_size_m", type=float, default=0.03, help="Ground pixel size in meters")
    parser.add_argument("--depth_size_m", type=float, default=0.03, help="DEM depth resolution in meters")
    parser.add_argument("--save_debug", action="store_true", help="Save debug visualizations")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root_dir = Path(__file__).resolve().parent
    data_root = resolve_path(root_dir, args.data_root)
    results_root = resolve_path(root_dir, args.results_root)
    sam_checkpoint = resolve_path(root_dir, args.sam_checkpoint)
    exp_dir = resolve_path(root_dir, args.exp_name)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logging.info(f"Using device {device}")
    logging.info(f"Experiment directory: {exp_dir}")

    img_dir = data_root / "imgs"
    mask_dir = data_root / "masks"
    dem_dir = data_root / "dems"

    dataset = EarthworkDataset(
        img_dir,
        mask_dir,
        dem_dir,
        target_size=args.target_size,
        in_channels=args.in_channels,
        threshold_m=args.threshold_m
    )

    predictor = None
    if args.use_sam:
        logging.info("Loading SAM model")
        sam = sam_model_registry[args.sam_model_type](checkpoint=str(sam_checkpoint)).to(device)
        predictor = SamPredictor(sam)

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    fold_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir()])

    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under: {exp_dir}")

    for fold_path in fold_dirs:
        checkpoint_path = fold_path / "checkpoint_epoch.pth"
        split_path = fold_path / "fold_split.txt"

        if not checkpoint_path.exists():
            logging.warning(f"Skipping {fold_path.name}: missing checkpoint_epoch.pth")
            continue
        if not split_path.exists():
            logging.warning(f"Skipping {fold_path.name}: missing fold_split.txt")
            continue

        logging.info(f"Processing {fold_path.name} ...")

        with open(split_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            val_idx = ast.literal_eval(lines[-1].strip().split(" ", 2)[-1])

        val_set = torch.utils.data.Subset(dataset, val_idx)
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True
        )

        net = UNet(
            n_channels=args.in_channels,
            n_classes=args.classes,
            bilinear=args.bilinear,
            use_sam=args.use_sam,
            fusion_method=args.fusion_method
        )
        net.to(device=device)

        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict.pop("mask_values", [0, 1])
        net.load_state_dict(state_dict)

        logging.info("Model loaded")

        for idx, data in enumerate(val_loader):
            logging.info(f"Predicting image {idx} ...")
            predict_img(
                net=net,
                data=data,
                sam_predictor=predictor,
                device=device,
                fold_dir=fold_path.name,
                results_root=results_root,
                exp_name=args.exp_name,
                min_component_area=args.min_component_area,
                max_aspect_ratio=args.max_aspect_ratio,
                pixel_size_m=args.pixel_size_m,
                depth_size_m=args.depth_size_m,
                save_debug=args.save_debug,
            )
