import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.hipify.hipify_python import str2bool

from src.utils.EarthworkDataset import EarthworkDataset
from src.unet import UNet
from src.segment_anything import sam_model_registry, SamPredictor
from src.utils.dice_score import multiclass_dice_coeff


def compute_metrics(mask_pred, mask_true):
    """Compute Dice, mIoU, and Pixel Accuracy."""
    mask_true_one_hot = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
    mask_pred_one_hot = F.one_hot(mask_pred, 2).permute(0, 3, 1, 2).float()
    dice = multiclass_dice_coeff(
        mask_pred_one_hot[:, 1:],
        mask_true_one_hot[:, 1:],
        reduce_batch_first=False,
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


def clean_mask(mask):
    """Remove small or elongated connected foreground regions."""
    mask = mask[0].cpu().numpy()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    for i in range(1, len(stats)):
        area = stats[i][cv2.CC_STAT_AREA]
        width = stats[i][cv2.CC_STAT_WIDTH]
        height = stats[i][cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width / height, height / width)
        if area < 1000 or aspect_ratio > 5:
            mask[labels == i] = 0
    return mask


def compute_volume(depth, dem_scale, pixel_size_m=0.03, depth_size=0.03):
    """Compute excavation volume in cubic meters."""
    pixel_area_m2 = (dem_scale * pixel_size_m) ** 2
    depth_m = depth * depth_size
    volume_m3 = np.sum(depth_m) * pixel_area_m2
    print(f"Excavation volume: {volume_m3:.2f} cubic meters")
    return volume_m3


def predict_img(net, data, sam_predictor, device, fold_dir, root_dir, exp_name):
    net.eval()

    image_name = data["image_name"][0]
    save_dir = root_dir / f"logs/results/{exp_name}/{fold_dir}/{image_name}"
    os.makedirs(save_dir, exist_ok=True)

    image = data["image_dem_norm"]
    mask_true = data["true_mask"]
    valid_mask = data["valid_mask"]
    image_pad = data["image_pad"]
    elev_valid_mask = data["elev_valid_mask"]

    image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    mask_true_orig = mask_true.to(device=device, dtype=torch.long)
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    unlabeled = (mask_true != 1).unsqueeze(1)

    if sam_predictor:
        with torch.no_grad():
            image_pad = image_pad.to(device=device, dtype=torch.uint8)
            cv2.imwrite(f"{save_dir}/image_pad.png", image_pad[0].detach().cpu().numpy())
            cv2.imwrite(f"{save_dir}/mask_true.png", mask_true_orig[0].detach().cpu().numpy() * 255)

            sam_logits_pred = torch.zeros(
                (image_pad.shape[0], 2, image_pad.shape[1], image_pad.shape[2]),
                device=device,
            )

            for idx in range(image_pad.shape[0]):
                elev_valid_mask_256 = F.interpolate(
                    elev_valid_mask[0].float().unsqueeze(0).unsqueeze(0),
                    size=(256, 256),
                    mode="nearest",
                )
                elev_valid_mask_256 = (
                    elev_valid_mask_256.squeeze(0).detach().cpu().numpy() * 255
                )

                sam_predictor.set_image(image_pad[idx].detach().cpu().numpy())
                sam_logits, _, _ = sam_predictor.predict(
                    mask_input=elev_valid_mask_256,
                    multimask_output=False,
                    return_logits=True,
                )

                sam_pred_mask = sam_logits > 0.0
                sam_binary_mask = (sam_pred_mask.astype(np.uint8) * 255)
                cv2.imwrite(
                    f"{save_dir}/{image_name}_metrics_sam_pred_mask1.png",
                    sam_binary_mask.squeeze(),
                )

                sam_unlabeled_mask = (
                    sam_binary_mask.squeeze()
                    * unlabeled[idx].squeeze().detach().cpu().numpy().astype(np.uint8)
                )
                cv2.imwrite(
                    f"{save_dir}/{image_name}_metrics_sam_unlabeled_mask1.png",
                    sam_unlabeled_mask,
                )

                sam_prob_foreground = torch.sigmoid(torch.tensor(sam_logits, device=device))
                sam_prob_background = 1.0 - sam_prob_foreground
                sam_probs = torch.cat([sam_prob_background, sam_prob_foreground], dim=0)
                sam_probs_logits = torch.log(sam_probs + 1e-8)
                sam_probs_logits = torch.where(
                    valid_mask[idx] > 0,
                    sam_probs_logits,
                    torch.zeros_like(sam_probs_logits),
                )
                sam_logits_pred[idx] = sam_probs_logits

    with torch.no_grad():
        if sam_predictor:
            mask_pred_logits, unet_logits = net(image, sam_logits_pred)
            unet_logits = torch.where(
                valid_mask.unsqueeze(1) > 0,
                unet_logits,
                torch.zeros_like(unet_logits),
            )
            unet_pred = unet_logits.argmax(dim=1)
            cleaned_unet_np = clean_mask(unet_pred)
        else:
            mask_pred_logits = net(image)
            cleaned_unet_np = None

        mask_pred_logits = torch.where(
            valid_mask.unsqueeze(1) > 0,
            mask_pred_logits,
            torch.zeros_like(mask_pred_logits),
        )
        mask_pred = mask_pred_logits.argmax(dim=1)

        dice_score, mean_iou, pixel_accuracy = compute_metrics(mask_pred, mask_true_orig)
        print(
            f"{image_name} dice score: {dice_score:.4f}, "
            f"mIoU: {mean_iou:.4f}, Pixel Acc: {pixel_accuracy:.4f}"
        )

        cleaned_mask_np = clean_mask(mask_pred)
        cleaned_mask = torch.from_numpy(cleaned_mask_np).unsqueeze(0).to(device)

        dice_score_clean, mean_iou_clean, pixel_accuracy_clean = compute_metrics(
            cleaned_mask,
            mask_true_orig,
        )
        print(
            f"{image_name} (cleaned) dice score: {dice_score_clean:.4f}, "
            f"mIoU: {mean_iou_clean:.4f}, Pixel Acc: {pixel_accuracy_clean:.4f}"
        )

    dem = data["dem_pad"][0].cpu().numpy()
    dem_scale = data["dem_scale"][0].item()
    dem_excavation = dem[cleaned_mask_np > 0]
    dem_excavation_pic = dem * cleaned_mask_np
    out_filename_dem = f"{save_dir}/{image_name}_dem_excavation.png"
    cv2.imwrite(out_filename_dem, dem_excavation_pic)
    volume = compute_volume(dem_excavation, dem_scale)

    with open(f"{save_dir}/{image_name}_metrics.txt", "w") as f:
        f.write(f"Dice Score: {dice_score:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Pixel Accuracy: {pixel_accuracy:.4f}\n")
        f.write(f"Dice Score (cleaned): {dice_score_clean:.4f}\n")
        f.write(f"Mean IoU (cleaned): {mean_iou_clean:.4f}\n")
        f.write(f"Pixel Accuracy (cleaned): {pixel_accuracy_clean:.4f}\n")
        f.write(f"Excavation Volume (cubic meters): {volume:.2f}\n")

    out_filename = f"{save_dir}/{image_name}_mask_pred.png"
    cv2.imwrite(out_filename, cleaned_mask_np.astype(np.uint8) * 255)

    if cleaned_unet_np is not None:
        out_unet_filename = f"{save_dir}/{image_name}_unet_pred.png"
        cv2.imwrite(out_unet_filename, cleaned_unet_np.astype(np.uint8) * 255)

    unlabeled_mask_np = (
        mask_pred.squeeze().detach().cpu().numpy()
        * unlabeled[0].squeeze().detach().cpu().numpy().astype(np.uint8)
        * 255
    )
    cv2.imwrite(f"{save_dir}/{image_name}_unlabeled_mask_pred.png", unlabeled_mask_np)


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model",
        "-m",
        default="./checkpoints/fold_0/checkpoint_epoch10.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument("--input", "-i", metavar="INPUT", nargs="+", help="Filenames of input images")
    parser.add_argument("--output", "-o", metavar="OUTPUT", nargs="+", help="Filenames of output images")
    parser.add_argument("--viz", "-v", action="store_true", help="Visualize the images as they are processed")
    parser.add_argument("--no-save", "-n", action="store_true", help="Do not save the output masks")
    parser.add_argument(
        "--mask-threshold",
        "-t",
        type=float,
        default=0.5,
        help="Minimum probability value to consider a mask pixel white",
    )
    parser.add_argument("--target_size", "-s", type=int, default=1024, help="Scale factor for the input images")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes")

    parser.add_argument("--exp_name", type=str, default="baseline", help="Name of the experiment")
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Number of input channels to the model, e.g., 4 for RGB + DEM, 3 for RGB",
    )
    parser.add_argument("--use_sam", type=str2bool, default=False, help="Whether to use SAM model")
    parser.add_argument("--sam_model_type", type=str, default="vit_l", help="Type of SAM model to use")
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="model_weights/SAM/" + "sam_vit_l_0b3195.pth",
        help="Path to SAM model checkpoint",
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="add",
        choices=["add", "csaf", "none"],
        help="Method to fuse SAM and UNet features",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "dice", "cross_entropy_dice", "tcs"],
        help="Loss function used during training",
    )
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for final output GT loss")
    parser.add_argument("--lambda2", type=float, default=0.5, help="Weight for UNet intermediate GT loss")
    parser.add_argument("--lambda3", type=float, default=0.5, help="Weight for final output SAM loss")
    parser.add_argument("--threshold_m", type=float, default=5.91, help="Elevation threshold in meters")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    root_dir = Path(__file__).resolve().parent
    args.sam_checkpoint = root_dir / "model_weights/SAM" / "sam_vit_l_0b3195.pth"
    img_dir = root_dir / "data" / "Earthwork" / "imgs"
    mask_dir = root_dir / "data" / "Earthwork" / "masks"
    dem_dir = root_dir / "data" / "Earthwork" / "dems"

    dataset = EarthworkDataset(
        img_dir,
        mask_dir,
        dem_dir,
        target_size=args.target_size,
        in_channels=args.in_channels,
        threshold_m=args.threshold_m,
    )

    predictor = None
    if args.use_sam:
        logging.info("Loading SAM model")
        sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(device)
        predictor = SamPredictor(sam)

    checkpoints_dir = args.exp_name
    all_fold_checkpoints = os.listdir(root_dir / f"{checkpoints_dir}")

    for fold_dir in all_fold_checkpoints:
        logging.info(f"Processing {fold_dir} ...")
        args.model = root_dir / f"{checkpoints_dir}/{fold_dir}/checkpoint_epoch.pth"
        fold_split_file = os.path.dirname(args.model) + "/fold_split.txt"

        with open(fold_split_file, "r") as f:
            lines = f.readlines()
            val_idx = eval(lines[-1].strip().split(" ", 2)[-1])

        val_set = torch.utils.data.Subset(dataset, val_idx)
        loader_args = dict(batch_size=1, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

        net = UNet(
            n_channels=args.in_channels,
            n_classes=args.classes,
            bilinear=args.bilinear,
            use_sam=args.use_sam,
        )

        net.to(device=device)
        state_dict = torch.load(args.model, map_location=device)
        state_dict.pop("mask_values", [0, 1])
        net.load_state_dict(state_dict)

        logging.info("Model loaded!")

        for idx, data in enumerate(val_loader):
            logging.info(f"Predicting image {idx} ...")
            predict_img(
                net=net,
                data=data,
                sam_predictor=predictor if args.use_sam else None,
                device=device,
                fold_dir=fold_dir,
                root_dir=root_dir,
                exp_name=checkpoints_dir,
            )
