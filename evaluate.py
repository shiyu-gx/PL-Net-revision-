import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, sam_predictor=None):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(
            dataloader,
            total=num_val_batches,
            desc="Validation round",
            unit="batch",
            leave=False,
        ):
            image = batch["image_dem_norm"]
            mask_true = batch["true_mask"]
            valid_mask = batch["valid_mask"]
            image_pad = batch["image_pad"]
            elev_valid_mask = batch["elev_valid_mask"]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)

            if sam_predictor:
                with torch.no_grad():
                    image_pad = image_pad.to(device=device, dtype=torch.uint8)
                    sam_logits_pred = torch.zeros(
                        (image_pad.shape[0], 2, image_pad.shape[1], image_pad.shape[2]),
                        device=device,
                    )  # [B, 2, H, W]

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

                        sam_prob_foreground = torch.sigmoid(
                            torch.tensor(sam_logits, device=device)
                        )
                        sam_prob_background = 1.0 - sam_prob_foreground
                        sam_probs = torch.cat(
                            [sam_prob_background, sam_prob_foreground], dim=0
                        )  # (2, H, W)
                        sam_probs_logits = torch.log(sam_probs + 1e-8)
                        sam_probs_logits = torch.where(
                            valid_mask[idx] > 0,
                            sam_probs_logits,
                            torch.zeros_like(sam_probs_logits),
                        )
                        sam_logits_pred[idx] = sam_probs_logits

            # predict the mask
            if sam_predictor:
                mask_pred, unet_logits = net(image, sam_logits_pred)
            else:
                mask_pred = net(image)

            # apply valid_mask to ignore invalid regions in the evaluation
            mask_pred = torch.where(
                valid_mask.unsqueeze(1) > 0,
                mask_pred,
                torch.zeros_like(mask_pred),
            )

            if net.n_classes == 1:
                assert (
                    mask_true.min() >= 0 and mask_true.max() <= 1
                ), "True mask indices should be in [0, 1]"
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert (
                    mask_true.min() >= 0 and mask_true.max() < net.n_classes
                ), "True mask indices should be in [0, n_classes)"
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(
                    0, 3, 1, 2
                ).float()
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False
                )

    net.train()

    if num_val_batches == 0:
        return dice_score

    return dice_score / num_val_batches
