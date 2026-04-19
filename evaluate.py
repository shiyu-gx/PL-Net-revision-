import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import numpy as np

from src.utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, sam_predictor=None):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, valid_mask, image_pad, elev_valid_mask = batch['image_dem_norm'], batch['true_mask'], batch['valid_mask'], batch["image_pad"], batch["elev_valid_mask"]
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)

            if sam_predictor:
                with torch.no_grad():
                    image_pad = image_pad.to(device=device, dtype=torch.uint8)
                    sam_logits_pred = torch.zeros((image_pad.shape[0], 2, image_pad.shape[1], image_pad.shape[2])).to(device)  # [B, 2, H, W]
                    for idx in range(image_pad.shape[0]):  # 每个图像单独计算
                        elev_valid_mask_256 = F.interpolate(elev_valid_mask[0].float().unsqueeze(0).unsqueeze(0), size=(256, 256), mode='nearest')  # 将elev_valid_mask缩放回256*256，用于SAM输入
                        elev_valid_mask_256 = elev_valid_mask_256.squeeze(0).detach().cpu().numpy() * 255  # 必须乘以255，不然引导SAM分割的效果很差
                        sam_predictor.set_image(image_pad[idx].detach().cpu().numpy())  # 设置当前批次图像
                        sam_logits, _, _ = sam_predictor.predict(
                            mask_input=elev_valid_mask_256,
                            multimask_output=False,
                            return_logits=True
                        )

                        # 可视化 直接对SAM原始预测可视化
                        sam_pred_mask = sam_logits > 0.0
                        sam_binary_mask = (sam_pred_mask.astype(np.uint8) * 255)
                        cv2.imwrite("eval_sam_pred_mask1.png", sam_binary_mask.squeeze())

                        sam_prob_foreground = torch.sigmoid(torch.tensor(sam_logits, device=device))  # 前景概率 (1, H, W)
                        sam_prob_background = 1.0 - sam_prob_foreground  # 背景概率 = 1 - 前景概率 (1, H, W)
                        sam_probs = torch.cat([sam_prob_background, sam_prob_foreground], dim=0)  # 拼接前景和背景概率，shape: (2, H, W)
                        sam_probs_logits = torch.log(sam_probs + 1e-8)  # 转换为logits形式，shape: (2, H, W)
                        sam_probs_logits = torch.where(valid_mask[idx] > 0, sam_probs_logits, torch.zeros_like(sam_probs_logits))
                        sam_logits_pred[idx] = sam_probs_logits

            # predict the mask
            if sam_predictor:
                mask_pred, unet_logits = net(image, sam_logits_pred)  # 预测为概率值 [B, 2, H, W]
            else:
                mask_pred = net(image)  # 预测为概率值 [B, 2, H, W]
            # apply valid_mask to ignore invalid regions in the evaluation
            mask_pred = torch.where(valid_mask.unsqueeze(1) > 0, mask_pred, torch.zeros_like(mask_pred))  # 掩膜仅对有效区域计算loss
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
