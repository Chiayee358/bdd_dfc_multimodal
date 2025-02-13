
#%%
import warnings
import src
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from pathlib import Path
import os
from torch.utils.data import DataLoader
from src.dataset import BDD3  # Ensure this matches your dataset class

warnings.filterwarnings("ignore")

# Load model

model = smp.UnetSiamese(
    classes=4,  # Ensure this matches training
    # encoder_name='efficientnet-b4',  # Must match what was used during training
    encoder_name='resnet34',  # Must match what was used during training

    encoder_weights=None,  # Since we're loading custom weights, don't use pre-trained weights
    activation=None,
    decoder_attention_type="scse",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
checkpoint_path = "/home/chia/bdd_multimodal-main/weights_unet/unetsiamese-resnet34_b6_e200_s200_bceloss_jaccardmccweighted.pth" #0.58
# checkpoint_path = "/home/chia/bdd_multimodal-main/weights_unet/unetsiamese-efficientnet-b4_b6_e210_s200_bceloss_jaccardmccweighted.pth" #0.43
# checkpoint_path = "/home/chia/bdd_multimodal-main/weights_unet/unetsiamese-resnet34_b6_e210_s200_bceloss_jaccardmccweighted.pth"
checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])

model = model.to(device).eval()

dataset = "test"
data_dir = f"/home/chia/bdd_multimodal-main/BDD_dataset_GAN/{dataset}"
fn_list = [str(f) for f in Path(data_dir).rglob("*.tif") if "post_disaster" in str(f)]
print(f"Making predictions on {len(fn_list)} samples, using {dataset}")

def compute_mean_iou(pred, target, num_classes, ignore_empty=True):
    iou_list = [
        (np.logical_and(pred == cls, target == cls).sum() / np.logical_or(pred == cls, target == cls).sum())
        if np.logical_or(pred == cls, target == cls).sum() > 0 else np.nan
        for cls in range(num_classes)
    ]
    return np.nanmean(iou_list), iou_list

# Prepare to store results
predictions, indices, iou_scores, per_class_ious = [], [], [], []
num_classes = 4  # Adjust if needed

for i, file_path in enumerate(fn_list):
    print(f"Processing: {file_path}")

    # --- Load Post-Disaster SAR (Grayscale) ---
    sar_post = src.dataset.load_grayscale(file_path)  # Post-SAR

    sar_post = np.stack([sar_post] * 4, axis=0)  # Convert from (H, W) → (4, H, W)

    # --- Load Pre-Disaster Optical (Multiband) ---
    optical_path = file_path.replace("post_disaster", "pre_disaster").replace(
        "_pre_disaster_sar.tif", "_pre_disaster.tif"
    )  # Optical
    optical = src.dataset.load_multiband(optical_path)
    
        # Ensure Optical Image is RGB (3 channels)
    if optical.ndim == 2:  # If grayscale (H, W), convert to (3, H, W)
        optical = np.stack([optical] * 3, axis=0)
    elif optical.shape[-1] == 3:  # If (H, W, 3), convert to (3, H, W)
        optical = np.transpose(optical, (2, 0, 1))

    # --- Load Pre-Disaster SAR (Grayscale) ---
    sar_pre_path = file_path.replace("post_disaster", "pre_disaster").replace(
        "_post_disaster_sar", "_pre_disaster_sar"
    )  # Pre-SAR
    sar_pre = src.dataset.load_grayscale(sar_pre_path)
    
    if sar_pre.ndim == 2:  # If (H, W), convert to (1, H, W)
        sar_pre = np.expand_dims(sar_pre, axis=0)


    # --- Load Ground Truth (Building Damage) ---
    gtd_path = file_path.replace("post_disaster", "target").replace(
        "target", "building_damage"
    ).replace("_sar", "")  # Ground truth
    gtd = src.dataset.load_grayscale(gtd_path)
    
    num_classes = 4  # {0, 1, 2, 3}

    gtd_loc = (gtd > 0).astype("uint8")  # 1 if building exists, else 0
    gtd_seg = torch.tensor(gtd, dtype=torch.long)  # Keep as class labels (512, 512)

    gtd_seg = gtd_seg.unsqueeze(0).to(device)


    
    print(f"🔹 Optical Tensor Shape Before Model: {optical.shape}")  # Expected: (B, 3, H, W)
    print(f"🔹 SAR Pre Shape Before Model: {sar_pre.shape}")  # Expected: (B, 1, H, W)
    print(f"🔹 SAR Post Shape Before Model: {sar_post.shape}")  # Expected: (B, 4, H, W)
    print(f"🔹 gtd_seg Shape Before Squeeze: {gtd_seg.shape}")



    optical = torch.tensor(optical, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
    sar_pre = torch.tensor(sar_pre, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
    sar_post = torch.tensor(sar_post, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
    
        # Ensure Batch Dimension


    with torch.no_grad():
        loc_out, dmg_out = model(optical=optical, sar_pre=sar_pre, sar_post=sar_post)  # Corrected
        seg_prob = torch.softmax(dmg_out, dim=1).cpu().numpy()  # Use `dmg_out`

    final_pred = seg_prob.argmax(axis=1)[0]  # Convert probabilities to class labels

    # Compute IoU
    gtd_seg_np = gtd_seg.squeeze(0).cpu().numpy()
    mean_iou, per_class_iou = compute_mean_iou(final_pred, gtd_seg_np, num_classes)

    print(f"Image {i} - Mean IoU: {mean_iou:.3f}, Per-class IoU: {per_class_iou}")
    iou_scores.append(mean_iou)
    per_class_ious.append(per_class_iou)
    predictions.append(final_pred)
    indices.append(i)
print(f"🔹 Unique values in final_pred: {np.unique(final_pred)}")  # Expected: {0,1,2,3}

# Compute overall mean IoU
print("Overall Mean IoU:", np.nanmean(iou_scores))

save_dir = "/home/chia/bdd_multimodal-main/predictions_GAN_unetsiamese-resnet34_b6_e200_s200_bceloss_jaccardmccweighted"
os.makedirs(save_dir, exist_ok=True)
for pred, idx in zip(predictions, indices):
    # pred_uint8 = (pred * (255 / (num_classes - 1))).astype(np.uint8)  # Scale to 0-255
    filename = os.path.join(save_dir, f"pred_{idx}.png")
    cv2.imwrite(filename, pred)

print("All predictions saved successfully!")


# %%
