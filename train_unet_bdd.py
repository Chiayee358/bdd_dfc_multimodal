
import os
import warnings
import json
import time
import numpy as np
import random
import torch
import argparse
import src
from pathlib import Path
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp


warnings.filterwarnings("ignore")

import os
from glob import glob

# Define the directories for each split
# DATA_DIR = "BDD_dataset"
DATA_DIR = "BDD_dataset_GAN-whole"


train_list = list(Path(DATA_DIR, "train", "post_disaster").rglob("*.tif"))
valid_list = list(Path(DATA_DIR, "val", "post_disaster").rglob("*.tif"))

print("Number of training samples:", len(train_list))
print("Number of validation samples:", len(valid_list))

# # Define base directories for each split
# base_dirs = {
#     'train': '/home/chia/bdd_multimodal-main/BDD_dataset/train',
#     'val': '/home/chia/bdd_multimodal-main/BDD_dataset/val',
#     'test': '/home/chia/bdd_multimodal-main/BDD_dataset/test'
# }

# # Function to get file lists for each subfolder in a given split
# def get_split_data(split_dir):
#     pre_disaster = sorted(glob(os.path.join(split_dir, 'pre_disaster', '*')))
#     post_disaster = sorted(glob(os.path.join(split_dir, 'post_disaster', '*')))
#     target = sorted(glob(os.path.join(split_dir, 'target', '*')))
#     return pre_disaster, post_disaster, target

# # Load data for each split
# train_pre, train_post, train_target = get_split_data(base_dirs['train'])
# val_pre, val_post, val_target = get_split_data(base_dirs['val'])
# test_pre, test_post, test_target = get_split_data(base_dirs['test'])

# # Print out the number of samples in each category
# print(f"Training samples: {len(train_pre)}")
# print(f"Validation samples: {len(val_pre)}")
# print(f"Testing samples: {len(test_pre)}")


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--encoder", type=str, default="efficientnet-b4")
    args = parser.parse_args()

    seed = args.seed
    encoder_name = args.encoder
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -----------------------
    # --- Main parameters ---
    # -----------------------

    # DATA_DIR = "/home/bruno/Dropbox/DATASETS/dfc25_track2_trainval/train"
    # DATA_DIR = "/home/chia/bdd_multimodal-main/BDD_dataset_GAN/train"
    DATA_DIR = "/home/chia/bdd_multimodal-main/BDD_dataset_GAN-whole/train"
    CLASSES = [0, 1, 2, 3]
    LR = 1e-4
    IMG_SIZE = 512
    BATCH_SIZE = 6
    NUM_EPOCHS = 180

    CLS_WEIGHTS = [0.0258, 0.373, 0.398, 0.202]
    # CLS_WEIGHTS = [0.0048, 0.0302, 0.5828, 0.3822] #dfc25
    WEIGHTS_DIR = "weights_unet"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------
    # --- split training and validation sets ---
    # -------------------------------------------
    
    # fn_list = [f for f in Path(DATA_DIR).rglob("*.tif") if "post-event" in str(f)]

    # train_list = []
    # valid_list = []
    # train_list, valid_list = train_test_split(fn_list, test_size=0.2, random_state=seed)

    # ---------------------------
    # --- Define data loaders ---
    # ---------------------------
    train_data = src.dataset.BDD3(
        train_list,
        img_size=IMG_SIZE,
        classes=CLASSES,
        augm=src.transforms.train_augm_comp2,
    )
    valid_data = src.dataset.BDD3(
        valid_list,
        img_size=IMG_SIZE,
        classes=CLASSES,
        augm=src.transforms.valid_augm2,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=False,
        pin_memory=True,
    )

    # ---------------------
    # --- network setup ---
    # # ---------------------
    # model = smp.UnetSiamese(

    #     classes=len(CLASSES),
    #     encoder_name=encoder_name,
    #     encoder_weights="imagenet",
    #     activation=None,
    #     decoder_attention_type="scse",
    # )
    
    model = smp.UnetSiamese(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=4,  # SAR + Optical
        classes=len(CLASSES),  # Example: 5 segmentation classes
        activation=None,
        decoder_attention_type="scse",
    )

#     model = smp.SegmentationModelSiamese(
#     classes=len(CLASSES),
#     encoder_name=encoder_name,
#     encoder_weights="imagenet",
#     activation=None,
#     decoder_attention_type="scse",
# )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    criterion = [
        src.losses.BCEWithLogits(),
        # src.losses.CEWithLogits(weight=np.asarray(CLS_WEIGHTS)),
        src.losses.JaccardMCCLoss(weights=np.asarray(CLS_WEIGHTS)),
    ]

    # ------------------------
    # --- Network training ---
    # ------------------------
    snapshot_name = "{}_b{}_e{}_s{}_{}".format(
        model.name,
        BATCH_SIZE,
        NUM_EPOCHS,
        seed,
        "_".join([x.name for x in criterion]),
    ).lower()

    print("Train samples  :", len(train_list))
    print("Valid samples  :", len(valid_list))
    print("Global seed    :", seed)
    print("Batch size     :", BATCH_SIZE)
    print("Network        :", snapshot_name)
    print("Num of epochs  :", NUM_EPOCHS)
    print("Learning rate  :", LR)

    n_batch = 0
    max_score = 0
    train_hist = []
    valid_hist = []
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch + 1}")

        torch.cuda.empty_cache()
        train_logs = src.runners.train_epoch2(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=device,
        )

        torch.cuda.empty_cache()
        valid_logs = src.runners.valid_epoch2(
            model=model,
            criterion=criterion,
            dataloader=valid_data_loader,
            device=device,
        )

        train_hist.append(train_logs)
        valid_hist.append(valid_logs)
        src.utils.progress(
            train_hist,
            valid_hist,
            NUM_EPOCHS,
            WEIGHTS_DIR,
            snapshot_name,
        )
        with open(os.path.join(WEIGHTS_DIR, f"{snapshot_name}.json"), "w") as fjson:
            json.dump({"train": train_hist, "valid": valid_hist}, fjson, indent=4)

        score = valid_logs["Score"]
        if max_score < score:
            max_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_score": max_score,
                },
                os.path.join(WEIGHTS_DIR, f"{snapshot_name}.pth"),
            )
            print("model saved")

    print("Time: {:.3f} min".format((time.time() - start) / 60.0))

# %%
