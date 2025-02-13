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

    DATA_DIR = "/home/bruno/Dropbox/DATASETS/dfc25_track2_trainval/train"
    CLASSES = [0, 1, 2, 3]
    LR = 1e-4
    IMG_SIZE = 512
    BATCH_SIZE = 12
    NUM_EPOCHS = 150
    WEIGHTS_DIR = "weights_segformer"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------
    # --- split training and validation sets ---
    # -------------------------------------------
    fn_list = [f for f in Path(DATA_DIR).rglob("*.tif") if "post-event" in str(f)]

    train_list = []
    valid_list = []
    train_list, valid_list = train_test_split(fn_list, test_size=0.2, random_state=seed)

    # ---------------------------
    # --- Define data loaders ---
    # ---------------------------
    train_data = src.dataset.DFC25(
        train_list,
        img_size=IMG_SIZE,
        classes=CLASSES,
        augm=src.transforms.train_augm_comp,
    )
    valid_data = src.dataset.DFC25(
        valid_list,
        img_size=IMG_SIZE,
        classes=CLASSES,
        augm=src.transforms.valid_augm,
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
    # ---------------------
    model = smp.SegformerSiamese(
        classes=len(CLASSES),
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        activation=None,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    criterion = src.losses.JaccardMCCFocalLoss()

    # ------------------------
    # --- Network training ---
    # ------------------------
    snapshot_name = "{}_b{}_e{}_s{}_{}".format(
        model.name,
        BATCH_SIZE,
        NUM_EPOCHS,
        seed,
        criterion.name,
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
        train_logs = src.runners.train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=device,
        )

        torch.cuda.empty_cache()
        valid_logs = src.runners.valid_epoch(
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
