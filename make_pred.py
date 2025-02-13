import os
import shutil
import numpy as np
import torch
import src
import warnings
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # -----------------------
    # --- Main parameters ---
    # -----------------------
    DATA_DIR = "/home/bruno/Dropbox/DATASETS/dfc25_track2_trainval/val"
    CLASSES = [0, 1, 2, 3]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # --- read testing dataset ---
    # ----------------------------
    fn_list = [f for f in Path(DATA_DIR).rglob("*.tif") if "post-event" in str(f)]
    print("Num. of testing files:", len(fn_list))

    checkpoint = (
        "weights/segformersiamese-efficientnet-b4_b10_e100_s200_jaccardmccfocal.pth"
    )

    # model = smp.UnetSiamese(
    #     classes=len(CLASSES),
    #     encoder_name="efficientnet-b4",
    #     encoder_weights="imagenet",
    #     activation=None,
    #     decoder_attention_type="scse",
    # )

    model = smp.SegformerSiamese(
        classes=4,
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        activation=None,
    )

    model, *_ = src.utils.load_checkpoint(checkpoint, model, verbose=True)
    model = model.eval().to(device)

    outdir = os.path.join("predictions", "{}".format(checkpoint.split("/")[-1][:-4]))
    os.makedirs(outdir, exist_ok=True)

    for idx in tqdm(range(len(fn_list))):
        fout = Path(
            str(fn_list[idx])
            .replace("post-event", "target")
            .replace("post_disaster", "building_damage")
        )

        # read SAR and RGB images
        sar = src.dataset.load_grayscale(fn_list[idx])
        rgb = src.dataset.load_multiband(
            str(fn_list[idx])
            .replace("post-event", "pre-event")
            .replace("post_disaster", "pre_disaster")
        )

        # compose the input image
        img = (
            np.moveaxis(
                np.concatenate([np.stack([sar, sar, sar], axis=-1), rgb], axis=-1),
                -1,
                0,
            )
            / 255.0
        )

        imgs = []
        imgs.append(img.copy())
        imgs.append(img[:, :, ::-1].copy())
        imgs.append(img[:, ::-1, :].copy())
        imgs.append(img[:, ::-1, ::-1].copy())
        tensor = (
            torch.cat([torch.from_numpy(x).unsqueeze(0) for x in imgs], dim=0)
            .float()
            .to(device)
        )

        pred = []
        with torch.no_grad():
            *_, msk = model.forward(tensor)
            msk = torch.softmax(msk[:, :, ...], dim=1).cpu().numpy()
            pred.append(msk[0, :, :, :])
            pred.append(msk[1, :, :, ::-1])
            pred.append(msk[2, :, ::-1, :])
            pred.append(msk[3, :, ::-1, ::-1])
        pred = np.asarray(pred).mean(axis=0).argmax(axis=0)

        Image.fromarray(pred.astype("uint8")).save(
            os.path.join(outdir, f"{fout.name[:-4]}.png")
        )

    shutil.make_archive(outdir, "zip", outdir)
    # shutil.rmtree(outdir)
