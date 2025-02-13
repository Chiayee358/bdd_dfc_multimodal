import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def load_checkpoint(snap_to_load, model, weights_dir="./", verbose=False):
    """

    Args:
        checkpoint (path/str): Path to saved torch model
        model (object): torch model

    Returns:
        _type_: _description_
    """
    fn_model = os.path.join(weights_dir, snap_to_load)
    checkpoint = torch.load(fn_model, map_location="cpu")
    loaded_dict = checkpoint["state_dict"]
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    if verbose:
        print(
            "Loaded checkpoint: {} (Epoch={}, Score={:.3f})".format(
                Path(snap_to_load).name, checkpoint["epoch"], checkpoint["best_score"]
            )
        )
    return model, checkpoint["best_score"]


def progress(train_logs, valid_logs, nepochs, outdir, fn_out):
    loss_t = [dic["Loss"] for dic in train_logs]
    loss_v = [dic["Loss"] for dic in valid_logs]
    score_t = [dic["Score"] for dic in train_logs]
    score_v = [dic["Score"] for dic in valid_logs]

    epochs = range(0, len(score_t))
    plt.figure(figsize=(5, 12))

    # Train and validation metric
    # ---------------------------
    plt.subplot(2, 1, 1)
    label = f"Train score={max(score_t):6.4f} in Epoch={np.argmax(score_t)}"
    plt.plot(epochs, score_t, "b", label=label)
    label = f"Valid score={max(score_v):6.4f} in Epoch={np.argmax(score_v)}"
    plt.plot(epochs, score_v, "r", label=label)
    plt.title("Training and Validation Metric")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()

    # Train and validation loss
    # -------------------------
    plt.subplot(2, 1, 2)
    ymax = max(max(loss_t), max(loss_v))
    ymin = min(min(loss_t), min(loss_v))
    ymax = 1 if ymax <= 1 else ymax + 0.5
    ymin = 0 if ymin <= 0.5 else ymin - 0.5

    label = f"Train loss={min(loss_t):6.4f} in Epoch:{np.argmin(loss_t)}"
    plt.plot(epochs, loss_t, "b", label=label)
    label = f"Valid loss={min(loss_v):6.4f} in Epoch:{np.argmin(loss_v)}"
    plt.plot(epochs, loss_v, "r", label=label)

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel("Loss")
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig(f"{outdir}/{fn_out}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
    return
