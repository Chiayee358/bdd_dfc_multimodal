import torch
import torch.nn as nn
from . import metrics


class JaccardMCCLoss(nn.Module):
    # class JaccardMCCFocalLoss(nn.Module):
    def __init__(self, weights=None, device="cuda"):
        super().__init__()
        self.name = "JaccardMCC" if weights is None else "JaccardMCCweighted"
        self.weights = (
            torch.from_numpy(weights).float().to(device)
            if weights is not None
            else torch.ones(100, device=device)
        )
        self.jaccard = JaccardLoss()
        self.mcc = MCCLoss()
        # self.focal = FocalLoss(logits=False)

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)

        losses = 0
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += (1 - metrics.fscore(ypr, ygt)) * self.weights[i]
            losses += self.mcc(ypr, ygt) * self.weights[i]
            # losses += self.focal(ypr, ygt) * self.weights[i]
        return losses


# -------------------
# --- JaccardLoss ---
# -------------------
class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "JaccardLoss"

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        losses = 0
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.fscore(ypr, ygt)
        return losses


# ----------------
# --- DiceLoss ---
# ----------------
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "DiceLoss"

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        losses = 0
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.iou(ypr, ygt)
        return losses


# ------------------------
# --- CEWithLogitsLoss ---
# ------------------------
class CEWithLogits(nn.Module):
    def __init__(self, weights=None, device="cuda"):
        super().__init__()
        self.weight = torch.from_numpy(weights).float() if weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=self.weight.to(device))
        self.name = "CELoss"

    def forward(self, input, target):
        loss = self.criterion(input, target.argmax(dim=1))
        return loss


class BCEWithLogits(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weight = torch.from_numpy(weights).float() if weights is not None else None
        # self.criterion = nn.BCEWithLogitsLoss(weights=self.weight)
        self.criterion = nn.BCEWithLogitsLoss(weight=self.weight)



        self.name = "BCELoss"

    def forward(self, input, target):
        loss = self.criterion(input, target)
        return loss


# ---------------
# --- MCCLoss ---
# ---------------
class MCCLoss(nn.Module):
    """
    Compute Matthews Correlation Coefficient Loss for image segmentation
    Reference: https://github.com/kakumarabhishek/MCC-Loss
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.name = "MCC"

    def forward(self, y_pred, y_true):
        bs = y_true.shape[0]

        y_pred = torch.sigmoid(y_pred)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp)
            * torch.add(tp, fn)
            * torch.add(tn, fp)
            * torch.add(tn, fn)
        )

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss


# -----------------
# --- FocalLoss ---
# -----------------
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=1,
        gamma=2,
        class_weights=None,
        logits=True,
        reduction="mean",
        label_smoothing=None,
    ):
        super().__init__()
        assert reduction in ["mean", None]
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.label_smoothing = label_smoothing
        self.name = "Focal"

    def forward(self, pr, gt):
        if self.logits:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                pr, gt, reduction="none"
            )
        else:
            bce_loss = nn.functional.binary_cross_entropy(pr, gt, reduction="none")

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()

        return focal_loss


# ----------------
# --- OHEMLoss ---
# ----------------
class OHEMBCELoss(nn.Module):
    """
    Taken but modified from:
    https://github.com/PkuRainBow/OCNet.pytorch/blob/master/utils/loss.py
    """

    def __init__(self, thresh=0.7, min_kept=10000):
        super(OHEMBCELoss, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.name = "OHEM"

    def forward(self, input, target):

        probs = torch.sigmoid(input)[:, 0, :, :].float()
        ygt = target[:, 0, :, :].float()

        # keep hard examples
        kept_flag = torch.zeros_like(probs).bool()
        # foreground pixels with low foreground probability
        kept_flag[ygt == 1] = probs[ygt == 1] <= self.thresh
        # background pixel with high foreground probability
        kept_flag[ygt == 0] = probs[ygt == 0] >= 1 - self.thresh

        if kept_flag.sum() < self.min_kept:
            # hardest examples have a probability closest to 0.5.
            # The network is very unsure whether they belong to the foreground
            # prob=1 or background prob=0
            hardest_examples = torch.argsort(
                torch.abs(probs - 0.5).contiguous().view(-1)
            )[: self.min_kept]
            kept_flag.contiguous().view(-1)[hardest_examples] = True
        return self.criterion(input[kept_flag, 0], target[kept_flag, 0])
