import numpy as np
import cv2
import albumentations as A
import torchvision.transforms.functional as TF
import torch

# class ToTensor:
#     def __init__(self, classes):
#         self.classes = classes

#     def __call__(self, sample):
#         msks = [(sample["gtd"] == v) * 1 for v in self.classes]
#         return (
#             # TF.to_tensor((sample["sar"] / 255.0).astype(np.float32)),
#             TF.to_tensor((sample["sar_post"] / 255.0).astype(np.float32)), ### added by chia
#             TF.to_tensor((sample["optical"] / 255.0).astype(np.float32)), ### added by chia

#             # TF.to_tensor((sample["rgb"] / 255.0).astype(np.float32)),
#             TF.to_tensor(np.stack(msks, axis=-1)),
#         )

#####################################
########### added by chia ###########
#####################################
# class ToTensor:
#     def __init__(self, classes):
#         self.classes = classes

#     def __call__(self, sample):
#         msks = [(sample["gtd"] == v) * 1 for v in self.classes]
#         return {
#             "sar_post": TF.to_tensor((sample["sar_post"] / 255.0).astype(np.float32)),
#             "optical": TF.to_tensor((sample["optical"] / 255.0).astype(np.float32)),
#             "sar_pre": TF.to_tensor((sample["sar_pre"] / 255.0).astype(np.float32)),
#             "gtd": TF.to_tensor(np.stack(msks, axis=-1)),
#         }

# class ToTensor:
#     def __init__(self, classes):
#         self.classes = classes  # This might be ignored if it's binary

#     def __call__(self, sample):
#         # Convert ground truth to a binary mask. For instance, consider any pixel > 0 as positive.
#         mask = (sample["gtd"] > 0).astype(np.float32)  # shape: (H, W)
#         mask = np.expand_dims(mask, axis=-1)            # shape: (H, W, 1)
#         return {
#             "sar_post": TF.to_tensor((sample["sar_post"] / 255.0).astype(np.float32)),
#             "optical": TF.to_tensor((sample["optical"] / 255.0).astype(np.float32)),
#             "sar_pre": TF.to_tensor((sample["sar_pre"] / 255.0).astype(np.float32)),
#             "gtd": TF.to_tensor(mask)
#         }
        


# class ToTensor:
#     def __init__(self, classes):
#         self.classes = classes

#     def __call__(self, sample):
#         # msks = [(sample["gtd"] == v) * 1 for v in self.classes]
#         msks = [(sample["gtd"] == v) * 1 for v in self.classes]
#         TF.to_tensor(np.stack(msks, axis=-1))


#         # This produces an array of shape (H, W, num_classes)
#         # TF.to_tensor converts it to (num_classes, H, W)
#         return {
#             "sar_post": TF.to_tensor((sample["sar_post"] / 255.0).astype(np.float32)),
#             "optical": TF.to_tensor((sample["optical"] / 255.0).astype(np.float32)),
#             "sar_pre": TF.to_tensor((sample["sar_pre"] / 255.0).astype(np.float32)),
#             "gtd": TF.to_tensor(np.stack(msks, axis=-1)),
#         }


class ToTensor:
    def __init__(self, classes):
        self.classes = classes  # e.g. [0, 1, 2, 3] for 4 classes

    def __call__(self, sample):
        # One-hot encode for segmentation
        msks = [(sample["gtd"] == v) * 1 for v in self.classes]
        gtd_seg = TF.to_tensor(np.stack(msks, axis=-1))  # shape: (num_classes, H, W)

        # For localization: create a binary mask (assume any non-zero value is foreground)
        gtd_loc_np = (sample["gtd"] > 0).astype(np.float32)  # shape: (H, W)
        # Expand dims so that the channel is last
        gtd_loc_np = np.expand_dims(gtd_loc_np, axis=-1)  # shape: (H, W, 1)
        gtd_loc = TF.to_tensor(gtd_loc_np)  # shape: (1, H, W)

        return {
            "sar_post": TF.to_tensor((sample["sar_post"] / 255.0).astype(np.float32)),
            "optical": TF.to_tensor((sample["optical"] / 255.0).astype(np.float32)),
            "sar_pre": TF.to_tensor((sample["sar_pre"] / 255.0).astype(np.float32)),
            "gtd_loc": gtd_loc,
            "gtd_seg": gtd_seg
        }




def valid_augm2(sample, size=512):
    augms = A.Compose(
        [
            A.Resize(height=size, width=size, interpolation=cv2.INTER_NEAREST, p=1.0)
        ],
        additional_targets={
            "optical": "image",
            "sar_post": "image",
            "sar_pre": "image"
        },
    )
    # Choose which image to use as the primary "image" for augmentation.
    # For example, if you want to use the optical image as primary:
    return augms(
        image=sample["optical"],
        optical=sample["optical"],
        sar_post=sample["sar_post"],
        sar_pre=sample["sar_pre"],
        mask=sample["gtd"]
    )

def valid_augm(sample, size=512):
    augms = A.Compose(
        [A.Resize(height=size, width=size, interpolation=cv2.INTER_NEAREST, p=1.0)],
        additional_targets={"image1": "image"},
    )
    return augms(image=sample["sar"], image1=sample["rgb"], mask=sample["gtd"])


def train_augm(sample, size=512):
    augms = A.Compose(
        [
            A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
            A.RandomCrop(size, size, p=1.0),
        ],
        additional_targets={"image1": "image"},
    )
    return augms(image=sample["sar"], image1=sample["rgb"], mask=sample["gtd"])


def train_augm_comp(sample, size=512):
    augms = A.Compose(
        [
            A.Rotate(limit=(-45, 45), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.25),
            A.PadIfNeeded(size, size, border_mode=0, p=1.0),
            A.RandomCrop(size, size, p=1.0),
            A.RandomRotate90(p=0.5),
            # color transforms
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=(70, 130), p=1),
                    A.ChannelShuffle(p=0.2),
                    A.HueSaturationValue(
                        hue_shift_limit=30,
                        sat_shift_limit=40,
                        val_shift_limit=30,
                        p=1.0,
                    ),
                    A.RGBShift(
                        r_shift_limit=30,
                        g_shift_limit=30,
                        b_shift_limit=30,
                        p=1.0,
                    ),
                ],
                p=0.8,
            ),
            # noise transforms
            A.OneOf(
                [
                    A.GaussNoise(p=1),
                    A.MultiplicativeNoise(p=1),
                    A.Sharpen(p=1),
                    A.GaussianBlur(p=1),
                ],
                p=0.5,
            ),
        ],
        additional_targets={"image1": "image"},
    )
    return augms(image=sample["sar"], image1=sample["rgb"], mask=sample["gtd"])

#####apply to both sar and rgb###
####################################
# def train_augm_comp2(sample, size=512):
#     augms = A.Compose(
#         [
#             A.Rotate(limit=(-45, 45), p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.RandomScale(scale_limit=0.1, p=0.25),
#             A.PadIfNeeded(size, size, border_mode=0, p=1.0),
#             A.RandomCrop(size, size, p=1.0),
#             A.RandomRotate90(p=0.5),
#             A.OneOf(
#                 [
#                     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
#                     A.RandomGamma(gamma_limit=(70, 130), p=1),
#                     A.ChannelShuffle(p=0.2),
#                     A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0),
#                     A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
#                 ],
#                 p=0.8,
#             ),
#             A.OneOf(
#                 [
#                     A.GaussNoise(p=1),
#                     A.MultiplicativeNoise(p=1),
#                     A.Sharpen(p=1),
#                     A.GaussianBlur(p=1),
#                 ],
#                 p=0.5,
#             ),
#         ],
#         additional_targets={
#             "optical": "image",
#             "sar_pre": "image",
#             "sar_post": "image"  # This can also be the main image if you prefer
#         },
#     )

#     return augms(
#         image=sample["optical"],
#         optical=sample["optical"],
#         sar_pre=sample["sar_pre"],
#         sar_post=sample["sar_post"],
#         mask=sample["gtd"]
#     )
#########################################

def train_augm_comp2(sample, size=512):
    augms = A.Compose(
        [
            A.Rotate(limit=(-45, 45), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.25),
            A.PadIfNeeded(size, size, border_mode=0, p=1.0),
            A.RandomCrop(size, size, p=1.0),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.RandomGamma(gamma_limit=(70, 130), p=1),
                    A.ChannelShuffle(p=0.2),
                    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0),
                    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=1),
                    A.MultiplicativeNoise(p=1),
                    A.Sharpen(p=1),
                    A.GaussianBlur(p=1),
                ],
                p=0.5,
            ),
        ]
    )

    augmented = augms(image=sample["optical"])  # ✅ Apply augmentation **only to `optical`**
    
    # ✅ Replace only the optical image in `sample`
    sample["optical"] = augmented["image"]  

    return sample  # ✅ Return updated sample with augmented `optical`


