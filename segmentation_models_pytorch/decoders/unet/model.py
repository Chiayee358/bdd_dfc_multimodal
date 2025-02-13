import torch
from typing import Any, Optional, Union, Callable, Sequence

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
    SegmentationModelSiamese,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading


from .decoder import UnetDecoder
import torch.nn as nn


# # *******************************************************************************
# # added by B. Adriano
# # *******************************************************************************
# class UnetSiamese(SegmentationModelSiamese):
#     """
#     U-Net is a fully convolutional neural network architecture designed for semantic image segmentation.

#     It consists of two main parts:

#     1. An encoder (downsampling path) that extracts increasingly abstract features
#     2. A decoder (upsampling path) that gradually recovers spatial details

#     The key is the use of skip connections between corresponding encoder and decoder layers.
#     These connections allow the decoder to access fine-grained details from earlier encoder layers,
#     which helps produce more precise segmentation masks.

#     The skip connections work by concatenating feature maps from the encoder directly into the decoder
#     at corresponding resolutions. This helps preserve important spatial information that would
#     otherwise be lost during the encoding process.

#     Args:
#         encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
#             to extract features of different spatial resolution
#         encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
#             two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
#             with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
#             Default is 5
#         encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
#             other pretrained weights (see table with available weights for each encoder_name)
#         decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
#             Length of the list should be the same as **encoder_depth**
#         decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
#             is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
#             Available options are **True, False, "inplace"**
#         decoder_attention_type: Attention module used in decoder of the model. Available options are
#             **None** and **scse** (https://arxiv.org/abs/1808.08127).
#         decoder_interpolation_mode: Interpolation mode used in decoder of the model. Available options are
#             **"nearest"**, **"bilinear"**, **"bicubic"**, **"area"**, **"nearest-exact"**. Default is **"nearest"**.
#         in_channels: A number of input channels for the model, default is 3 (RGB images)
#         classes: A number of classes for output mask (or you can think as a number of channels of output mask)
#         activation: An activation function to apply after the final convolution layer.
#             Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
#                 **callable** and **None**.
#             Default is **None**
#         aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
#             on top of encoder if **aux_params** is not **None** (default). Supported params:
#                 - classes (int): A number of classes
#                 - pooling (str): One of "max", "avg". Default is "avg"
#                 - dropout (float): Dropout factor in [0, 1)
#                 - activation (str): An activation function to apply "sigmoid"/"softmax"
#                     (could be **None** to return logits)
#         kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with ``None`` values are pruned before passing.

#     Returns:
#         ``torch.nn.Module``: Unet

#     Example:
#         .. code-block:: python

#             import torch
#             import segmentation_models_pytorch as smp

#             model = smp.Unet("resnet18", encoder_weights="imagenet", classes=5)
#             model.eval()

#             # generate random images
#             images = torch.rand(2, 3, 256, 256)

#             with torch.inference_mode():
#                 mask = model(images)

#             print(mask.shape)
#             # torch.Size([2, 5, 256, 256])

#     .. _Unet:
#         https://arxiv.org/abs/1505.04597

#     """

#     requires_divisible_input_shape = False

#     @supports_config_loading
#     def __init__(
#         self,
#         encoder_name: str = "resnet34",
#         encoder_depth: int = 5,
#         encoder_weights: Optional[str] = "imagenet",
#         decoder_use_batchnorm: bool = True,
#         decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
#         decoder_attention_type: Optional[str] = None,
#         decoder_interpolation_mode: str = "nearest",
#         in_channels: int = 9,
#         classes: int = 1,
#         activation: Optional[Union[str, Callable]] = None,
#         aux_params: Optional[dict] = None,
#         **kwargs: dict[str, Any],
#     ):
#         super().__init__()

#         self.encoder = get_encoder(
#             encoder_name,
#             in_channels=in_channels,
#             depth=encoder_depth,
#             weights=encoder_weights,
#             **kwargs,
#         )

#         add_center_block = encoder_name.startswith("vgg")
#         self.decoder = UnetDecoder(
#             encoder_channels=self.encoder.out_channels,
#             decoder_channels=decoder_channels,
#             n_blocks=encoder_depth,
#             use_batchnorm=decoder_use_batchnorm,
#             add_center_block=add_center_block,
#             attention_type=decoder_attention_type,
#             interpolation_mode=decoder_interpolation_mode,
#         )

#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=classes,
#             activation=activation,
#             kernel_size=3,
#         )

#         if aux_params is not None:
#             self.classification_head = ClassificationHead(
#                 in_channels=self.encoder.out_channels[-1], **aux_params
#             )
#         else:
#             self.classification_head = None

#         self.name = "unetsiamese-{}".format(encoder_name)
#         self.siamese = torch.nn.Conv2d(classes * 2, classes, 1, stride=1, padding=0)
#         self.dmg = torch.nn.Conv2d(classes * 2, classes, 1, stride=1, padding=0)
#         self.loc = torch.nn.Conv2d(classes, 1, 1, stride=1, padding=0)
#         self.initialize()
        
       

############
#chia
#####

class UnetSiamese(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation_mode: str = "nearest",
        in_channels: int = 4,  # 4 channels (Optical + SAR Pre)
        classes: int = 4,  # 3 classes + background
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        self.name = f"unetsiamese-{encoder_name}"


        self.encoder_pre_disaster = get_encoder(
            encoder_name,
            in_channels=in_channels,  # 4 channels (Optical + SAR Pre)
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.encoder_post_disaster = get_encoder(
            encoder_name,
            in_channels=in_channels,  # 4 channels (SAR Post)
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        fused_encoder_channels = [c * 2 for c in self.encoder_pre_disaster.out_channels]

        self.decoder = UnetDecoder(
            encoder_channels=fused_encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            add_center_block=encoder_name.startswith("vgg"),
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation_mode,
        )

        # ✅ Corrected Conv layers
        # self.loc = nn.Conv2d(self.encoder_pre_disaster.out_channels[-1], 1, kernel_size=1)  # Binary mask
        self.dmg = nn.Conv2d(decoder_channels[-1], classes, kernel_size=1)  # Multi-class damage
        self.loc = nn.Conv2d(16, 1, 1, stride=1, padding=0)


        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.initialize()

    def initialize(self):
        """Initialize decoder and output heads."""
        for module in [self.decoder, self.segmentation_head, self.loc, self.dmg]:
            if isinstance(module, nn.Module):
                module.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, optical, sar_pre, sar_post):  #this one is modified by Chia
    # # def forward(self, sar_pre, sar_post, optical=None):
    #     if optical is None:
    #         # If optical is missing, handle it (e.g., use a zero tensor, skip fusion layers, etc.)
    #         print("Warning: Optical input missing during inference!")
    #     return self.segmentation_head(self.encoder(sar_pre, sar_post, optical))
        """Late Fusion Siamese Model: Process each encoder separately, then merge features."""
        # print(f"🔍 Input Shapes | Optical: {optical.shape}, SAR Pre: {sar_pre.shape}, SAR Post: {sar_post.shape}")

        assert optical.shape[1] == 3, f"Expected `optical` to have 3 channels, but got {optical.shape[1]}"
        assert sar_pre.shape[1] == 1, f"Expected `sar_pre` to have 1 channel, but got {sar_pre.shape[1]}"
        assert sar_post.shape[1] == 4, f"Expected `sar_post` to have 4 channels, but got {sar_post.shape[1]}"

        # ✅ Merge Pre-Disaster inputs (Optical + SAR Pre) → (B, 4, H, W)
        pre_disaster = torch.cat([optical, sar_pre], dim=1)  
        post_disaster = sar_post  

        # print(f"✅ Pre-Disaster Shape: {pre_disaster.shape}, Post-Disaster Shape: {post_disaster.shape}")

        # ✅ Extract features from both encoders
        pre_disaster_features = self.encoder_pre_disaster(pre_disaster)  
        post_disaster_features = self.encoder_post_disaster(post_disaster)  

        assert isinstance(pre_disaster_features, list), "Pre-Disaster features must be a list!"
        assert isinstance(post_disaster_features, list), "Post-Disaster features must be a list!"

        # # print("\n🔍 Encoder Feature Shapes:")
        # for i in range(len(pre_disaster_features)):
        #     print(f"  🔹 Pre-Disaster Feature {i}: {pre_disaster_features[i].shape}")
        #     print(f"  🔸 Post-Disaster Feature {i}: {post_disaster_features[i].shape}")

        # ✅ Fuse feature maps from both encoders
        fused_features = [
            torch.cat([pre_disaster_features[i], post_disaster_features[i]], dim=1)  
            for i in range(len(pre_disaster_features))
        ]

        # print("\n🚀 Fused Feature Shapes:")
        # for i, x in enumerate(fused_features):
        #     print(f"  🔥 Feature {i}: {x.shape}")

        # ✅ Pass features to decoder
        decoder_output = self.decoder(fused_features)

        # print(f"🛠 Decoder Output Shape: {decoder_output.shape}")
        # print(f"🛠 Expected Input for loc: {self.loc.in_channels}, dmg: {self.dmg.in_channels}")

        # # ✅ Output Heads
        # loc_output = self.loc(pre_disaster_features[-1])  # Localization uses Pre-Disaster only
        # dmg_output = self.dmg(decoder_output)  # Damage segmentation uses fused features
        # ✅ Use decoder output instead of encoder
        loc_output = self.loc(decoder_output)  # Now it has (B, 1, 512, 512)
        dmg_output = self.dmg(decoder_output)


        return loc_output, dmg_output


# class UnetSiamese(nn.Module):
#     """
#     Siamese U-Net with two encoders (one for pre-disaster, one for post-disaster).
#     """

#     requires_divisible_input_shape = False

#     def __init__(
#         self,
#         encoder_name: str = "resnet34",
#         encoder_depth: int = 5,
#         encoder_weights: str | None = "imagenet",
#         decoder_use_batchnorm: bool = True,
#         decoder_channels: tuple = (256, 128, 64, 32, 16),
#         decoder_attention_type: str | None = None,
#         decoder_interpolation_mode: str = "nearest",
#         in_channels: int = 4,  # Changed to 4 (SAR + Optical for pre/post)
#         classes: int = 4,
#         activation: str | None = None,
#         aux_params: dict | None = None,
#         **kwargs,
#     ):
#         super().__init__()
        
#         self.name = f"unetsiamese-{encoder_name}"
#         self.decoder_channels = decoder_channels


#         self.encoder_pre_disaster = get_encoder(
#             encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights, **kwargs
#         )

#         self.encoder_post_disaster = get_encoder(
#             encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights, **kwargs
#         )

#         fused_encoder_channels = [c * 2 for c in self.encoder_pre_disaster.out_channels]

#         self.decoder = UnetDecoder(
#             encoder_channels=fused_encoder_channels,
#             decoder_channels=decoder_channels,
#             n_blocks=encoder_depth,
#             use_batchnorm=decoder_use_batchnorm,
#             add_center_block=encoder_name.startswith("vgg"),
#             attention_type=decoder_attention_type,
#             interpolation_mode=decoder_interpolation_mode,
#         )

#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1], out_channels=classes, activation=activation, kernel_size=3
            
#         )
#         decoder_output = self.decoder(fused_features)

#         print(f"🛠 Decoder Output Shape: {decoder_output.shape}")
#         print(f"🛠 Expected Input for loc: {self.loc.in_channels}, dmg: {self.dmg.in_channels}")


#         self.loc = nn.Conv2d(classes, 1, 1, stride=1, padding=0)
#         self.dmg = nn.Conv2d(classes * 2, classes, 1, stride=1, padding=0)

#         # self.loc = torch.nn.Conv2d(self.encoder_pre_disaster.out_channels[-1], 1, 1, stride=1, padding=0)
#         # # self.dmg = torch.nn.Conv2d(self.decoder_channels[-1], classes, 1, stride=1, padding=0)
#         # self.loc = torch.nn.Conv2d(self.decoder_channels[-1], 1, 1, stride=1, padding=0)  # Binary output
#         # self.dmg = torch.nn.Conv2d(self.decoder_channels[-1], classes, 1, stride=1, padding=0)  # Multi-class output



#         if aux_params is not None:
#             self.classification_head = ClassificationHead(
#                 in_channels=self.encoder_pre_disaster.out_channels[-1], **aux_params
#             )
#         else:
#             self.classification_head = None

#         self.initialize()

#     def initialize(self):
#         """Initialize decoder and output heads."""
#         for module in [self.decoder, self.segmentation_head, self.loc, self.dmg]:
#             if isinstance(module, nn.Module):
#                 module.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, optical, sar_pre, sar_post):
#         """Late Fusion Siamese Model: Process each encoder separately, then merge features."""

#         print(f"🔍 Input Shapes | Optical: {optical.shape}, SAR Pre: {sar_pre.shape}, SAR Post: {sar_post.shape}")

#         # ✅ Ensure correct input channels
#         assert optical.shape[1] == 3, f"Expected `optical` to have 3 channels, but got {optical.shape[1]}"
#         assert sar_pre.shape[1] == 1, f"Expected `sar_pre` to have 1 channel, but got {sar_pre.shape[1]}"
#         assert sar_post.shape[1] == 4, f"Expected `sar_post` to have 4 channels, but got {sar_post.shape[1]}"

#         # ✅ Merge Pre-Disaster inputs (Optical + SAR Pre) → (B, 4, H, W)
#         pre_disaster = torch.cat([optical, sar_pre], dim=1)  
#         post_disaster = sar_post  # Keep as 4 channels

#         print(f"✅ Pre-Disaster Shape: {pre_disaster.shape}, Post-Disaster Shape: {post_disaster.shape}")

#         # ✅ Extract features from both encoders
#         pre_disaster_features = self.encoder_pre_disaster(pre_disaster)  
#         post_disaster_features = self.encoder_post_disaster(post_disaster)  

#         assert isinstance(pre_disaster_features, list), "Pre-Disaster features must be a list!"
#         assert isinstance(post_disaster_features, list), "Post-Disaster features must be a list!"

#         print("\n🔍 Encoder Feature Shapes:")
#         for i in range(len(pre_disaster_features)):
#             print(f"  🔹 Pre-Disaster Feature {i}: {pre_disaster_features[i].shape}")
#             print(f"  🔸 Post-Disaster Feature {i}: {post_disaster_features[i].shape}")

#         # ✅ Fuse feature maps from both encoders
#         fused_features = [
#             torch.cat([pre_disaster_features[i], post_disaster_features[i]], dim=1)  
#             for i in range(len(pre_disaster_features))
#         ]

#         print("\n🚀 Fused Feature Shapes:")
#         for i, x in enumerate(fused_features):
#             print(f"  🔥 Feature {i}: {x.shape}")

#         # ✅ Pass features to decoder
#         decoder_output = self.decoder(fused_features)

#         # ✅ Output Heads
#         loc_output = self.loc(pre_disaster_features[-1])  # Localization uses Pre-Disaster only
#         dmg_output = self.dmg(decoder_output)  # Damage segmentation uses fused features

#         return loc_output, dmg_output



# *******************************************************************************


class Unet(SegmentationModel):
    """
    U-Net is a fully convolutional neural network architecture designed for semantic image segmentation.

    It consists of two main parts:

    1. An encoder (downsampling path) that extracts increasingly abstract features
    2. A decoder (upsampling path) that gradually recovers spatial details

    The key is the use of skip connections between corresponding encoder and decoder layers.
    These connections allow the decoder to access fine-grained details from earlier encoder layers,
    which helps produce more precise segmentation masks.

    The skip connections work by concatenating feature maps from the encoder directly into the decoder
    at corresponding resolutions. This helps preserve important spatial information that would
    otherwise be lost during the encoding process.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        decoder_interpolation_mode: Interpolation mode used in decoder of the model. Available options are
            **"nearest"**, **"bilinear"**, **"bicubic"**, **"area"**, **"nearest-exact"**. Default is **"nearest"**.
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: Unet

    Example:
        .. code-block:: python

            import torch
            import segmentation_models_pytorch as smp

            model = smp.Unet("resnet18", encoder_weights="imagenet", classes=5)
            model.eval()

            # generate random images
            images = torch.rand(2, 3, 256, 256)

            with torch.inference_mode():
                mask = model(images)

            print(mask.shape)
            # torch.Size([2, 5, 256, 256])

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    requires_divisible_input_shape = False

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation_mode: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        add_center_block = encoder_name.startswith("vgg")
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            add_center_block=add_center_block,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation_mode,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes+1,   ###modified by Chia
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
