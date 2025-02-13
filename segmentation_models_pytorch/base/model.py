import torch
from typing import TypeVar, Type

from . import initialization as init
from .hub_mixin import SMPHubMixin
from .utils import is_torch_compiling

T = TypeVar("T", bound="SegmentationModel")

from segmentation_models_pytorch.encoders import get_encoder ###chia
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder ###chia
from segmentation_models_pytorch.base.heads import SegmentationHead
import torch.nn as nn




# *******************************************************************************
# added by B. Adriano
# *******************************************************************************
class SegmentationModelSiamese(torch.nn.Module, SMPHubMixin):
    """Base class for all segmentation models."""

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    # if model supports shape not divisible by 2 ^ n set to False
    requires_divisible_input_shape = True

    # Fix type-hint for models, to avoid HubMixin signature
    def __new__(cls: Type[T], *args, **kwargs) -> T:
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        # if self.classification_head is not None:
        #     init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        """Check if the input shape is divisible by the output stride.
        If not, raise a RuntimeError.
        """
        if not self.requires_divisible_input_shape:
            return

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward_once(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if not (
            torch.jit.is_scripting() or torch.jit.is_tracing() or is_torch_compiling()
        ):
            self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

##original 
    def forward(self, x):
        """added by B. Adriano"""

        x1 = self.forward_once(x[:, :3, :, :])
        x2 = self.forward_once(x[:, 3:, :, :])
    

        return self.loc(x2), self.dmg(torch.cat([x1, x2], dim=1))
    
    def forward(self, x):

        x1 = self.forward_once(x)  # 

        return self.loc(x1), self.dmg(x1)  # Use single forward pass



#######
### added by chia
#######

    # def forward(self, sar_post, optical, sar_pre):
    #     """Late Fusion: Process SAR Post, Optical, and SAR Pre separately, then fuse features."""

    #     optical_features = self.forward_once(optical)

    #     sar_pre_features = self.forward_once(sar_pre)

    #     sar_post_features = self.forward_once(sar_post)

    #     fused_features = torch.cat([optical_features, sar_pre_features, sar_post_features], dim=1)

    #     loc_output = self.loc(sar_post_features)  # Localization uses SAR Post only
    #     dmg_output = self.dmg(fused_features)  # Damage segmentation uses fused features

    #     return loc_output, dmg_output




    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x

    def load_state_dict(self, state_dict, **kwargs):
        # for compatibility of weights for
        # timm- ported encoders with TimmUniversalEncoder
        from segmentation_models_pytorch.encoders import TimmUniversalEncoder

        if not isinstance(self.encoder, TimmUniversalEncoder):
            return super().load_state_dict(state_dict, **kwargs)

        patterns = ["regnet", "res2", "resnest", "mobilenetv3", "gernet"]

        is_deprecated_encoder = any(
            self.encoder.name.startswith(pattern) for pattern in patterns
        )

        if is_deprecated_encoder:
            keys = list(state_dict.keys())
            for key in keys:
                new_key = key
                if key.startswith("encoder.") and not key.startswith("encoder.model."):
                    new_key = "encoder.model." + key.removeprefix("encoder.")
                if "gernet" in self.encoder.name:
                    new_key = new_key.replace(".stages.", ".stages_")
                state_dict[new_key] = state_dict.pop(key)

        return super().load_state_dict(state_dict, **kwargs)


# # *******************************************************************************
# # added by Chia
# # *******************************************************************************
# class SegmentationModelSiamese(torch.nn.Module, SMPHubMixin):
#     """Base class for all segmentation models."""

#     _is_torch_scriptable = True
#     _is_torch_exportable = True
#     _is_torch_compilable = True

#     # if model supports shape not divisible by 2 ^ n set to False
#     requires_divisible_input_shape = True

#     # Fix type-hint for models, to avoid HubMixin signature
#     def __new__(cls: Type[T], *args, **kwargs) -> T:
#         instance = super().__new__(cls, *args, **kwargs)
#         return instance

#     def initialize(self):
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
    

#     def check_input_shape(self, x):
#         """Check if the input shape is divisible by the output stride.
#         If not, raise a RuntimeError.
#         """
#         if not self.requires_divisible_input_shape:
#             return

#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (
#                 (h // output_stride + 1) * output_stride
#                 if h % output_stride != 0
#                 else h
#             )
#             new_w = (
#                 (w // output_stride + 1) * output_stride
#                 if w % output_stride != 0
#                 else w
#             )
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#                 )
#     def forward_once(self, x, encoder_type="rgb"):
#         """Pass input `x` through the appropriate encoder and return features."""

#         if encoder_type == "rgb":
#             features = self.encoder_rgb(x)
#         elif encoder_type == "sar_pre":
#             features = self.encoder_sar_pre(x)
#         elif encoder_type == "sar_post":
#             features = self.encoder_sar_post(x)
#         else:
#             raise ValueError("Invalid encoder type. Choose from 'rgb', 'sar_pre', 'sar_post'.")

#         # 🚀 Fix: Print the shape of each feature map in the list
#         if isinstance(features, list):
#             print(f"Encoder: {encoder_type}")
#             for i, feat in enumerate(features):
#                 print(f"  - Feature {i} shape: {feat.shape}")

#         return features  # Return feature maps as a list

#     def forward(self, sar_post, optical, sar_pre):
#         """Late Fusion Model: Process each input separately, then merge before segmentation."""

#         print("🚀 Forward function is being called!")

#         rgb_features = self.forward_once(optical, encoder_type="rgb")  # List of feature maps
#         sar_pre_features = self.forward_once(sar_pre, encoder_type="sar_pre")  # List of feature maps
#         sar_post_features = self.forward_once(sar_post, encoder_type="sar_post")  # List of feature maps

#         # 🚨 Ensure encoder outputs are lists
#         if not isinstance(rgb_features, list) or not isinstance(sar_pre_features, list) or not isinstance(sar_post_features, list):
#             raise TypeError("🚨 Encoder outputs must be lists of feature maps.")

#         # ✅ Ensure all lists have the same length
#         min_features = min(len(rgb_features), len(sar_pre_features), len(sar_post_features))
#         print(f"✅ Using {min_features} feature maps from each encoder")

#         # ✅ Create a list of concatenated features for each scale
#         decoder_input = [
#             torch.cat([rgb_features[i], sar_pre_features[i], sar_post_features[i]], dim=1)
#             for i in range(min_features)
#         ]
        
#         print("🚀 Feature shapes before channel reduction:", [x.shape for x in decoder_input])

#         # ✅ Reduce channels adaptively: Apply reduction only for necessary feature maps
#         reduced_decoder_input = []
#         for i, x in enumerate(decoder_input):
#             in_channels = x.shape[1]  # Get the number of channels at this scale
            
#             if in_channels > 608:  # Only apply reduction if needed
#                 reduction_layer = nn.Conv2d(in_channels, 608, kernel_size=1, stride=1, padding=0).to(x.device)
#                 reduced_decoder_input.append(reduction_layer(x))
#             else:
#                 reduced_decoder_input.append(x)

#         print(f"✅ Feature shapes after channel reduction: {[x.shape for x in reduced_decoder_input]}")

#         # 🚨 Check decoder input
#         assert all(x.shape[1] == 608 for x in reduced_decoder_input), "🚨 Some feature maps still have incorrect channels!"

#         # 🚨 If `decoder_input` is a tensor, raise an error
#         if isinstance(reduced_decoder_input, torch.Tensor):
#             raise TypeError(f"🚨 Expected `decoder_input` to be a list, but got a tensor with shape: {reduced_decoder_input.shape}")

#         # ✅ Pass as a list
#         decoder_output = self.decoder(reduced_decoder_input)
#         print("✅ Decoder output computed")

#         # ✅ Output Heads
#         loc_output = self.loc(sar_post_features[-1])  # Localization uses SAR Post only
#         dmg_output = self.dmg(decoder_output)  # Damage segmentation uses fused features

#         return loc_output, dmg_output



#     def __init__(self, encoder_name="resnet34", encoder_depth=5, encoder_weights="imagenet",
#                 decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16),
#                 decoder_attention_type=None, decoder_interpolation_mode="nearest",
#                 classes=1, activation=None, aux_params=None, **kwargs):
#         super().__init__()

#         #  Three separate encoders for Optical & SAR Pre/Post
#         self.encoder_rgb = get_encoder(encoder_name, in_channels=3, depth=encoder_depth, weights=encoder_weights, **kwargs)
#         self.encoder_sar_pre = get_encoder(encoder_name, in_channels=3, depth=encoder_depth, weights=encoder_weights, **kwargs)
#         self.encoder_sar_post = get_encoder(encoder_name, in_channels=3, depth=encoder_depth, weights=encoder_weights, **kwargs)

#         add_center_block = encoder_name.startswith("vgg")

#         # ✅ Fix Decoder Initialization
#         encoder_channels = [c * 3 for c in self.encoder_rgb.out_channels]  # Multiply each encoder output by 3 (concatenation)
        
#         self.decoder = UnetDecoder(
#             encoder_channels=encoder_channels,  # ✅ Corrected 
#             decoder_channels=decoder_channels,
#             n_blocks=encoder_depth,
#             use_batchnorm=decoder_use_batchnorm,
#             add_center_block=add_center_block,
#             attention_type=decoder_attention_type,
#             interpolation_mode=decoder_interpolation_mode,
#         )
#         self.channel_reduction = nn.Conv2d(1536, 608, kernel_size=1, stride=1, padding=0)


#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=classes,
#             activation=activation,
#             kernel_size=3,
#         )

#         self.loc = torch.nn.Conv2d(classes, 1, 1, stride=1, padding=0)  # Localization head
#         self.dmg = torch.nn.Conv2d(classes * 2, classes, 1, stride=1, padding=0)  # Damage head

#         self.initialize()


class SegmentationModel(torch.nn.Module, SMPHubMixin):
    """Base class for all segmentation models."""

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    # if model supports shape not divisible by 2 ^ n set to False
    requires_divisible_input_shape = True

    # Fix type-hint for models, to avoid HubMixin signature
    def __new__(cls: Type[T], *args, **kwargs) -> T:
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        """Check if the input shape is divisible by the output stride.
        If not, raise a RuntimeError.
        """
        if not self.requires_divisible_input_shape:
            return

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )
            
    def forward_once(self, x, encoder_type="rgb"):
        """Pass input `x` through the appropriate encoder and return feature maps."""
        
        if encoder_type == "rgb":
            features = self.encoder_rgb(x)
        elif encoder_type == "sar_pre":
            features = self.encoder_sar_pre(x)
        elif encoder_type == "sar_post":
            features = self.encoder_sar_post(x)
        else:
            raise ValueError("Invalid encoder type. Choose from 'rgb', 'sar_pre', 'sar_post'.")

        # 🚨 Debug print statement
        if not isinstance(features, list) or len(features) == 0:
            print(f"🚨 WARNING: Encoder `{encoder_type}` returned an empty feature list!")
        
    #     return features 

    # def forward(self, sar_post, optical, sar_pre):
    #     """Late Fusion Model: Process each input separately, then merge before segmentation."""
        
    #     # Extract feature maps
    #     rgb_features = self.forward_once(optical, encoder_type="rgb")  # List of feature maps
    #     sar_pre_features = self.forward_once(sar_pre, encoder_type="sar_pre")  # List of feature maps
    #     sar_post_features = self.forward_once(sar_post, encoder_type="sar_post")  # List of feature maps

    #     # Ensure the extracted features are valid lists
    #     assert isinstance(rgb_features, list), "RGB features must be a list!"
    #     assert isinstance(sar_pre_features, list), "SAR Pre features must be a list!"
    #     assert isinstance(sar_post_features, list), "SAR Post features must be a list!"
        
  


    #     # Concatenate feature maps across all levels
    #     decoder_input = [
    #         torch.cat([rgb_features[i], sar_pre_features[i], sar_post_features[i]], dim=1)
    #         for i in range(len(rgb_features))
    #     ]

    #     # ✅ Reduce channels before passing to decoder
    #     decoder_input = [self.channel_reduction(conv) for conv in decoder_input]
    #     print(f"✅ Before Channel Reduction: {[x.shape for x in decoder_input]}")

    #     decoder_input = [self.channel_reduction(conv) for conv in decoder_input]

    #     print(f"✅ After Channel Reduction: {[x.shape for x in decoder_input]}")



    #     # Pass to decoder
    #     decoder_output = self.decoder(decoder_input)  

    #     # Output layers
    #     loc_output = self.loc(sar_post_features[-1])  # Localization head
    #     dmg_output = self.dmg(decoder_output)  # Damage segmentation head

    #     return loc_output, dmg_output
    
    def forward(self, sar_post, optical, sar_pre):
        """Late Fusion Siamese Model: Process each encoder separately, then merge features."""

        if optical.shape[1] != 3:  # Ensure it's not already (C, H, W)
            optical = optical.permute(2, 0, 1)  # (3, 512, 512)

        assert sar_pre.shape[1] == 1, f"Expected SAR Pre to have 1 channel, but got {sar_pre.shape[1]}"

        # pre_disaster = torch.cat([optical, sar_pre], dim=0)  # (4, 512, 512)
        
                # Ensure `optical` has shape (3, H, W)
        if optical.shape[0] != 3:  
            raise ValueError(f"Expected Optical shape (3, H, W), but got {optical.shape}")

        # Ensure `sar_pre` has shape (1, H, W)
        if sar_pre.shape[0] != 1:  
            raise ValueError(f"Expected SAR Pre shape (1, H, W), but got {sar_pre.shape}")
        print(f"🔍 Optical shape: {optical.shape}, SAR Pre shape: {sar_pre.shape}")


        # ✅ Now stack properly
        pre_disaster = torch.cat([optical, sar_pre], dim=0)  # Should be (4, 512, 512)


        post_disaster = sar_post  # Already (4, 512, 512)

        # ✅ Extract features from both encoders
        pre_disaster_features = self.forward_once(pre_disaster, encoder_type="pre_disaster")  
        post_disaster_features = self.forward_once(post_disaster, encoder_type="post_disaster")  

        # ✅ Ensure encoder outputs are lists
        assert isinstance(pre_disaster_features, list), "Pre-Disaster features must be a list!"
        assert isinstance(post_disaster_features, list), "Post-Disaster features must be a list!"

        print("\n🔍 Encoder Feature Shapes Before Fusion:")
        for i in range(len(pre_disaster_features)):
            print(f"  🔹 Pre-Disaster Feature {i}: {pre_disaster_features[i].shape}")
            print(f"  🔸 Post-Disaster Feature {i}: {post_disaster_features[i].shape}")

        # ✅ Fuse feature maps from both encoders at each level
        fused_features = [
            torch.cat([pre_disaster_features[i], post_disaster_features[i]], dim=1)  
            for i in range(len(pre_disaster_features))
        ]

        print("\n🚀 Fused Feature Shapes:")
        for i, x in enumerate(fused_features):
            print(f"  🔥 Feature {i}: {x.shape}")


        # ✅ Apply channel reduction (1x1 Conv) for decoder compatibility
        reduced_decoder_input = [self.channel_reduction(x) if x.shape[1] > self.decoder.in_channels else x for x in fused_features]

        # ✅ Pass fused features to decoder
        decoder_output = self.decoder(reduced_decoder_input)

        # ✅ Output Heads
        loc_output = self.loc(pre_disaster_features[-1])  # Localization uses Pre-Disaster only
        dmg_output = self.dmg(decoder_output)  # Damage segmentation uses fused features

        return loc_output, dmg_output



    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x

    def load_state_dict(self, state_dict, **kwargs):
        # for compatibility of weights for
        # timm- ported encoders with TimmUniversalEncoder
        from segmentation_models_pytorch.encoders import TimmUniversalEncoder

        if not isinstance(self.encoder, TimmUniversalEncoder):
            return super().load_state_dict(state_dict, **kwargs)

        patterns = ["regnet", "res2", "resnest", "mobilenetv3", "gernet"]

        is_deprecated_encoder = any(
            self.encoder.name.startswith(pattern) for pattern in patterns
        )

        if is_deprecated_encoder:
            keys = list(state_dict.keys())
            for key in keys:
                new_key = key
                if key.startswith("encoder.") and not key.startswith("encoder.model."):
                    new_key = "encoder.model." + key.removeprefix("encoder.")
                if "gernet" in self.encoder.name:
                    new_key = new_key.replace(".stages.", ".stages_")
                state_dict[new_key] = state_dict.pop(key)

        return super().load_state_dict(state_dict, **kwargs)





# *******************************************************************************
# added by Chia. (siamese architecture one encoder for SAR one endcoder for RGB)
# *********************************************************
# class SegmentationModelSiamese(nn.Module):
#     """Siamese-style segmentation model for disaster analysis."""

#     def __init__(self, encoder_name="resnet34", encoder_depth=5, encoder_weights="imagenet",
#                  decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16),
#                  decoder_attention_type=None, decoder_interpolation_mode="nearest",
#                  classes=1, activation=None, **kwargs):
#         super().__init__()

#         self.decoder_channels = decoder_channels  

#         self.encoder_pre_disaster = get_encoder(
#             encoder_name, in_channels=4, depth=encoder_depth, weights=encoder_weights, **kwargs
#         )

#         self.encoder_post_disaster = get_encoder(
#             encoder_name, in_channels=4, depth=encoder_depth, weights=encoder_weights, **kwargs
#         )

#         add_center_block = encoder_name.startswith("vgg")

#         encoder_channels = [c * 2 for c in self.encoder_pre_disaster.out_channels]  

#         # self.decoder = UnetDecoder(
#         #     encoder_channels=encoder_channels,  
#         #     decoder_channels=decoder_channels,
#         #     n_blocks=encoder_depth,
#         #     use_batchnorm=decoder_use_batchnorm,
#         #     add_center_block=add_center_block,
#         #     attention_type=decoder_attention_type,
#         #     interpolation_mode=decoder_interpolation_mode,
#         # )
#         # ✅ Fix: Explicitly set correct fused feature channels for decoder
#         self.decoder = UnetDecoder(
#             encoder_channels=[8, 128, 128, 256, 512, 1024],  # Match fused feature channels
#             decoder_channels=decoder_channels,  # Keep standard
#             n_blocks=encoder_depth,
#             use_batchnorm=decoder_use_batchnorm,
#             add_center_block=add_center_block,
#             attention_type=decoder_attention_type,
#             interpolation_mode=decoder_interpolation_mode,
#         )


#         # self.channel_reduction = nn.Conv2d(
#         #     in_channels=encoder_channels[-1], 
#         #     out_channels=decoder_channels[0], 
#         #     kernel_size=1
#         # )
#         self.channel_reduction = nn.Conv2d(
#             in_channels=8,  # Fix input channels
#             out_channels=1024,  # Match expected decoder input
#             kernel_size=1
#         )


#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=classes,
#             activation=activation,
#             kernel_size=3,
#         )

#         self.loc = nn.Conv2d(classes, 1, 1, stride=1, padding=0)  
#         self.dmg = nn.Conv2d(classes * 2, classes, 1, stride=1, padding=0)  

#         self.initialize()

#     def initialize(self):
#         """Initialize decoder and output heads."""
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         init.initialize_head(self.loc)
#         init.initialize_head(self.dmg)

#     def forward_once(self, x, encoder_type="pre_disaster"):
#         """Pass input through the appropriate encoder."""
#         features = self.encoder_pre_disaster(x) if encoder_type == "pre_disaster" else self.encoder_post_disaster(x)
        
#         print(f"\n🚀 {encoder_type} Encoder Outputs:")
#         for i, f in enumerate(features):
#             print(f"  🔹 Layer {i}: {f.shape}")
        
#         return features

#     def forward(self, rgb_pre, sar_pre, sar_post):
#         """Late Fusion Siamese Model: Process each encoder separately, then merge features."""
        
#         # Debugging input shapes
#         print(f"🔍 Input Shapes | RGB Pre: {rgb_pre.shape}, SAR Pre: {sar_pre.shape}, SAR Post: {sar_post.shape}")

#         # ✅ Ensure correct input channels
#         assert rgb_pre.shape[1] == 3, f"Expected `rgb_pre` to have 3 channels, but got {rgb_pre.shape[1]}"
#         assert sar_pre.shape[1] == 1, f"Expected `sar_pre` to have 1 channel, but got {sar_pre.shape[1]}"
#         assert sar_post.shape[1] == 4, f"Expected `sar_post` to have 4 channels, but got {sar_post.shape[1]}"

#         # ✅ Stack Pre-Disaster inputs (RGB + SAR) → (B, 4, H, W)
#         pre_disaster = torch.cat([rgb_pre, sar_pre], dim=1)  
#         post_disaster = sar_post  # Keep as 4 channels
        
#         print(f"✅ Pre-Disaster Shape: {pre_disaster.shape}, Post-Disaster Shape: {post_disaster.shape}")

#         # ✅ Extract features from both encoders
#         pre_disaster_features = self.forward_once(pre_disaster, encoder_type="pre_disaster")  
#         post_disaster_features = self.forward_once(post_disaster, encoder_type="post_disaster")  

#         # ✅ Ensure encoder outputs are lists
#         assert isinstance(pre_disaster_features, list), "Pre-Disaster features must be a list!"
#         assert isinstance(post_disaster_features, list), "Post-Disaster features must be a list!"

#         # ✅ Debugging feature shapes from encoders
#         print("\n🔍 Encoder Feature Shapes:")
#         for i in range(len(pre_disaster_features)):
#             print(f"  🔹 Pre-Disaster Feature {i}: {pre_disaster_features[i].shape}")
#             print(f"  🔸 Post-Disaster Feature {i}: {post_disaster_features[i].shape}")

#         # ✅ Fuse feature maps from both encoders
#         # fused_features = [
#         #     torch.cat([pre_disaster_features[i], post_disaster_features[i]], dim=1)  
#         #     for i in range(len(pre_disaster_features))
#         # ]
#         fused_features = []
#         for i in range(len(pre_disaster_features)):
#             fused = torch.cat([pre_disaster_features[i], post_disaster_features[i]], dim=1)

#             expected_channels = self.decoder_channels[i]  # Decoder expects this many channels

#             # ✅ Reduce channels if they don't match the expected input
#         if fused_features[0].shape[1] != 1024:
#             print(f"⚠️ Expanding Feature 0 from {fused_features[0].shape[1]} to 1024 channels...")
#             fused_features[0] = self.channel_reduction(fused_features[0])  




                
#         print("\n🚀 Fused Feature Shapes:")
#         for i, x in enumerate(fused_features):
#             print(f"  🔥 Feature {i}: {x.shape}")

#         # ✅ Apply channel reduction (Ensure correct decoder compatibility)
#         reduced_decoder_input = []
#         for i, x in enumerate(fused_features):
#             if x.shape[1] != self.decoder_channels[i]:
#                 reduced = self.channel_reduction(x)  # Apply 1x1 Conv to match channels
#                 print(f"  🔍 Before Reduction | Feature {i}: {x.shape} ➡️ ✅ After Reduction: {reduced.shape}")
#             else:
#                 reduced = x  # Keep as is if already correct
#             reduced_decoder_input.append(reduced)

#         # ✅ Final check before decoder
#         print("\n✅ Final Decoder Input Shapes:")
#         for i, x in enumerate(reduced_decoder_input):
#             print(f"  🛠 Decoder Input {i}: {x.shape} (Expected: {self.decoder_channels[i]})")

#         # ✅ Pass features to decoder
#         decoder_output = self.decoder(reduced_decoder_input)

#         # ✅ Output Heads
#         loc_output = self.loc(pre_disaster_features[-1])  # Localization uses Pre-Disaster only
#         dmg_output = self.dmg(decoder_output)  # Damage segmentation uses fused features

#         return loc_output, dmg_output

# *******************************************************************************

# class SegmentationModelSiamese_back(nn.Module):
#     """Siamese-style segmentation model for disaster analysis."""

#     def __init__(self, encoder_name="resnet34", encoder_depth=5, encoder_weights="imagenet",
#                  decoder_use_batchnorm=True, decoder_channels = (512, 256, 128, 64, 32),
#                  decoder_attention_type=None, decoder_interpolation_mode="nearest",
#                  classes=1, activation=None, **kwargs):
#         super().__init__()

#         self.decoder_channels = decoder_channels  

#         self.encoder_pre_disaster = get_encoder(
#             encoder_name, in_channels=4, depth=encoder_depth, weights=encoder_weights, **kwargs
#         )

#         self.encoder_post_disaster = get_encoder(
#             encoder_name, in_channels=4, depth=encoder_depth, weights=encoder_weights, **kwargs
#         )

#         add_center_block = encoder_name.startswith("vgg")

        
#         self.decoder_pre = UnetDecoder(
#             encoder_channels=self.encoder_pre_disaster.out_channels,  # Use pre-disaster encoder
#             decoder_channels=decoder_channels,  
#             n_blocks=encoder_depth,
#             use_batchnorm=decoder_use_batchnorm,
#             add_center_block=add_center_block,
#             attention_type=decoder_attention_type,
#             interpolation_mode=decoder_interpolation_mode,
#         )

#         self.decoder_post = UnetDecoder(
#             encoder_channels=self.encoder_post_disaster.out_channels,  # Use post-disaster encoder
#             decoder_channels=decoder_channels,  
#             n_blocks=encoder_depth,
#             use_batchnorm=decoder_use_batchnorm,
#             add_center_block=add_center_block,
#             attention_type=decoder_attention_type,
#             interpolation_mode=decoder_interpolation_mode,
#         )



#         self.channel_reductions = nn.ModuleList([
#             nn.Conv2d(in_channels=fused_encoder_channels[i], out_channels=decoder_channels[0], kernel_size=1)
#             if fused_encoder_channels[i] != decoder_channels[0] else nn.Identity()
#             for i in range(len(fused_encoder_channels))
#         ])

#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=classes,
#             activation=activation,
#             kernel_size=3,
#         )

#         self.loc = nn.Conv2d(classes, 1, 1, stride=1, padding=0)  
#         self.dmg = nn.Conv2d(classes * 2, classes, 1, stride=1, padding=0)  

#         self.initialize()

#     def initialize(self):
#         """Initialize decoder and output heads."""
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         init.initialize_head(self.loc)
#         init.initialize_head(self.dmg)
        
#     def forward_once(self, x, encoder_type="pre_disaster"):
#         """Pass input through the appropriate encoder."""
#         if encoder_type == "pre_disaster":
#             return self.encoder_pre_disaster(x)
#         elif encoder_type == "post_disaster":
#             return self.encoder_post_disaster(x)
#         else:
#             raise ValueError("Invalid encoder type. Choose 'pre_disaster' or 'post_disaster'.")


#     def forward(self, rgb_pre, sar_pre, sar_post):
#         """Late Fusion Siamese Model: Process each encoder separately, then merge features."""

#         print(f"🔍 Input Shapes | RGB Pre: {rgb_pre.shape}, SAR Pre: {sar_pre.shape}, SAR Post: {sar_post.shape}")

#         assert rgb_pre.shape[1] == 3, f"Expected `rgb_pre` to have 3 channels, but got {rgb_pre.shape[1]}"
#         assert sar_pre.shape[1] == 1, f"Expected `sar_pre` to have 1 channel, but got {sar_pre.shape[1]}"
#         assert sar_post.shape[1] == 4, f"Expected `sar_post` to have 4 channels, but got {sar_post.shape[1]}"

#         pre_disaster = torch.cat([rgb_pre, sar_pre], dim=1)  # (B, 4, H, W)
#         post_disaster = sar_post  # (B, 4, H, W)

#         print(f"✅ Pre-Disaster Shape: {pre_disaster.shape}, Post-Disaster Shape: {post_disaster.shape}")

#         pre_disaster_features = self.forward_once(pre_disaster, encoder_type="pre_disaster")  
#         post_disaster_features = self.forward_once(post_disaster, encoder_type="post_disaster")  

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

    
#         reduced_decoder_input = []
#         for i, x in enumerate(fused_features):
#             expected_channels = self.decoder_channels[i]  # Expected by decoder

#             if x.shape[1] != expected_channels:
#                 reduced = self.channel_reductions[i](x)  # ✅ Fix: Use `i` correctly
#                 print(f"  🔍 Before Reduction | Feature {i}: {x.shape} ➡️ ✅ After Reduction: {reduced.shape}")
#             else:
#                 reduced = x  # Keep as is if already correct
            
#             reduced_decoder_input.append(reduced)


#         print("\n✅ Final Decoder Input Shapes:")
#         for i, x in enumerate(reduced_decoder_input):
#             print(f"  🛠 Decoder Input {i}: {x.shape} (Expected: {self.decoder_channels[i]})")

#         # ✅ Pass features to decoder
#         decoder_output = self.decoder(reduced_decoder_input)

#         # ✅ Output Heads
#         loc_output = self.loc(pre_disaster_features[-1])  
#         dmg_output = self.dmg(decoder_output)  

#         return loc_output, dmg_output

####################################################################################

# class SegmentationModelSiamese(torch.nn.Module, SMPHubMixin):
    
#     _is_torch_scriptable = True
#     _is_torch_exportable = True
#     _is_torch_compilable = True

#     # if model supports shape not divisible by 2 ^ n set to False
#     requires_divisible_input_shape = True

#     # Fix type-hint for models, to avoid HubMixin signature
#     def __new__(cls: Type[T], *args, **kwargs) -> T:
#         instance = super().__new__(cls, *args, **kwargs)
#         return instance
#     """Siamese-style segmentation model for disaster analysis."""

#     def __init__(self, encoder_name="resnet34", encoder_depth=5, encoder_weights="imagenet",
#                  decoder_use_batchnorm=True, decoder_channels = (512, 256, 128, 64, 32),  # ✅ 5 elements
#                  decoder_attention_type=None, decoder_interpolation_mode="nearest",
#                  classes=1, activation=None, **kwargs):
#         super().__init__()

#         self.decoder_channels = decoder_channels  

#         self.encoder_pre_disaster = get_encoder(
#             encoder_name, in_channels=4, depth=encoder_depth, weights=encoder_weights, **kwargs
#         )

#         self.encoder_post_disaster = get_encoder(
#             encoder_name, in_channels=4, depth=encoder_depth, weights=encoder_weights, **kwargs
#         )

#         add_center_block = encoder_name.startswith("vgg")
        
#         fused_encoder_channels = [8, 128, 128, 256, 512, 1024]  # ✅ Corrected

#         # ✅ Two separate decoders (Fixed decoder_channels to match depth)
#         self.decoder_pre = UnetDecoder(
#             encoder_channels=self.encoder_pre_disaster.out_channels,  
#             decoder_channels=decoder_channels,  # ✅ Now has only 5 elements
#             n_blocks=encoder_depth,  
#             use_batchnorm=decoder_use_batchnorm,
#             add_center_block=add_center_block,
#             attention_type=decoder_attention_type,
#             interpolation_mode=decoder_interpolation_mode,
#         )

#         self.decoder_post = UnetDecoder(
#             encoder_channels=self.encoder_post_disaster.out_channels,  
#             decoder_channels=decoder_channels,  # ✅ Now has only 5 elements
#             n_blocks=encoder_depth,  
#             use_batchnorm=decoder_use_batchnorm,
#             add_center_block=add_center_block,
#             attention_type=decoder_attention_type,
#             interpolation_mode=decoder_interpolation_mode,
#         )

#         # ✅ Output heads
#         self.loc = nn.Conv2d(decoder_channels[-1], 1, 1, stride=1, padding=0)  
#         self.dmg = nn.Conv2d(decoder_channels[-1] * 2, classes, 1, stride=1, padding=0)  

#         self.initialize()

#     def initialize(self):
#         """Initialize decoder and output heads."""
#         init.initialize_decoder(self.decoder_pre)
#         init.initialize_decoder(self.decoder_post)
#         init.initialize_head(self.loc)
#         init.initialize_head(self.dmg)

#     def forward_once(self, x, encoder_type="pre_disaster"):
#         """Pass input through the appropriate encoder."""
#         if encoder_type == "pre_disaster":
#             return self.encoder_pre_disaster(x)
#         elif encoder_type == "post_disaster":
#             return self.encoder_post_disaster(x)
#         else:
#             raise ValueError("Invalid encoder type. Choose 'pre_disaster' or 'post_disaster'.")

#     def forward(self, rgb_pre, sar_pre, sar_post):
#         """Late Fusion Siamese Model: Process each encoder separately, then merge features."""

#         print(f"🔍 Input Shapes | RGB Pre: {rgb_pre.shape}, SAR Pre: {sar_pre.shape}, SAR Post: {sar_post.shape}")

#         assert rgb_pre.shape[1] == 3, f"Expected `rgb_pre` to have 3 channels, but got {rgb_pre.shape[1]}"
#         assert sar_pre.shape[1] == 1, f"Expected `sar_pre` to have 1 channel, but got {sar_pre.shape[1]}"
#         assert sar_post.shape[1] == 4, f"Expected `sar_post` to have 4 channels, but got {sar_post.shape[1]}"

#         # ✅ Stack Pre-Disaster inputs (RGB + SAR) → (B, 4, H, W)
#         pre_disaster = torch.cat([rgb_pre, sar_pre], dim=1)  
#         post_disaster = sar_post  # Keep as 4 channels

#         print(f"✅ Pre-Disaster Shape: {pre_disaster.shape}, Post-Disaster Shape: {post_disaster.shape}")

#         # ✅ Extract features from both encoders
#         pre_disaster_features = self.forward_once(pre_disaster, encoder_type="pre_disaster")  
#         post_disaster_features = self.forward_once(post_disaster, encoder_type="post_disaster")  

#         assert isinstance(pre_disaster_features, list), "Pre-Disaster features must be a list!"
#         assert isinstance(post_disaster_features, list), "Post-Disaster features must be a list!"

#         print("\n🔍 Encoder Feature Shapes:")
#         for i in range(len(pre_disaster_features)):
#             print(f"  🔹 Pre-Disaster Feature {i}: {pre_disaster_features[i].shape}")
#             print(f"  🔸 Post-Disaster Feature {i}: {post_disaster_features[i].shape}")

#         # ✅ Decode separately
#         pre_decoded = self.decoder_pre(pre_disaster_features)
#         post_decoded = self.decoder_post(post_disaster_features)
        

#         print(f"✅ Pre-Decoded Shape: {pre_decoded.shape}, Post-Decoded Shape: {post_decoded.shape}")

#         # ✅ Fuse decoder outputs
        
        
#         fused_output = torch.cat([pre_decoded, post_decoded], dim=1)  

#         print(f"✅ Fused Output Shape: {fused_output.shape}")
        

#         # ✅ Output Heads
#         loc_output = self.loc(pre_decoded)  
#         dmg_output = self.dmg(fused_output)  

#         return loc_output, dmg_output

