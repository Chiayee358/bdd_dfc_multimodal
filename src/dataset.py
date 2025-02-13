import numpy as np
import torch
import rasterio
from . import transforms
import torch.nn.functional as F


def load_multiband(path):
    src = rasterio.open(path, "r")
    return np.moveaxis(src.read(), 0, -1)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return src.read(1)




class BDD (torch.utils.data.Dataset):

    def __init__(self, fn_list, img_size=512, classes=[0, 1], augm=None):
        self.post_event = [str(f) for f in fn_list]
        self.pre_event = [
            f.replace("post_disaster", "pre_disaster")
            for f in self.post_event
        ]
        # self.target = [
        #     f.replace("post_disaster", "building_damage"
        #     )
        #     for f in self.post_event
        # ]
        # self.target = [
        #     # f.replace("post-event", "target")
        #     f.replace("post_disaster", "target")
        #     .replace("target", "building_damage")
        #     .replace("_sar", "")
        #     for f in self.post_event
        # ]
        
        self.target = [
            f.replace("post_disaster", "target")
            .replace("target", "building_damage")
            .replace("_sar", "")
            for f in self.post_event
]

        print("Example target path:", self.target[0])




        self.augm = augm
        self.size = img_size
        self.to_tensor = transforms.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):

        sar = self.load_grayscale(self.post_event[idx]).astype("uint8")
        rgb = self.load_multiband(self.pre_event[idx]).astype("uint8")
        gtd = self.load_grayscale(self.target[idx]).astype("uint8")

        # make 3-channel sar image beacuse of albumentations
        sar = np.stack([sar, sar, sar], axis=-1)

        if self.augm:
            data = self.augm({"sar": sar, "rgb": rgb, "gtd": gtd}, size=self.size)
            sar, rgb, gtd = data["image"], data["image1"], data["mask"]

        # it includes a simple normalization because of 8-bit data
        sar, rgb, gtd = self.to_tensor({"sar": sar, "rgb": rgb, "gtd": gtd})

        # it makes a builfing footprint mask
        msk = ((torch.argmax(gtd, dim=0) != 0) * 1.0).unsqueeze(0)

        return sar, rgb, gtd, msk, self.post_event[idx]

    def __len__(self):
        return len(self.post_event)


##modiying for 2 encoder #########
class BDD3(torch.utils.data.Dataset):

    def __init__(self, fn_list, img_size=512, classes=[0, 1], augm=None):
        # Post-event (SAR) file paths from the post_disaster folder.
        self.post_event = [str(f) for f in fn_list]

        # Pre-disaster Optical:
        # Change the directory from "post_disaster" to "pre_disaster"
        # and the filename suffix from "_post_disaster_sar" to "_pre_disaster"
        self.pre_event_optical = [
        f.replace("_post_disaster_sar.tif", "_pre_disaster.tif")
        .replace("post_disaster/", "pre_disaster/")
        for f in self.post_event
        
    ]
        # Pre-disaster SAR:
        # Change the directory from "post_disaster" to "pre_disaster"
        # and the filename suffix from "_post_disaster_sar" to "_pre_disaster_sar"
        self.pre_event_sar = [
            f.replace("post_disaster", "pre_disaster")
             .replace("_post_disaster_sar", "_pre_disaster_sar")
            for f in self.post_event
        ]

        # Generate target paths.
        # (Assuming your target generation logic remains the same.)
        self.target = [
            f.replace("post_disaster", "target")
             .replace("target", "building_damage")
             .replace("_sar", "")
            for f in self.post_event
        ]
        
        self.augm = augm
        self.size = img_size
        # self.to_tensor = transforms.ToTensor(classes=classes)
        self.load_multiband = load_multiband  # For optical images (assumed multiband)
        self.load_grayscale = load_grayscale  # For SAR and target (assumed single-channel)
        
    # def __getitem__(self, idx):
    #     # Load post-disaster SAR (1-band → duplicated to 4)
    #     sar_post = self.load_grayscale(self.post_event[idx]).astype("uint8")
    #     sar_post = np.stack([sar_post] * 4, axis=0)  # (4, 512, 512)

    #     # Load pre-disaster optical image
    #     optical = self.load_multiband(self.pre_event_optical[idx]).astype("uint8")
    #     # print(f"🟢 Loaded Optical Shape BEFORE FIX: {optical.shape}")



    #     # ✅ Fix: Ensure optical image has 3 channels (RGB)
    #     if optical.shape[-1] == 3 and optical.shape[0] != 3:
    #         optical = np.transpose(optical, (2, 0, 1))  # Convert (H, W, 3) → (3, H, W)
    #     # print(f"🟢 Loaded Optical Shape: {optical.shape}")


    #     # ✅ Fix: Ensure correct shape (3, H, W)
    #     assert optical.shape == (3, 512, 512), f"Unexpected optical shape: {optical.shape}"

    #     # Load pre-disaster SAR (1-band)
    #     sar_pre = self.load_grayscale(self.pre_event_sar[idx]).astype("uint8")
    #     sar_pre = np.expand_dims(sar_pre, axis=0)  # Ensure it's (1, 512, 512)

    #     # Load ground truth
    #     gtd = self.load_grayscale(self.target[idx]).astype("uint8")
        
    #     # Convert images to float32 and normalize to [0,1]
    #     optical = torch.tensor(optical, dtype=torch.float32) / 255.0
    #     sar_pre = torch.tensor(sar_pre, dtype=torch.float32) / 255.0
    #     sar_post = torch.tensor(sar_post, dtype=torch.float32) / 255.0

    #     # Convert GT to long (integer labels)
    #     gtd = torch.tensor(gtd, dtype=torch.long)
        

    #     return optical, sar_pre, sar_post, gtd, gtd, self.post_event[idx]
    
    def __getitem__(self, idx):
        # Load post-disaster SAR (1-band → duplicated to 4)
        sar_post = self.load_grayscale(self.post_event[idx]).astype("uint8")
        sar_post = np.stack([sar_post] * 4, axis=0)  # (4, 512, 512)

        # Load pre-disaster optical image (RGB)
        optical = self.load_multiband(self.pre_event_optical[idx]).astype("uint8")

        # ✅ Ensure correct shape (RGB)
        if optical.shape[-1] == 3 and optical.shape[0] != 3:
            optical = np.transpose(optical, (2, 0, 1))  # Convert (H, W, 3) → (3, H, W)

        assert optical.shape == (3, 512, 512), f"Unexpected optical shape: {optical.shape}"

        # Load pre-disaster SAR (1-band)
        sar_pre = self.load_grayscale(self.pre_event_sar[idx]).astype("uint8")
        sar_pre = np.expand_dims(sar_pre, axis=0)  # Ensure (1, 512, 512)

        # ✅ Load Ground Truth (Damage Map)
        gtd = self.load_grayscale(self.target[idx]).astype("uint8")
        num_classes = 4  # {0, 1, 2, 3}
       

        # ✅ Separate Binary (Localization) and Multi-class (Segmentation) Masks
        gtd_loc = (gtd > 0).astype("uint8")  # 1 if building exists, else 0
        gtd_seg = gtd  # Multi-class segmentation (0,1,2,3)
        gtd_seg = torch.tensor(gtd_seg, dtype=torch.long)  # Ensure long dtype
        gtd_seg = F.one_hot(gtd_seg, num_classes=num_classes)  # (512, 512, 4)
        gtd_seg = gtd_seg.permute(2, 0, 1).float()  # Convert to (4, 512, 512)

        # ✅ Convert to Tensors
        optical = torch.tensor(optical, dtype=torch.float32) / 255.0
        sar_pre = torch.tensor(sar_pre, dtype=torch.float32) / 255.0
        sar_post = torch.tensor(sar_post, dtype=torch.float32) / 255.0
        gtd_loc = torch.tensor(gtd_loc, dtype=torch.long)  # Binary (0/1)
        gtd_seg = torch.tensor(gtd_seg, dtype=torch.long)  # Multi-class (0,1,2,3)
        # Convert GT to Long (for CrossEntropyLoss)
        # print(f"Unique values in gtd_seg: {torch.unique(gtd_seg)}")
        
        # num_classes = 4  # Since we have {0,1,2,3}

        # gtd_seg = gtd_seg.squeeze(1)  # Ensure it's (B, H, W)
        # gtd_seg = F.one_hot(gtd_seg.long(), num_classes=num_classes)  # Convert to one-hot (B, H, W, C)
        # gtd_seg = gtd_seg.permute(0, 3, 1, 2).float()  # Reorder to (B, C, H, W)
        return optical, sar_pre, sar_post, gtd_loc, gtd_seg, self.post_event[idx]


                

        # # Debugging prints
        # print(f"✅ Fixed Shapes | Optical: {optical.shape}, SAR Pre: {sar_pre.shape}, SAR Post: {sar_post.shape}, GT: {gtd.shape}")

        # # return optical, sar_pre, sar_post, gtd, gtd, self.post_event[idx]
        # return (
        #     torch.tensor(optical, dtype=torch.float32),
        #     torch.tensor(sar_pre, dtype=torch.float32),
        #     torch.tensor(sar_post, dtype=torch.float32),
        #     torch.tensor(gtd, dtype=torch.long),  # Assuming GT is a label, use long
        #     torch.tensor(gtd, dtype=torch.long),  # Assuming GT is a label, use long
        #     self.post_event[idx]
        # )



    # def __getitem__(self, idx):
    #     # Load post-event SAR image
    #     sar_post = self.load_grayscale(self.post_event[idx]).astype("uint8")

    #     # Load pre-disaster optical image
    #     optical = self.load_multiband(self.pre_event_optical[idx]).astype("uint8")

    #     # Load pre-disaster SAR image
    #     sar_pre = self.load_grayscale(self.pre_event_sar[idx]).astype("uint8")

    #     # Load ground truth mask
    #     gtd = self.load_grayscale(self.target[idx]).astype("uint8")

    #     # Ensure data is correctly loaded
    #     if sar_post is None or optical is None or sar_pre is None or gtd is None:
    #         print(f"❌ Warning: Missing data at index {idx}")
    #         return None  # Skip None samples

    #     # ✅ Ensure `sar_post` has 4 channels
    #     sar_post = np.stack([sar_post] * 4, axis=-1)  # Convert single-channel SAR to 4 bands

    #     # ✅ Ensure `sar_pre` remains single-channel
    #     sar_pre = np.expand_dims(sar_pre, axis=-1)  # Keep single-channel SAR

    #     # Prepare sample
    #     sample = {
    #         "sar_post": sar_post,
    #         "optical": optical,
    #         "sar_pre": sar_pre,
    #         "gtd": gtd
    #     }

    #     return sample  # ✅ Return without augmentation

        
    def __len__(self):
        return len(self.post_event)

        
        
    ##########
    ###chia (for 3 encoder)
    ##########
    
    
    # def __getitem__(self, idx):
    #     # Load images
    #     sar_post = self.load_grayscale(self.post_event[idx]).astype("uint8")
    #     optical = self.load_multiband(self.pre_event_optical[idx]).astype("uint8")
    #     sar_pre = self.load_grayscale(self.pre_event_sar[idx]).astype("uint8")
    #     gtd = self.load_grayscale(self.target[idx]).astype("uint8")  # Ground truth

    #     # Convert single-channel SAR images to 3-channel
    #     sar_post = np.stack([sar_post, sar_post, sar_post], axis=-1)
    #     sar_pre = np.stack([sar_pre, sar_pre, sar_pre], axis=-1)

    #     # Define ground truth masks
    #     gtd_loc = (gtd > 0).astype("uint8")  # Binary mask for building localization
    #     gtd_seg = np.eye(len(np.unique(gtd)))[gtd]  # One-hot encoding for segmentation

    #     # Convert to PyTorch tensors
    #     sample_tensors = self.to_tensor({
    #         "sar_post": sar_post,
    #         "optical": optical,
    #         "sar_pre": sar_pre,
    #         "gtd_loc": gtd_loc,  # ✅ Now correctly included
    #         "gtd_seg": gtd_seg   # ✅ Now correctly included
    #     })

    #     # ✅ Debugging to ensure everything exists
    #     print(f"Index {idx} - Sample Tensors: {sample_tensors.keys()}")

    #     return (
    #         sample_tensors["sar_post"],
    #         sample_tensors["optical"],
    #         sample_tensors["sar_pre"],
    #         sample_tensors["gtd_loc"],  # ✅ Now exists
    #         sample_tensors["gtd_seg"],  #
    #         self.post_event[idx]
    #     )
    
    # def __len__(self):
    #     return len(self.post_event)


        
    ##########
    ###chia (for 3 encoder)
    ##########
    



        # Fuse the two pre-disaster modalities by averaging (to get a 3-channel tensor)
        # pre_disaster = (sample_tensors["optical"] + sample_tensors["sar_pre"]) / 2

        # Return a tuple containing:
        # 1. Post-disaster SAR tensor (3 channels)
        # 2. Fused pre-disaster tensor (3 channels)
        # 3. Ground truth for localization (gtd_loc, shape [1, H, W])
        # 4. Ground truth for segmentation (gtd_seg, shape [num_classes, H, W])
        # 5. The file path (for reference)
        
        #option (early fusion)
        # return (sample_tensors["sar_post"], pre_disaster, 
        #         sample_tensors["gtd_loc"], sample_tensors["gtd_seg"], 
        #         self.post_event[idx])
        
        #option (early fusion)
                # Concatenate SAR (pre), Optical, and SAR (post) into one tensor
        # fused_input = torch.cat([
        #     sample_tensors["sar_post"],  # 3-channel Post-disaster SAR
        #     sample_tensors["optical"],   # 3-channel RGB Optical
        #     sample_tensors["sar_pre"]    # 3-channel Pre-disaster SAR
        # ], dim=0)  # Shape: (9, H, W)

        #Now return only ONE tensor for input
        # return (fused_input,  # (9, H, W)
        #         sample_tensors["gtd_loc"],  # (1, H, W) for building localization
        #         sample_tensors["gtd_seg"],  # (num_classes, H, W) for damage segmentation
        #         self.post_event[idx])
        # print(f"Index {idx}:")
        # print(f"  - SAR Post shape: {sample_tensors['sar_post'].shape if sample_tensors['sar_post'] is not None else 'None'}")
        # print(f"  - Optical shape: {sample_tensors['optical'].shape if sample_tensors['optical'] is not None else 'None'}")
        # print(f"  - SAR Pre shape: {sample_tensors['sar_pre'].shape if sample_tensors['sar_pre'] is not None else 'None'}")
        # print(f"  - Ground Truth shape: {sample_tensors['gtd'].shape if sample_tensors['gtd'] is not None else 'None'}")

        #option (late fusion)
        return (sample_tensors["sar_post"], 
            sample_tensors["optical"], 
            sample_tensors["sar_pre"],
            sample_tensors["gtd_loc"], 
            sample_tensors["gtd_seg"], 
            self.post_event[idx])
        
        
    # def __len__(self):
    #     return len(self.post_event)

    

    
    # def __getitem__(self, idx):
    # # Load post-event SAR image (post-disaster)
    #     sar_post = self.load_grayscale(self.post_event[idx]).astype("uint8")

    #     # Load pre-disaster optical image
    #     optical = self.load_multiband(self.pre_event_optical[idx]).astype("uint8")

    #     # Load pre-disaster SAR image
    #     sar_pre = self.load_grayscale(self.pre_event_sar[idx]).astype("uint8")

    #     # Load ground truth mask
    #     gtd = self.load_grayscale(self.target[idx]).astype("uint8")

    #     # Convert single-channel SAR images to 3-channel (if needed)
    #     sar_post = np.stack([sar_post, sar_post, sar_post], axis=-1)
    #     sar_pre = np.stack([sar_pre, sar_pre, sar_pre], axis=-1)

    #     # Use "gtd" as the key for the ground truth mask.
    #     sample = {
    #         "sar_post": sar_post,   # post-disaster SAR
    #         "optical": optical,     # pre-disaster optical
    #         "sar_pre": sar_pre,     # pre-disaster SAR
    #         "gtd": gtd              # ground truth mask
    #     }

    #     if self.augm:
    #         sample = self.augm(sample, size=self.size)
    #         sar_post = sample["sar_post"]
    #         optical = sample["optical"]
    #         sar_pre = sample["sar_pre"]
    #         # gtd = sample["gtd"]

    #     sample_tensors = self.to_tensor({
    #         "sar_post": sar_post,
    #         "optical": optical,
    #         "sar_pre": sar_pre,
    #         "gtd": gtd
    #     })

    #     pre_disaster = (sample_tensors["optical"] + sample_tensors["sar_pre"]) / 2

    #     # pre_disaster = torch.cat([sample_tensors["optical"], sample_tensors["sar_pre"]], dim=0)
    #     return sample_tensors["sar_post"], pre_disaster, sample_tensors["gtd"], self.post_event[idx]

    # def __len__(self):
    #     return len(self.post_event)



    # def __getitem__(self, idx):
    #     # Load post-event SAR image (post-disaster)
    #     sar_post = self.load_grayscale(self.post_event[idx]).astype("uint8")

    #     # Load pre-disaster optical image
    #     optical = self.load_multiband(self.pre_event_optical[idx]).astype("uint8")

    #     # Load pre-disaster SAR image
    #     sar_pre = self.load_grayscale(self.pre_event_sar[idx]).astype("uint8")

    #     # Load ground truth mask
    #     gtd = self.load_grayscale(self.target[idx]).astype("uint8")

    #     # Convert single-channel SAR images to 3-channel (if needed)g
    #     sar_post = np.stack([sar_post, sar_post, sar_post], axis=-1)
    #     sar_pre = np.stack([sar_pre, sar_pre, sar_pre], axis=-1)

    #     sample = {
    #         "sar_post": sar_post,   # post-disaster SAR
    #         "optical": optical,     # pre-disaster optical
    #         "sar_pre": sar_pre,     # pre-disaster SAR
    #         "target": gtd              # target mask
    #     }

    #     if self.augm:
    #         sample = self.augm(sample, size=self.size)
    #         sar_post = sample["sar_post"]
    #         optical = sample["optical"]
    #         sar_pre = sample["sar_pre"]
    #         gtd = sample["gtd"]

    #     sample_tensors = self.to_tensor({
    #         "sar_post": sar_post,
    #         "optical": optical,
    #         "sar_pre": sar_pre,
    #         "gtd": gtd
    #     })

    #     # # Option A (Late Fusion / Dual Encoder):
    #     # return (
    #     #     sample_tensors["sar_post"],
    #     #     sample_tensors["optical"],
    #     #     sample_tensors["sar_pre"],
    #     #     sample_tensors["gtd"],
    #     #     self.post_event[idx]
    #     # 

    #     # Option B (Early Fusion):
    #     # If you want to fuse the pre-disaster inputs early, you could concatenate optical and SAR:
    #     pre_disaster = torch.cat([sample_tensors["optical"], sample_tensors["sar_pre"]], dim=0)
    #     return sample_tensors["sar_post"], pre_disaster, sample_tensors["gtd"], self.post_event[idx]

    # def __len__(self):
    #     return len(self.post_event)


class BDD2(torch.utils.data.Dataset):

    def __init__(self, fn_list, img_size=512, classes=[0, 1], augm=None):
        # These are the post-event (SAR) file paths
        self.post_event = [str(f) for f in fn_list]

        # Pre-disaster optical: change folder from "post_disaster" to "pre_disaster"
        self.pre_event_optical = [
            f.replace("post_disaster", "pre_disaster")
            for f in self.post_event
        ]
        print("Example post-event path:", self.post_event[0])


        # Pre-disaster SAR: assuming these are stored in a different folder,
        # e.g., "pre_disaster_sar" (adjust this if necessary)
        self.pre_event_sar = [
            f.replace("post_disaster", "pre_disaster_sar")
            for f in self.post_event
        ]
        print("Example pre-event optical path:", self.pre_event_optical[0])
        print("Example pre-event path:", self.pre_event_sar[0])

        # Generate target paths.
        # Here we assume that target images are located by replacing "post_disaster"
        # with "target" and then "target" with "building_damage", also removing any "_sar".
        self.target = [
            f.replace("post_disaster", "target")
             .replace("target", "building_damage")
             .replace("_sar", "")
            for f in self.post_event
        ]

        print("Example target path:", self.target[0])

        self.augm = augm
        self.size = img_size
        self.to_tensor = transforms.ToTensor(classes=classes)
        self.load_multiband = load_multiband  # For optical images (assumed multiband)
        self.load_grayscale = load_grayscale  # For SAR and target (assumed single-channel)

    def __getitem__(self, idx):
        # Load post-event SAR image (for post-disaster branch)
        sar_post = self.load_grayscale(self.post_event[idx]).astype("uint8")

        # Load pre-disaster optical image
        optical = self.load_multiband(self.pre_event_optical[idx]).astype("uint8")

        # Load pre-disaster SAR image
        sar_pre = self.load_grayscale(self.pre_event_sar[idx]).astype("uint8")

        # Load ground truth mask
        gtd = self.load_grayscale(self.target[idx]).astype("uint8")

        # For albumentations, the inputs are expected to be 3-channel.
        # Convert the single-channel SAR images to 3-channel by stacking.
        sar_post = np.stack([sar_post, sar_post, sar_post], axis=-1)
        sar_pre = np.stack([sar_pre, sar_pre, sar_pre], axis=-1)

        # If your augmentation pipeline supports multiple inputs, prepare a dictionary.
        # Here we return separate keys for optical and SAR pre-disaster images.
        sample = {
            "sar_post": sar_post,   # post-disaster SAR
            "optical": optical,     # pre-disaster optical
            "sar_pre": sar_pre,     # pre-disaster SAR
            "gtd": gtd              # target mask
        }

        if self.augm:
            # Ensure your augmentation pipeline handles these keys appropriately.
            sample = self.augm(sample, size=self.size)
            sar_post = sample["sar_post"]
            optical = sample["optical"]
            sar_pre = sample["sar_pre"]
            gtd = sample["gtd"]

        # Convert images/masks to tensors.
        # You may need to adjust your to_tensor function to work with a dictionary.
        sample_tensors = self.to_tensor({
            "sar_post": sar_post,
            "optical": optical,
            "sar_pre": sar_pre,
            "gtd": gtd
        })

        # Option A (Late Fusion / Dual Encoder):
        # Return the tensors separately so your model can process each branch independently.
        # return (
        #     sample_tensors["sar_post"],
        #     sample_tensors["optical"],
        #     sample_tensors["sar_pre"],
        #     sample_tensors["gtd"],
        #     self.post_event[idx]
        # )

        # Option B (Early Fusion):
        # If you want to fuse the two pre-disaster inputs early, you could concatenate them.
        # For example:
        pre_disaster = torch.cat([sample_tensors["optical"], sample_tensors["sar_pre"]], dim=0)
        # and then return:
        return sample_tensors["sar_post"], pre_disaster, sample_tensors["gtd"], self.post_event[idx]

    def __len__(self):
        return len(self.post_event)


 

class DFC25(torch.utils.data.Dataset):

    def __init__(self, fn_list, img_size=512, classes=[0, 1], augm=None):
        self.post_event = [str(f) for f in fn_list]
        self.pre_event = [
            f.replace("post-event", "pre-event").replace(
                "post_disaster", "pre_disaster"
            )
            for f in self.post_event
        ]
        self.target = [
            f.replace("post-event", "target").replace(
                "post_disaster", "building_damage"
            )
            for f in self.post_event
        ]
        self.augm = augm
        self.size = img_size
        self.to_tensor = transforms.ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):

        sar = self.load_grayscale(self.post_event[idx]).astype("uint8")
        rgb = self.load_multiband(self.pre_event[idx]).astype("uint8")
        gtd = self.load_grayscale(self.target[idx]).astype("uint8")

        # make 3-channel sar image beacuse of albumentations
        sar = np.stack([sar, sar, sar], axis=-1)

        if self.augm:
            data = self.augm({"sar": sar, "rgb": rgb, "gtd": gtd}, size=self.size)
            sar, rgb, gtd = data["image"], data["image1"], data["mask"]

        # it includes a simple normalization because of 8-bit data
        sar, rgb, gtd = self.to_tensor({"sar": sar, "rgb": rgb, "gtd": gtd})

        # it makes a builfing footprint mask
        msk = ((torch.argmax(gtd, dim=0) != 0) * 1.0).unsqueeze(0)

        return sar, rgb, gtd, msk, self.post_event[idx]

    def __len__(self):
        return len(self.post_event)


class OpenEarthMapDataset(torch.utils.data.Dataset):
    """
    OpenEarthMap dataset
    Geoinformatics Unit, RIKEN AIP

    Args:
        fn_list (str): List containing images paths
        classes (int): list of of class-code
        img_size (int): image size
        augm (albumentations): transfromation pipeline (e.g. flip, cut, etc.)
    """

    def __init__(
        self,
        img_list,
        classes,
        img_size=512,
        augm=None,
        mu=None,
        sig=None,
    ):
        self.fn_imgs = [str(f) for f in img_list]
        self.fn_msks = [f.replace("/images/", "/labels/") for f in self.fn_imgs]
        self.augm = augm
        self.to_tensor = (
            transforms.ToTensor(classes=classes)
            if mu is None
            else transforms.ToTensorNorm(classes=classes, mu=mu, sig=sig)
        )
        self.size = img_size
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fn_imgs[idx])
        msk = self.load_grayscale(self.fn_msks[idx])

        data = self.to_tensor(self.augm({"image": img, "mask": msk}, self.size))
        return data["image"], data["mask"], self.fn_imgs[idx]

    def __len__(self):
        return len(self.fn_imgs)



################# original BDD########################

# class BDD3(torch.utils.data.Dataset):

#     def __init__(self, fn_list, img_size=512, classes=[0, 1], augm=None):
#         # Post-event (SAR) file paths from the post_disaster folder.
#         self.post_event = [str(f) for f in fn_list]

#         # Pre-disaster Optical:
#         # Change the directory from "post_disaster" to "pre_disaster"
#         # and the filename suffix from "_post_disaster_sar" to "_pre_disaster"
#         self.pre_event_optical = [
#         f.replace("_post_disaster_sar.tif", "_pre_disaster.tif")
#         .replace("post_disaster/", "pre_disaster/")
#         for f in self.post_event
#     ]


#         # Pre-disaster SAR:
#         # Change the directory from "post_disaster" to "pre_disaster"
#         # and the filename suffix from "_post_disaster_sar" to "_pre_disaster_sar"
#         self.pre_event_sar = [
#             f.replace("post_disaster", "pre_disaster")
#              .replace("_post_disaster_sar", "_pre_disaster_sar")
#             for f in self.post_event
#         ]

#         # Generate target paths.
#         # (Assuming your target generation logic remains the same.)
#         self.target = [
#             f.replace("post_disaster", "target")
#              .replace("target", "building_damage")
#              .replace("_sar", "")
#             for f in self.post_event
#         ]

#         # print("Example post-event path:", self.post_event[0])
#         # print("Example pre-event optical path:", self.pre_event_optical[0])
#         # print("Example pre-event SAR path:", self.pre_event_sar[0])
#         # print("Example target path:", self.target[0])

#         self.augm = augm
#         self.size = img_size
#         self.to_tensor = transforms.ToTensor(classes=classes)
#         self.load_multiband = load_multiband  # For optical images (assumed multiband)
#         self.load_grayscale = load_grayscale  # For SAR and target (assumed single-channel)
        
#     def __getitem__(self, idx):
#         # Load post-event SAR image (for post-disaster branch)
#         sar_post = self.load_grayscale(self.post_event[idx]).astype("uint8")


#         # Load pre-disaster optical image
#         optical = self.load_multiband(self.pre_event_optical[idx]).astype("uint8")

#         # Load pre-disaster SAR image
#         sar_pre = self.load_grayscale(self.pre_event_sar[idx]).astype("uint8")

#         # Load ground truth mask
#         gtd = self.load_grayscale(self.target[idx]).astype("uint8")

#         # # Convert single-channel SAR images to 3-channel (if needed)
#         sar_post = np.stack([sar_post, sar_post, sar_post], axis=-1)
#         sar_pre = np.stack([sar_pre, sar_pre, sar_pre], axis=-1)
    

#         # Prepare sample dictionary
#         sample = {
#             "sar_post": sar_post,
#             "optical": optical,
#             "sar_pre": sar_pre,
#             "gtd": gtd
#         }

#         if self.augm:
#             sample = self.augm(sample, size=self.size)
#             sar_post = sample["sar_post"]
#             optical = sample["optical"]
#             sar_pre = sample["sar_pre"]
#             # gtd = sample["gtd"]
  
      




#         sample_tensors = self.to_tensor({
#             "sar_post": sar_post,
#             "optical": optical,
#             "sar_pre": sar_pre,
#             "gtd": gtd
#         })
        
        
#     def __len__(self):
#         return len(self.post_event)
