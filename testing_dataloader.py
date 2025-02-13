
#%%
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import src  # Make sure 'src' is in your Python path

DATA_DIR = "/home/chia/bdd_multimodal-main/BDD_dataset_GAN/test"

# Filter for files that contain "post-event" in their path and have the .tif extension
fn_list = [f for f in Path(DATA_DIR).rglob("*.tif") if "post_disaster" in str(f)]
print(fn_list)
# Instantiate the dataset using your custom BDD3 class
dataset = src.dataset.BDD3(fn_list, img_size=512, augm=None)

# Create the DataLoader for debugging (num_workers=0 ensures that data loading happens in the main process)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# Now you can iterate over the dataloader to check if it loads the correct images.
# for batch in dataloader:
#     optical, sar_pre, sar_post, gtd_loc, gtd_seg, file_path = batch
#     print("Optical shape:", optical.shape)   # Expected: (B, 3, 512, 512)
#     print("SAR Pre shape:", sar_pre.shape)     # Expected: (B, 1, 512, 512)
#     print("SAR Post shape:", sar_post.shape)   # Expected: (B, 4, 512, 512)
#     print("gtd_loc shape:", gtd_loc.shape)       # Expected: (B, 1, 512, 512)
#     print("gtd_seg shape:", gtd_seg.shape)       # Expected: (B, 4, 512, 512) if one-hot or (B, 512, 512) for class indices
#     print("File paths:", file_path)
    
#     break  # Test with the first batch only


for batch in dataloader:
    optical, sar_pre, sar_post, gtd_loc, gtd_seg, file_path = batch
    print("Optical shape:", optical.shape)   # Expected: (B, 3, 512, 512)
    print("SAR Pre shape:", sar_pre.shape)     # Expected: (B, 1, 512, 512)
    print("SAR Post shape:", sar_post.shape)   # Expected: (B, 4, 512, 512)
    print("gtd_loc shape:", gtd_loc.shape)       # Expected: (B, 1, 512, 512)
    print("gtd_seg shape:", gtd_seg.shape)       # Expected: (B, 4, 512, 512) if one-hot, or (B, 512, 512) for class indices

    # Print each file path in the batch
    for fp in file_path:
        print("File path:", fp)
    
    break  # Only print the first batch for testing



    

# %%
