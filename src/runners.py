import numpy as np
import torch
from tqdm import tqdm
from . import metrics


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)


def metric(input, target):
    """
    Args:
        input (tensor): prediction
        target (tensor): reference data

    Returns:
        float: harmonic fscore without including backgorund
    """
    input = torch.softmax(input, dim=1)
    scores = []

    for i in range(1, input.shape[1]):  # background is not included
        ypr = input[:, i, :, :].view(input.shape[0], -1)
        ygt = target[:, i, :, :].view(target.shape[0], -1)
        # scores.append(metrics.fscore(ypr, ygt).item())
        scores.append(metrics.iou(ypr, ygt).item())

    # scores = np.array(scores)
    # return scores.shape[0] / np.sum(1.0 / (scores + 1e-6))
    return np.mean(scores)


def train_epoch(model, optimizer, criterion, dataloader, device="cpu"):

    # banary loss for localization
    loss_loc, loss_dmg = criterion

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()

    iterator = tqdm(
        dataloader, desc="Train", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )
    for x1, x2, y1, y2, *_ in iterator:

        x = torch.cat([x1, x2], dim=1).to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        n = x.shape[0]

        optimizer.zero_grad()
        loc_out, dmg_out = model.forward(x)
        loss = 0.25 * loss_loc(loc_out, y2) + 0.75 * loss_dmg(dmg_out, y1)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=n)

        with torch.no_grad():
            score_meter.update(metric(dmg_out, y1), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs


def valid_epoch(model=None, criterion=None, dataloader=None, device="cpu"):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    *_, loss_dmg = criterion

    iterator = tqdm(
        dataloader, desc="Valid", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )
    for x1, x2, y, *_ in iterator:
        x = torch.cat([x1, x2], dim=1).to(device)
        y = y.to(device)
        n = x.shape[0]

        with torch.no_grad():
            *_, outputs = model.forward(x)
            loss = loss_dmg(outputs, y)

            loss_meter.update(loss.item(), n=n)
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs

# def train_epoch2(model, optimizer, criterion, dataloader, device="cpu"):
#     loss_loc, loss_dmg = criterion

#     loss_meter = AverageMeter()
#     score_meter = AverageMeter()
#     logs = {}

#     model.to(device).train()

#     iterator = tqdm(
#         dataloader, 
#         desc="Train", 
#         bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
#     )
    
#     # Unpack exactly the tuple that __getitem__ returns:
    
#     for sar_post, fused_pre, gtd, file_path in iterator:
#         sar_post = sar_post.to(device)
#         fused_pre = fused_pre.to(device)
#         gtd = gtd.to(device)
        
    

#         n = sar_post.shape[0]  # or use fused_pre.shape[0]

#         optimizer.zero_grad()
 

#         print("sar_post shape:", sar_post.shape)     # Expect [B, 3, H, W]
#         print("fused_pre shape:", fused_pre.shape)     # Expect [B, 3, H, W]
#         x = torch.cat([sar_post, fused_pre], dim=1)
#         print("x shape:", x.shape)                     # Expect [B, 6, H, W]
        
#         x = torch.cat([sar_post, fused_pre], dim=1)
        
    
#         # Forward pass: assume your model returns two outputs.
#         loc_out, dmg_out = model(x)
        
#         # Compute the losses.
#         loss = 0.25 * loss_loc(loc_out, gtd) + 0.75 * loss_dmg(dmg_out, gtd)
#         loss.backward()
#         optimizer.step()

#         loss_meter.update(loss.item(), n=n)

#         with torch.no_grad():
#             score_meter.update(metric(dmg_out, gtd), n=n)

#         logs.update({"Loss": loss_meter.avg, "Score": score_meter.avg})
#         iterator.set_postfix_str(format_logs(logs))
    
#     return logs

################. below is try #########

def train_epoch_back(model, optimizer, criterion, dataloader, device="cpu"):
    loss_loc, loss_dmg = criterion  # Assuming loss_loc is for localization and loss_dmg for segmentation

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()

    iterator = tqdm(
        dataloader, 
        desc="Train", 
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )
    
    
    # # ✅ Unpack exactly the tuple that __getitem__ returns
    # for fused_input, gtd_loc, gtd_seg, file_path in iterator:
    #     fused_input = fused_input.to(device)
    #     gtd_loc = gtd_loc.to(device)  # binary mask for localization
    #     gtd_seg = gtd_seg.to(device)  # one-hot encoded mask for segmentation
        
    #     n = fused_input.shape[0]
    #     print ("fused_input shape:", fused_input.shape)

    #     optimizer.zero_grad()
        
    #     # ✅ Forward pass with a single fused input tensor (9 channels)
    #     output = model(fused_input)  # Expected output: (batch, 1 + num_damage_classes, H, W)

    #     # ✅ Extract building localization & damage segmentation outputs
    #     loc_out = output[:, 0, :, :]  # First channel → Building mask
    #     dmg_out = output[:, 1:, :, :]  # Remaining channels → Damage classification
        
    #     # ✅ Compute losses
    #     loss = 0.25 * loss_loc(loc_out, gtd_loc) + 0.75 * loss_dmg(dmg_out, gtd_seg)


    #     loss.backward()
    #     optimizer.step()

    #     loss_meter.update(loss.item(), n=n)
    #     with torch.no_grad():
    #         score_meter.update(metric(dmg_out, gtd_seg), n=n)

    #     logs.update({"Loss": loss_meter.avg, "Score": score_meter.avg})
    #     iterator.set_postfix_str(format_logs(logs))

    # return logs

    ##################
    ### LATE FUSION ###
    ##################
    # for sar_post, optical, sar_pre, gtd_loc, gtd_seg, file_path in iterator:
    for optical, sar_pre, sar_post, gtd_loc, gtd_seg, file_path in iterator:


        sar_post = sar_post.to(device)
        optical = optical.to(device)
        sar_pre = sar_pre.to(device)
        gtd_loc = gtd_loc.to(device)  # binary mask for localization
        gtd_seg = gtd_seg.to(device)  # one-hot encoded mask for segmentation
        
        n = sar_post.shape[0]

        optimizer.zero_grad()
        # Instead of concatenating fused_pre, pass the separate modalities:
        # Let the model perform late fusion.
        # loc_out, dmg_out = model(sar_post, optical, sar_pre)
        loc_out, dmg_out = model(optical, sar_pre, sar_post)
        print(f"✅ Model Outputs - loc_out: {loc_out.shape}, dmg_out: {dmg_out.shape}")
        print(f"✅ Ground Truth - gtd_loc: {gtd_loc.shape}, gtd_seg: {gtd_seg.shape}")
 
        # Compute losses: use gtd_loc (shape [B, 1, H, W]) for localization,
        # and gtd_seg (shape [B, num_classes, H, W]) for segmentation.
        loss = 0.25 * loss_loc(loc_out, gtd_loc) + 0.75 * loss_dmg(dmg_out, gtd_seg)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=n)
        with torch.no_grad():
            score_meter.update(metric(dmg_out, gtd_seg), n=n)

        logs.update({"Loss": loss_meter.avg, "Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
        # print(f"Shapes | SAR Post: {sar_post.shape}, Optical: {optical.shape}, SAR Pre: {sar_pre.shape}")

    
    return logs

    
    ###################
    ## EARLY FUSION ###
    ###################
    

    # # # Unpack exactly the tuple that __getitem__ returns:(early fusion)
    # for sar_post, fused_pre, gtd_loc, gtd_seg, file_path in iterator:
    #     sar_post = sar_post.to(device)
    #     fused_pre = fused_pre.to(device)
    #     gtd_loc = gtd_loc.to(device)  # binary mask for localization
    #     gtd_seg = gtd_seg.to(device)  # one-hot encoded mask for segmentation
        
    #     n = sar_post.shape[0]

    #     optimizer.zero_grad()
    #     # Concatenate the post-disaster SAR and fused pre-disaster image
    #     # (Here fused_pre is computed in your dataset by averaging optical and sar_pre,
    #     # resulting in a 3-channel tensor.)
    #     x = torch.cat([sar_post, fused_pre], dim=1)  # shape: [B, 6, H, W]
        
    #     # Forward pass; model returns two outputs
    #     loc_out, dmg_out = model(x)
        
    #     # Compute losses: use gtd_loc (shape [B, 1, H, W]) for localization,
    #     # and gtd_seg (shape [B, num_classes, H, W]) for segmentation.
    #     loss = 0.25 * loss_loc(loc_out, gtd_loc) + 0.75 * loss_dmg(dmg_out, gtd_seg)
    #     # loss = 0.3 * loss_loc(loc_out, gtd_loc) + 0.7 * loss_dmg(dmg_out, gtd_seg)

    #     loss.backward()
    #     optimizer.step()

    #     loss_meter.update(loss.item(), n=n)
    #     with torch.no_grad():
    #         score_meter.update(metric(dmg_out, gtd_seg), n=n)

    #     logs.update({"Loss": loss_meter.avg, "Score": score_meter.avg})
    #     iterator.set_postfix_str(format_logs(logs))
    
    # return logs


##################################
### early fusion #################
### contatenate all the tensors###
##################################

def train_epoch2(model, optimizer, criterion, dataloader, device="cpu"):
    loss_loc, loss_dmg = criterion  # Loss functions for localization & segmentation
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()

    print(f"🔄 Starting Training Epoch with {len(dataloader)} batches...")

    iterator = tqdm(
        dataloader, 
        desc="Train", 
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )

    for batch_idx, (optical, sar_pre, sar_post, gtd_loc, gtd_seg, file_path) in enumerate(iterator):
        # print(f"🟢 Processing Batch {batch_idx + 1}/{len(dataloader)}")

        # Move tensors to GPU
        sar_post = sar_post.to(device)
        optical = optical.to(device)
        sar_pre = sar_pre.to(device)
        gtd_loc = gtd_loc.to(device)
        gtd_seg = gtd_seg.to(device)

        n = sar_post.shape[0]

        optimizer.zero_grad()
        gtd_loc = gtd_loc.unsqueeze(1).float()  # Ensure shape (B, 1, H, W)


        try:
            loc_out, dmg_out = model(optical, sar_pre, sar_post)
            # print(f"✅ Model Outputs - loc_out: {loc_out.shape}, dmg_out: {dmg_out.shape}")
            # print(f"✅ Ground Truth - gtd_loc: {gtd_loc.shape}, gtd_seg: {gtd_seg.shape}")
        except Exception as e:
            # print(f"🚨 Model Forward Pass Failed: {e}")
            continue  # Skip this batch if model fails

        # Compute loss
        loss = 0.25 * loss_loc(loc_out, gtd_loc) + 0.75 * loss_dmg(dmg_out, gtd_seg)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=n)
        with torch.no_grad():
            score_meter.update(metric(dmg_out, gtd_seg), n=n)

        logs.update({"Loss": loss_meter.avg, "Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))

    return logs




# def train_epoch2(model, optimizer, criterion, dataloader, device="cpu"):
#     # Unpack your two loss functions (e.g., one for localization and one for damage)
#     loss_loc, loss_dmg = criterion

#     loss_meter = AverageMeter()
#     score_meter = AverageMeter()
#     logs = {}

#     model.to(device).train()

#     iterator = tqdm(
#         dataloader, 
#         desc="Train", 
#         bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
#     )
    
    
#     # Here we assume each batch returns: sar_post, optical, sar_pre, gtd, (optional extra)
#     for sar_post, optical, sar_pre, gtd, *_ in iterator:
#     # for sar_post, optical, sar_pre, target, *_ in iterator:

#         # Move each input to the proper device
#         sar_post = sar_post.to(device)   # post-disaster SAR
#         optical  = optical.to(device)     # pre-disaster optical
#         sar_pre  = sar_pre.to(device)     # pre-disaster SAR
#         gtd      = gtd.to(device)         # ground truth mask
#         # target = target.to(device)

        
#         # Late Fusion:
#         # First, fuse the two pre-disaster inputs (optical and SAR) along the channel dimension.
#         pre_disaster = torch.cat([optical, sar_pre], dim=1)
#         # Then, combine the post-disaster input with the fused pre-disaster input.
#         x = torch.cat([sar_post, pre_disaster], dim=1)
        
#         n = x.shape[0]

#         optimizer.zero_grad()
#         # Forward pass – assume your model returns two outputs: one for localization and one for damage.
#         loc_out, dmg_out = model(x)
        
#         # Compute the losses.
#         # Here we use the same ground truth for both losses.
#         # (If you have separate ground truths for localization and damage, adjust accordingly.)
#         loss = 0.25 * loss_loc(loc_out, gtd) + 0.75 * loss_dmg(dmg_out, gtd)
#         # loss = 0.25 * loss_loc(loc_out, target) + 0.75 * loss_dmg(dmg_out, target)

#         loss.backward()
#         optimizer.step()

#         loss_meter.update(loss.item(), n=n)

#         with torch.no_grad():
#             score_meter.update(metric(dmg_out, gtd), n=n)
#             # score_meter.update(metric(dmg_out, target), n=n)

            
            
            

#         logs.update({"Loss": loss_meter.avg, "Score": score_meter.avg})
#         iterator.set_postfix_str(format_logs(logs))
    
#     return logs

# def valid_epoch2(model=None, criterion=None, dataloader=None, device="cpu"):
#     loss_meter = AverageMeter()
#     score_meter = AverageMeter()
#     logs = {}
#     model.to(device).eval()

#     # We assume criterion is a tuple like (loss_loc, loss_dmg)
#     # For validation we use only loss_dmg.
#     *_, loss_dmg = criterion

#     iterator = tqdm(
#         dataloader, 
#         desc="Valid", 
#         bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
#     )
    
#     # Loop over the validation dataloader.
#     # We expect each batch to return: sar_post, optical, sar_pre, gtd, (optional extras)
#     for sar_post, optical, sar_pre, gtd, *_ in iterator:
#         # Move the inputs to the device.
#         sar_post = sar_post.to(device)
#         optical  = optical.to(device)
#         sar_pre  = sar_pre.to(device)
#         gtd      = gtd.to(device)
#         n = sar_post.shape[0]

#         # Fuse the two pre-disaster modalities (optical and SAR) along the channel dimension.
#         pre_disaster = torch.cat([optical, sar_pre], dim=1)
#         # Then, fuse with the post-disaster SAR image.
#         x = torch.cat([sar_post, pre_disaster], dim=1)

#         with torch.no_grad():
#             # Assume the model returns two outputs, e.g., loc_out and dmg_out,
#             # and that for validation we use the damage output.
#             *_, outputs = model(x)
#             loss = loss_dmg(outputs, gtd)

#             loss_meter.update(loss.item(), n=n)
#             score_meter.update(metric(outputs, gtd), n=n)

#         logs.update({"Loss": loss_meter.avg})
#         logs.update({"Score": score_meter.avg})
#         iterator.set_postfix_str(format_logs(logs))
    
#     return logs
  ##############below is try ############
  
def valid_epoch2(model=None, criterion=None, dataloader=None, device="cpu"):
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    # We assume criterion is a tuple like (loss_loc, loss_dmg)
    # For validation we use only loss_dmg.
    *_, loss_dmg = criterion

    iterator = tqdm(
        dataloader, 
        desc="Valid", 
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )
      ###################
    ## EARLY FUSION ###
    ###################
    
    # # Expect each batch to return: sar_post, fused_pre, gtd_loc, gtd_seg, file_path
    # for sar_post, fused_pre, gtd_loc, gtd_seg, file_path in iterator:
    #     # Move the inputs to the device.
    #     sar_post = sar_post.to(device)
    #     fused_pre = fused_pre.to(device)
    #     gtd_seg = gtd_seg.to(device)  # use segmentation ground truth here
    #     n = sar_post.shape[0]

    #     # Concatenate the post-disaster and fused pre-disaster images.
    #     x = torch.cat([sar_post, fused_pre], dim=1)  # Expected shape: [B, 6, H, W]

    #     with torch.no_grad():
    #         *_, outputs = model(x)
    #         loss = loss_dmg(outputs, gtd_seg)

    #         loss_meter.update(loss.item(), n=n)
    #         score_meter.update(metric(outputs, gtd_seg), n=n)

    #     logs.update({"Loss": loss_meter.avg})
    #     logs.update({"Score": score_meter.avg})
    #     iterator.set_postfix_str(format_logs(logs))
    
    # return logs
   ###################
    ## LATE FUSION  ###
    ###################
    


    for sar_post, optical, sar_pre, gtd_loc, gtd_seg, file_path in iterator:
        # Move the inputs to the device.
        sar_post = sar_post.to(device)
        optical = optical.to(device)
        sar_pre = sar_pre.to(device)
        gtd_seg = gtd_seg.to(device)  # Use segmentation ground truth here
        n = sar_post.shape[0]

        with torch.no_grad():
            # Forward pass with separate inputs; model should handle the late fusion internally.
            loc_out, dmg_out = model(sar_post, optical, sar_pre)
            loss = loss_dmg(dmg_out, gtd_seg)

            loss_meter.update(loss.item(), n=n)
            score_meter.update(metric(dmg_out, gtd_seg), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    
    return logs
