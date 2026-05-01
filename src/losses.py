import torch
import torch.nn as nn
import torch.nn.functional as F

def l1_loss(pred, target):
    return F.l1_loss(pred, target)

def permutation_invariant_l1_loss(pred_1, pred_2, source_1, source_2):
    batch_size = pred_1.size(0)
    
    loss_a_1 = F.l1_loss(pred_1, source_1, reduction='none').view(batch_size, -1).mean(dim=1)
    loss_a_2 = F.l1_loss(pred_2, source_2, reduction='none').view(batch_size, -1).mean(dim=1)
    loss_a = loss_a_1 + loss_a_2
    
    loss_b_1 = F.l1_loss(pred_1, source_2, reduction='none').view(batch_size, -1).mean(dim=1)
    loss_b_2 = F.l1_loss(pred_2, source_1, reduction='none').view(batch_size, -1).mean(dim=1)
    loss_b = loss_b_1 + loss_b_2
    
    # Take min element-wise
    min_loss = torch.minimum(loss_a, loss_b)
    
    final_loss = min_loss.mean()
    
    return final_loss, loss_a, loss_b, min_loss == loss_a

def get_reconstruction_loss(pred_1, pred_2, mixture, alpha_tensor, is_a):
    mask_valid = alpha_tensor >= 0
    if not mask_valid.any():
        return torch.tensor(0.0, device=pred_1.device)
        
    alpha = alpha_tensor.view(-1, 1, 1, 1)
    
    recon_a = alpha * pred_1 + (1 - alpha) * pred_2
    recon_b = alpha * pred_2 + (1 - alpha) * pred_1
    
    # is_a indicates if pred_1 matches source_1 and pred_2 matches source_2 (shape [B, 1, 1, 1])
    is_a_expanded = is_a.view(-1, 1, 1, 1)
    recon_matched = torch.where(is_a_expanded, recon_a, recon_b)
    
    valid_recons = recon_matched[mask_valid]
    valid_mixtures = mixture[mask_valid]
    
    if valid_recons.numel() == 0:
        return torch.tensor(0.0, device=pred_1.device)
        
    return F.l1_loss(valid_recons, valid_mixtures)
