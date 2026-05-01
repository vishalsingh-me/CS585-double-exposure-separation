import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(VGGLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False
        self.to(device)

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

def get_perceptual_loss(pred_1, pred_2, source_1, source_2, is_a, vgg_loss):
    """
    Computes VGG perceptual loss given the matched assignments.
    """
    # Using ImageNet normalization means
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred_1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred_1.device)
    
    def normalize(t):
        return (t - mean) / std
        
    p1_matched = torch.where(is_a.view(-1, 1, 1, 1), pred_1, pred_2)
    p2_matched = torch.where(is_a.view(-1, 1, 1, 1), pred_2, pred_1)
    
    p1_norm = normalize(p1_matched)
    p2_norm = normalize(p2_matched)
    s1_norm = normalize(source_1)
    s2_norm = normalize(source_2)
    
    p1_feat = vgg_loss(p1_norm)
    p2_feat = vgg_loss(p2_norm)
    s1_feat = vgg_loss(s1_norm)
    s2_feat = vgg_loss(s2_norm)
    
    loss = 0.0
    for pf, sf in zip(p1_feat, s1_feat):
        loss += F.l1_loss(pf, sf)
    for pf, sf in zip(p2_feat, s2_feat):
        loss += F.l1_loss(pf, sf)
        
    return loss / 2.0

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

_warned_missing_alpha = False


def get_reconstruction_loss(pred_1, pred_2, mixture, alpha_tensor, is_a):
    global _warned_missing_alpha
    mask_valid = alpha_tensor >= 0
    if not mask_valid.any():
        if not _warned_missing_alpha:
            print("Warning: alpha is missing for reconstruction loss; skipping reconstruction term.")
            _warned_missing_alpha = True
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


def output_correlation_loss(pred_1, pred_2, eps=1e-8):
    batch_size = pred_1.size(0)
    x = pred_1.reshape(batch_size, -1)
    y = pred_2.reshape(batch_size, -1)

    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    x = x / (torch.linalg.vector_norm(x, dim=1, keepdim=True) + eps)
    y = y / (torch.linalg.vector_norm(y, dim=1, keepdim=True) + eps)

    return torch.abs((x * y).sum(dim=1)).mean()
