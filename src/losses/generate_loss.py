import torch
import torch.nn as nn
import torch.optim as optim

from einops import repeat

__all__ = [
    "generate_loss",
    "generate_optim",
    "generate_scheduler",
]

def generate_loss(loss_name):
    if loss_name == 'MSE':
        criterion = nn.MSELoss()
    elif loss_name == "MaskedMSE":
        criterion = MaskedMSELoss()
    elif loss_name == "RMSE":
        criterion = RMSELoss()
    elif loss_name == "MaskedRMSE":
        criterion = MaskedRMSELoss()
    elif loss_name == "L1":
        criterion = nn.L1Loss()
    elif loss_name == "MaskedL1":
        criterion = MaskedL1Loss()
    elif loss_name == "Reprojection":
        criterion = ReprojectionLoss()
    elif loss_name == "Reprojection_t":
        criterion = ReprojectionLoss_t()
    elif loss_name == "Smooth":
        criterion = SmoothLoss()
    else:
        raise ValueError(f'unsupport criterion {loss_name}')
    return criterion

def generate_optim(optim_name, lr, weight_decay, net):
    if optim_name == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optim_name == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optim_name == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f'unsupport optimizer {optim_name}')
    return optimizer

def generate_scheduler(opt, mode='min', factor=0.1, patience=10, verbose=False):
    return optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode, factor=factor, patience=patience, verbose=verbose)

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, sky_mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        if pred.size()[1] == 3:
            sky_mask = repeat(sky_mask, 'b 1 h w -> b 3 h w')
        diff = target[~sky_mask] - pred[~sky_mask]
        return (diff ** 2).mean()
    
class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, sky_mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        if pred.size()[1] == 3:
            sky_mask = repeat(sky_mask, 'b 1 h w -> b 3 h w')
        diff = target[~sky_mask] - pred[~sky_mask]
        return diff.abs().mean()
    
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))

class MaskedRMSELoss(nn.Module):
    def __init__(self):
        super(MaskedRMSELoss, self).__init__()
        self.mse = MaskedMSELoss()

    def forward(self, pred, target, sky_mask):
        return torch.sqrt(self.mse(pred, target, sky_mask))
    
class SSIMLoss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class SmoothLoss(nn.Module):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    def __init__(self):
        super(SmoothLoss, self).__init__()
    
    def forward(self, disp, img):
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), axis=1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), axis=1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

class ReprojectionLoss(nn.Module):
    """Computes reprojection loss between a batch of predicted and target images
    """
    def __init__(self):
        super(ReprojectionLoss, self).__init__()
        self.ssim = SSIMLoss()
        
    def forward(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(axis=1, keepdim=True)

        ssim_loss = self.ssim(pred, target).mean(axis=1, keepdim=True)
        
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss.mean()
    
class ReprojectionLoss_t(nn.Module):
    """Computes reprojection loss between a batch of predicted and target images
    """
    def __init__(self):
        super(ReprojectionLoss_t, self).__init__()
        self.ssim = SSIMLoss()
        
    def forward(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(axis=1, keepdim=True)

        ssim_loss = self.ssim(pred, target).mean(axis=1, keepdim=True)
        
        reprojection_loss = 0.15 * ssim_loss + 0.85 * l1_loss

        return reprojection_loss.mean()