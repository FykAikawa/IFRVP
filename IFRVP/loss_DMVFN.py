import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda")
grid = None

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, mode=None):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.mode = mode

    def forward(self, x, y, mask=None):
        N = x.size(1)
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        if mask is not None:
            loss = loss * mask
        if self.mode == 'sum':
            loss = torch.sum(loss) / N
        else:
            loss = loss.mean()
        return loss


def gauss_kernel(size=5, channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
