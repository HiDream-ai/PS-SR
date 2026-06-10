import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, in_channels=16, pretrained=True):
        super().__init__()
        # 加载 VGG16
        vgg16 = models.vgg16()

        # 改输入层：原来是 3 通道 → 16 通道
        vgg16.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        self.feature_extractor = vgg16.features  # backbone
        self.avgpool = vgg16.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 1)
        )

    def reconstruct_feature_extractor(self, use_pool=False):
        """
        根据 use_pool 构建特征提取器
        - use_pool=True：保留所有池化层（用于训练，全局判别）
        - use_pool=False：去除池化层（用于推理，pixel-wise score）
        """
        layers = []
        for layer in self.feature_extractor:
            if isinstance(layer, nn.MaxPool2d):
                if use_pool:
                    layers.append(layer)
                else:
                    # 去掉池化层，保持空间分辨率
                    continue
            else:
                layers.append(layer)
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        """
        x: [B, C=16, T, H, W]
        return_features: 是否返回中间特征
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

        feat = self.feature_extractor(x)   # [B*T, 512, H', W']
        pooled = self.avgpool(feat)        # [B*T, 512, 7, 7]
        flat = torch.flatten(pooled, 1)    # [B*T, 512*7*7]
        score = self.classifier(flat)      # [B*T, 1]

        score = score.view(B, T, 1).mean(dim=1)  # [B,1]

        if return_features:
            return score, feat              # 同时返回特征
        else:
            return score

    def forward_pixel_score(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

        feat = self.feature_extractor(x)   # [B*T, 512, H', W']
        B_T, C_feat, Hf, Wf = feat.shape

        # === 滑窗提取局部 patch ===
        kernel_size = 7
        padding = kernel_size // 2   # 保持尺寸
        stride = 1

        # unfold 提取滑窗 patch
        patches = F.unfold(feat, kernel_size=kernel_size, padding=padding, stride=stride)
        # patches: [B*T, 512*7*7, H'*W']

        # 转换成 [B*T*H'*W', 512*7*7]
        patches = patches.transpose(1, 2).reshape(-1, C_feat * kernel_size * kernel_size)

        # 送入 classifier
        score_flat = self.classifier(patches)  # [B*T*H'*W', 1]

        # reshape 回 feature map
        score_map = score_flat.view(B_T, Hf, Wf, 1).permute(0, 3, 1, 2)  # [B*T, 1, H', W']

        # 恢复 batch/time 结构
        score_map = score_map.view(B, T, 1, Hf, Wf)  # [B, T, 1, H', W']

        score_map = score_map.permute(0, 2, 1, 3, 4)  # [B, 1, T, H', W']

        threshold = -1.0
        score_map = (score_map <= threshold).int()  # 二值化

        return score_map
