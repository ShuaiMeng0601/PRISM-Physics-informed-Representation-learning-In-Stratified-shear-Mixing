import math

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIABLES = ["u", "v", "w", "th", "epsilon"]


def random_mask_tokens(tokens, mask_token, mask_prob):
    """Randomly replace ViT tokens by a learned mask token."""
    if mask_prob <= 0:
        return tokens

    batch, n_tokens, dim = tokens.shape
    keep = torch.rand(batch, n_tokens, device=tokens.device) > mask_prob
    keep = keep.unsqueeze(-1)
    mask = mask_token.expand(batch, n_tokens, dim)

    return torch.where(keep, tokens, mask)


class PatchEmbed(nn.Module):
    """Patchify one 2D variable field with a Conv2d."""

    def __init__(self, img_size=(501, 100), patch_size=(10, 10), dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        h, w = img_size
        py, px = patch_size

        self.pad_h = math.ceil(h / py) * py - h
        self.pad_w = math.ceil(w / px) * px - w
        self.grid_h = (h + self.pad_h) // py
        self.grid_w = (w + self.pad_w) // px
        self.n_patches = self.grid_h * self.grid_w

        self.proj = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: batch, 1, y, time
        x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VariableViT(nn.Module):
    """A small ViT encoder for one physical variable."""

    def __init__(self, img_size=(501, 100), patch_size=(10, 10), dim=64, depth=2, heads=4):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.patch.n_patches, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x, mask_prob=0.0):
        tokens = self.patch(x)
        tokens = tokens + self.pos
        tokens = random_mask_tokens(tokens, self.mask_token, mask_prob)
        tokens = self.encoder(tokens)
        return tokens


class CrossVariableAttention(nn.Module):
    """Attention across all variable tokens."""

    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens, return_attention=False):
        # tokens shape: batch, all_variable_tokens (#of var * dim per var), dim
        attn_out, attn_weights = self.attn(
            tokens, #q
            tokens, #k
            tokens, #v
            need_weights=return_attention,
            average_attn_weights=False,
        )
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.mlp(tokens))

        if return_attention:
            return tokens, attn_weights
        return tokens, None


class SharedEncoder(nn.Module):
    """Variable-specific ViTs, then cross-variable attention."""

    def __init__(
        self,
        img_size=(501, 100),
        patch_size=(10, 10),
        dim=64,
        depth=2,
        heads=4,
        out_channels=64,
        n_vars=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_vars = n_vars or len(VARIABLES)

        self.variable_vits = nn.ModuleList([
            VariableViT(img_size, patch_size, dim, depth, heads)
            for _ in range(self.n_vars)
        ])
        self.cross = CrossVariableAttention(dim, heads)
        self.var_embed = nn.Parameter(torch.zeros(1, self.n_vars, 1, dim))

        grid_h = self.variable_vits[0].patch.grid_h
        grid_w = self.variable_vits[0].patch.grid_w
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.to_feature = nn.Sequential(
            nn.Conv2d(self.n_vars * dim, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, mask_prob=0.0, return_attention=False):
        # x shape: batch, n_vars, y, time
        if x.shape[1] != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} input variables, got {x.shape[1]}")

        all_tokens = []
        for i, vit in enumerate(self.variable_vits): # 5ViT
            one_var = x[:, i:i + 1]
            tokens = vit(one_var, mask_prob=mask_prob)
            all_tokens.append(tokens)

        tokens = torch.stack(all_tokens, dim=1)
        tokens = tokens + self.var_embed

        batch, n_vars, n_patches, dim = tokens.shape
        tokens = tokens.reshape(batch, n_vars * n_patches, dim)

        tokens, attn = self.cross(tokens, return_attention=return_attention)

        tokens = tokens.reshape(batch, n_vars, n_patches, dim)
        tokens = tokens.reshape(batch, n_vars, self.grid_h, self.grid_w, dim)
        tokens = tokens.permute(0, 1, 4, 2, 3).reshape(
            batch,
            n_vars * dim,
            self.grid_h,
            self.grid_w,
        )

        feature = self.to_feature(tokens)
        feature = F.interpolate(feature, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return feature, attn


class RegressionHead(nn.Module):
    """Predict mean and log variance of pseudo momentum ratio."""

    def __init__(self, in_channels=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, feature):
        out = self.cnn(feature)
        out = self.mlp(out)
        mu = out[:, 0:1]
        log_var = torch.clamp(out[:, 1:2], min=-8.0, max=8.0)
        return mu, log_var


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """Small UNet used for conditional flow matching on theta."""

    def __init__(self, feature_channels=64, cond_dim=3):
        super().__init__()
        # input = noisy theta + encoder feature + Ri/a/Re condition maps + flow time map
        in_ch = 1 + feature_channels + cond_dim + 1

        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.mid = ConvBlock(128, 128)
        self.up1 = ConvBlock(128 + 64, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, noisy_theta, feature, condition, flow_time):
        batch, _, h, w = noisy_theta.shape

        cond_map = condition[:, :, None, None].expand(batch, condition.shape[1], h, w)
        time_map = flow_time[:, None, None, None].expand(batch, 1, h, w)

        x = torch.cat([noisy_theta, feature, cond_map, time_map], dim=1)

        skip = self.down1(x)
        x = F.avg_pool2d(skip, kernel_size=2)
        x = self.down2(x)
        x = self.mid(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up1(x)
        return self.out(x)


class FullModel(nn.Module):
    """Full framework: encoder + regression head + conditional flow branch."""

    def __init__(self, dim=64, patch_size=(10, 10), mask_prob=0.15, n_vars=None, img_size=(501, 100)):
        super().__init__()
        self.mask_prob = mask_prob
        self.encoder = SharedEncoder(img_size=img_size, dim=dim, patch_size=patch_size, out_channels=64, n_vars=n_vars)
        self.regression = RegressionHead(in_channels=64)
        self.flow_unet = TinyUNet(feature_channels=64, cond_dim=3)

    def encode(self, x, return_attention=False):
        mask_prob = self.mask_prob if self.training else 0.0
        return self.encoder(x, mask_prob=mask_prob, return_attention=return_attention)

    def forward_regression(self, x, return_attention=False):
        feature, attn = self.encode(x, return_attention=return_attention)
        mu, log_var = self.regression(feature)
        return mu, log_var, feature, attn

    def forward_flow(self, theta, feature, condition):
        batch = theta.shape[0]
        noise = torch.randn_like(theta)
        flow_time = torch.rand(batch, device=theta.device)

        t = flow_time[:, None, None, None]
        noisy_theta = (1.0 - t) * noise + t * theta

        # Conditional flow matching target velocity.
        target_velocity = theta - noise
        pred_velocity = self.flow_unet(noisy_theta, feature, condition, flow_time)

        # One-step estimate of theta at t=1. This is useful for reconstruction MSE.
        theta_hat = noisy_theta + (1.0 - t) * pred_velocity

        return pred_velocity, target_velocity, theta_hat


def gaussian_nll_loss(mu, log_var, target):
    var = torch.exp(log_var)
    return 0.5 * (log_var + (target - mu) ** 2 / var).mean()
