import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import RegressionHead, TinyUNet, random_mask_tokens


class ScalePatchEmbed(nn.Module):
    """Patchify every physical variable independently at one spatial scale."""

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
        # x shape: batch, n_vars, y, time
        batch, n_vars, h, w = x.shape
        x = x.reshape(batch * n_vars, 1, h, w)
        x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x.reshape(batch, n_vars, self.n_patches, -1)


class RelativePositionBias2D(nn.Module):
    """Learned 2D relative position bias shared across variables."""

    def __init__(self, grid_h, grid_w, heads):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.heads = heads
        self.table = nn.Parameter(torch.zeros((2 * grid_h - 1) * (2 * grid_w - 1), heads))

        coords_h = torch.arange(grid_h)
        coords_w = torch.arange(grid_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords = coords.flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel[0] += grid_h - 1
        rel[1] += grid_w - 1
        rel[0] *= 2 * grid_w - 1
        index = rel[0] + rel[1]
        self.register_buffer("index", index.long(), persistent=False)

    def forward(self):
        bias = self.table[self.index.reshape(-1)]
        bias = bias.reshape(self.grid_h * self.grid_w, self.grid_h * self.grid_w, self.heads)
        return bias.permute(2, 0, 1)


class ScaleVariableAttention(nn.Module):
    """Self-attention over (space patch, variable) tokens with a variable graph bias."""

    def __init__(
        self,
        dim=64,
        heads=4,
        n_vars=5,
        grid_size=(51, 10),
        graph_bias_weight=1.0,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")

        self.dim = dim
        self.heads = heads
        self.n_vars = n_vars
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.graph_bias_weight = graph_bias_weight

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel_pos = RelativePositionBias2D(grid_size[0], grid_size[1], heads)

        self.graph_logits = nn.Parameter(torch.zeros(heads, n_vars, n_vars))
        with torch.no_grad():
            eye = torch.eye(n_vars).unsqueeze(0)
            self.graph_logits.add_(eye)

    def _variable_bias(self, n_patches):
        var_ids = torch.arange(self.n_vars, device=self.graph_logits.device)
        var_ids = var_ids.repeat(n_patches)
        log_graph = F.log_softmax(self.graph_logits, dim=-1)
        return log_graph[:, var_ids[:, None], var_ids[None, :]]

    def _spatial_bias(self):
        patch_bias = self.rel_pos()
        return patch_bias.repeat_interleave(self.n_vars, dim=1).repeat_interleave(self.n_vars, dim=2)

    def variable_graph(self):
        return F.softmax(self.graph_logits, dim=-1)

    def forward(self, tokens, return_attention=False):
        # tokens shape: batch, n_vars, n_patches, dim
        batch, n_vars, n_patches, dim = tokens.shape
        if n_vars != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {n_vars}")

        tokens = tokens.transpose(1, 2).reshape(batch, n_patches * n_vars, dim)
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(batch, -1, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores + self._spatial_bias().unsqueeze(0)
        scores = scores + self.graph_bias_weight * self._variable_bias(n_patches).unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, n_patches * n_vars, dim)
        out = self.proj_drop(self.proj(out))
        out = out.reshape(batch, n_patches, n_vars, dim).transpose(1, 2)

        if return_attention:
            return out, attn
        return out, None


class ScaleVariableBlock(nn.Module):
    """Transformer block for one scale."""

    def __init__(
        self,
        dim=64,
        heads=4,
        n_vars=5,
        grid_size=(51, 10),
        mlp_ratio=4,
        graph_bias_weight=1.0,
        drop=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ScaleVariableAttention(
            dim=dim,
            heads=heads,
            n_vars=n_vars,
            grid_size=grid_size,
            graph_bias_weight=graph_bias_weight,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, tokens, return_attention=False):
        attn_out, attn = self.attn(self.norm1(tokens), return_attention=return_attention)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens, attn

    def variable_graph(self):
        return self.attn.variable_graph()


class ScaleVariableStage(nn.Module):
    """Patch embedding plus scale-variable Transformer blocks for one scale."""

    def __init__(
        self,
        img_size=(501, 100),
        patch_size=(10, 10),
        dim=64,
        depth=2,
        heads=4,
        n_vars=5,
        graph_bias_weight=1.0,
        drop=0.0,
    ):
        super().__init__()
        self.patch = ScalePatchEmbed(img_size=img_size, patch_size=patch_size, dim=dim)
        self.pos = nn.Parameter(torch.zeros(1, 1, self.patch.n_patches, dim))
        self.var_embed = nn.Parameter(torch.zeros(1, n_vars, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.blocks = nn.ModuleList([
            ScaleVariableBlock(
                dim=dim,
                heads=heads,
                n_vars=n_vars,
                grid_size=(self.patch.grid_h, self.patch.grid_w),
                graph_bias_weight=graph_bias_weight,
                drop=drop,
            )
            for _ in range(depth)
        ])

    @property
    def grid_size(self):
        return self.patch.grid_h, self.patch.grid_w

    def forward(self, x, mask_prob=0.0, return_attention=False):
        tokens = self.patch(x)
        tokens = tokens + self.pos + self.var_embed

        if mask_prob > 0:
            batch, n_vars, n_patches, dim = tokens.shape
            flat = tokens.reshape(batch, n_vars * n_patches, dim)
            flat_mask = self.mask_token.reshape(1, 1, dim)
            flat = random_mask_tokens(flat, flat_mask, mask_prob)
            tokens = flat.reshape(batch, n_vars, n_patches, dim)

        last_attn = None
        for block_id, block in enumerate(self.blocks):
            want_attn = return_attention and block_id == len(self.blocks) - 1
            tokens, attn = block(tokens, return_attention=want_attn)
            if want_attn:
                last_attn = attn

        return tokens, last_attn

    def variable_graphs(self):
        return torch.stack([block.variable_graph() for block in self.blocks], dim=0)


class GatedScaleFusion(nn.Module):
    """Fuse scale features after resizing them to the original snapshot size."""

    def __init__(self, channels=64, n_scales=3):
        super().__init__()
        self.gate = nn.Conv2d(channels * n_scales, n_scales, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, features):
        # features: list of batch, channels, height, width tensors with the same size
        stacked = torch.stack(features, dim=1)
        gate_logits = self.gate(torch.cat(features, dim=1))
        weights = F.softmax(gate_logits, dim=1).unsqueeze(2)
        fused = (weights * stacked).sum(dim=1)
        return self.refine(fused), weights.squeeze(2)


class ScaleVariableEncoder(nn.Module):
    """Scale-Variable ViT encoder for multivariate 2D snapshots."""

    def __init__(
        self,
        img_size=(501, 100),
        patch_sizes=((10, 10), (20, 20), (40, 40)),
        dim=64,
        depth=2,
        heads=4,
        out_channels=64,
        n_vars=5,
        graph_bias_weight=1.0,
        drop=0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_sizes = tuple(patch_sizes)
        self.dim = dim
        self.out_channels = out_channels
        self.n_vars = n_vars

        self.stages = nn.ModuleList([
            ScaleVariableStage(
                img_size=img_size,
                patch_size=patch_size,
                dim=dim,
                depth=depth,
                heads=heads,
                n_vars=n_vars,
                graph_bias_weight=graph_bias_weight,
                drop=drop,
            )
            for patch_size in self.patch_sizes
        ])
        self.scale_embed = nn.Parameter(torch.zeros(1, len(self.stages), 1, 1, dim))
        self.to_feature = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_vars * dim, out_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GELU(),
            )
            for _ in self.stages
        ])
        self.fusion = GatedScaleFusion(channels=out_channels, n_scales=len(self.stages))

    def forward(self, x, mask_prob=0.0, return_attention=False):
        # x shape: batch, n_vars, y, time
        if x.shape[1] != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} input variables, got {x.shape[1]}")

        scale_features = []
        attentions = {} if return_attention else None

        for scale_id, (stage, projector) in enumerate(zip(self.stages, self.to_feature)):
            tokens, attn = stage(x, mask_prob=mask_prob, return_attention=return_attention)
            tokens = tokens + self.scale_embed[:, scale_id]

            batch, n_vars, n_patches, dim = tokens.shape
            grid_h, grid_w = stage.grid_size
            feature = tokens.reshape(batch, n_vars, grid_h, grid_w, dim)
            feature = feature.permute(0, 1, 4, 2, 3).reshape(batch, n_vars * dim, grid_h, grid_w)
            feature = projector(feature)
            feature = F.interpolate(feature, size=x.shape[-2:], mode="bilinear", align_corners=False)
            scale_features.append(feature)

            if return_attention:
                attentions[f"scale_{scale_id}"] = attn

        feature, scale_weights = self.fusion(scale_features)
        if return_attention:
            attentions["scale_weights"] = scale_weights
            attentions["variable_graphs"] = self.variable_graphs()
        return feature, attentions

    def variable_graphs(self):
        return {
            f"scale_{scale_id}": stage.variable_graphs()
            for scale_id, stage in enumerate(self.stages)
        }


class ScaleVariableFullModel(nn.Module):
    """Full framework using ScaleVariableEncoder in place of the original SharedEncoder."""

    def __init__(
        self,
        dim=64,
        patch_sizes=((10, 10), (20, 20), (40, 40)),
        mask_prob=0.15,
        n_vars=5,
        img_size=(501, 100),
        depth=2,
        heads=4,
        graph_bias_weight=1.0,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.encoder = ScaleVariableEncoder(
            img_size=img_size,
            patch_sizes=patch_sizes,
            dim=dim,
            depth=depth,
            heads=heads,
            out_channels=64,
            n_vars=n_vars,
            graph_bias_weight=graph_bias_weight,
        )
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
        target_velocity = theta - noise
        pred_velocity = self.flow_unet(noisy_theta, feature, condition, flow_time)
        theta_hat = noisy_theta + (1.0 - t) * pred_velocity
        return pred_velocity, target_velocity, theta_hat


def epsilon_mask_schedule(epoch, max_epochs, start=0.5, end=0.9):
    """Linear curriculum for increasing no-epsilon training probability."""
    if epoch is None or max_epochs is None or max_epochs <= 1:
        return end
    progress = min(max(epoch - 1, 0) / (max_epochs - 1), 1.0)
    return start + (end - start) * progress


def spatial_gradient_l1(pred, target):
    """L1 loss on first-order spatial gradients for sharper epsilon fields."""
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    return F.l1_loss(pred_dy, target_dy) + F.l1_loss(pred_dx, target_dx)


class EpsilonFieldHead(nn.Module):
    """Decode encoder features into a 2D epsilon snapshot."""

    def __init__(self, in_channels=64, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature):
        return self.net(feature)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalUNetFlow(nn.Module):
    """UNet velocity field for conditional flow matching."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        feature_channels=64,
        cond_dim=3,
        base_channels=64,
        channel_mults=(1, 2, 4),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels
        self.cond_dim = cond_dim

        total_in = in_channels + feature_channels + cond_dim + 1
        channels = [base_channels * mult for mult in channel_mults]

        self.input = UNetBlock(total_in, channels[0])
        self.down_blocks = nn.ModuleList()
        for idx in range(1, len(channels)):
            self.down_blocks.append(UNetBlock(channels[idx - 1], channels[idx]))

        self.mid = UNetBlock(channels[-1], channels[-1])

        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        current = rev_channels[0]
        for skip_channels in rev_channels[1:]:
            self.up_blocks.append(UNetBlock(current + skip_channels, skip_channels))
            current = skip_channels

        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, noisy_field, feature, condition, flow_time):
        batch, _, h, w = noisy_field.shape
        cond_map = condition[:, :, None, None].expand(batch, condition.shape[1], h, w)
        time_map = flow_time[:, None, None, None].expand(batch, 1, h, w)
        x = torch.cat([noisy_field, feature, cond_map, time_map], dim=1)

        skips = []
        x = self.input(x)
        skips.append(x)
        for block in self.down_blocks:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            x = block(x)
            skips.append(x)

        x = self.mid(x)

        for block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.out(x)


class EpsilonDropoutScaleVariableModel(nn.Module):
    """
    SV-ViT model with epsilon modality dropout and an explicit epsilon prediction head.

    Expected input channel layout:
        x[:, epsilon_index] is the true epsilon channel when it is available.

    During training, epsilon is randomly zeroed with probability epsilon_mask_prob.
    The model still predicts epsilon from the remaining observable channels, and the
    epsilon loss is intended to be computed mainly on masked-epsilon samples.
    """

    def __init__(
        self,
        dim=64,
        patch_sizes=((10, 10), (20, 20), (40, 40)),
        mask_prob=0.15,
        epsilon_mask_prob=0.7,
        epsilon_index=4,
        n_vars=5,
        img_size=(501, 100),
        depth=2,
        heads=4,
        graph_bias_weight=1.0,
        use_epsilon_mask_channel=True,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.epsilon_mask_prob = epsilon_mask_prob
        self.epsilon_index = epsilon_index
        self.n_vars = n_vars
        self.use_epsilon_mask_channel = use_epsilon_mask_channel

        encoder_vars = n_vars + int(use_epsilon_mask_channel)
        self.encoder = ScaleVariableEncoder(
            img_size=img_size,
            patch_sizes=patch_sizes,
            dim=dim,
            depth=depth,
            heads=heads,
            out_channels=64,
            n_vars=encoder_vars,
            graph_bias_weight=graph_bias_weight,
        )
        self.regression = RegressionHead(in_channels=64)
        # Deterministic epsilon regression is intentionally disabled for now.
        # Epsilon is handled by the generative flow head in both training and inference.
        # self.epsilon_head = EpsilonFieldHead(in_channels=64)
        self.flow_unet = TinyUNet(feature_channels=64, cond_dim=3)
        self.epsilon_flow_unet = ConditionalUNetFlow(feature_channels=64, cond_dim=3)

    def _epsilon_available(self, x, variable_mask=None):
        batch = x.shape[0]
        device = x.device
        available = torch.ones(batch, dtype=torch.bool, device=device)
        if variable_mask is not None:
            available = variable_mask[:, self.epsilon_index].to(device=device, dtype=torch.bool)
        return available

    def prepare_input(
        self,
        x,
        variable_mask=None,
        force_mask_epsilon=None,
        epsilon_mask_prob=None,
    ):
        if self.epsilon_index < 0 or self.epsilon_index >= x.shape[1]:
            raise ValueError(
                f"epsilon_index={self.epsilon_index} is outside input with {x.shape[1]} channels"
            )

        x_in = x.clone()
        batch, _, h, w = x_in.shape
        device = x_in.device
        available = self._epsilon_available(x_in, variable_mask=variable_mask)

        if force_mask_epsilon is None:
            if self.training:
                prob = self.epsilon_mask_prob if epsilon_mask_prob is None else epsilon_mask_prob
                random_mask = torch.rand(batch, device=device) < prob
                mask_epsilon = random_mask | ~available
            else:
                mask_epsilon = ~available
        elif isinstance(force_mask_epsilon, bool):
            mask_epsilon = torch.full((batch,), force_mask_epsilon, dtype=torch.bool, device=device)
            mask_epsilon = mask_epsilon | ~available
        else:
            mask_epsilon = force_mask_epsilon.to(device=device, dtype=torch.bool) | ~available

        x_in[mask_epsilon, self.epsilon_index] = 0.0

        epsilon_visible = (~mask_epsilon).float().view(batch, 1, 1, 1).expand(batch, 1, h, w)
        if self.use_epsilon_mask_channel:
            x_in = torch.cat([x_in, epsilon_visible], dim=1)

        return x_in, mask_epsilon

    def encode(
        self,
        x,
        variable_mask=None,
        force_mask_epsilon=None,
        epsilon_mask_prob=None,
        return_attention=False,
    ):
        x_in, epsilon_masked = self.prepare_input(
            x,
            variable_mask=variable_mask,
            force_mask_epsilon=force_mask_epsilon,
            epsilon_mask_prob=epsilon_mask_prob,
        )
        token_mask_prob = self.mask_prob if self.training else 0.0
        feature, attn = self.encoder(
            x_in,
            mask_prob=token_mask_prob,
            return_attention=return_attention,
        )
        return feature, attn, epsilon_masked

    def forward(
        self,
        x,
        variable_mask=None,
        force_mask_epsilon=None,
        epsilon_mask_prob=None,
        return_attention=False,
    ):
        feature, attn, epsilon_masked = self.encode(
            x,
            variable_mask=variable_mask,
            force_mask_epsilon=force_mask_epsilon,
            epsilon_mask_prob=epsilon_mask_prob,
            return_attention=return_attention,
        )
        mu, log_var = self.regression(feature)
        return {
            "mu": mu,
            "log_var": log_var,
            "feature": feature,
            "attention": attn,
            "epsilon_masked": epsilon_masked,
        }

    def forward_regression(
        self,
        x,
        variable_mask=None,
        force_mask_epsilon=None,
        epsilon_mask_prob=None,
        return_attention=False,
        return_dict=False,
    ):
        out = self.forward(
            x,
            variable_mask=variable_mask,
            force_mask_epsilon=force_mask_epsilon,
            epsilon_mask_prob=epsilon_mask_prob,
            return_attention=return_attention,
        )
        if return_dict:
            return out
        return out["mu"], out["log_var"], out["feature"], out["attention"]

    def epsilon_loss(
        self,
        epsilon_hat,
        epsilon_target,
        epsilon_masked=None,
        gradient_weight=0.2,
        masked_only=True,
    ):
        """Legacy deterministic epsilon loss kept for experiments with EpsilonFieldHead."""
        if epsilon_target.ndim == 3:
            epsilon_target = epsilon_target.unsqueeze(1)

        if masked_only and epsilon_masked is not None:
            if not epsilon_masked.any():
                return torch.zeros((), device=epsilon_hat.device)
            epsilon_hat = epsilon_hat[epsilon_masked]
            epsilon_target = epsilon_target[epsilon_masked]

        value_loss = F.l1_loss(epsilon_hat, epsilon_target)
        grad_loss = spatial_gradient_l1(epsilon_hat, epsilon_target)
        return value_loss + gradient_weight * grad_loss

    def forward_epsilon_flow(self, epsilon, feature, condition):
        """Conditional flow matching branch for generating epsilon from noise."""
        batch = epsilon.shape[0]
        noise = torch.randn_like(epsilon)
        flow_time = torch.rand(batch, device=epsilon.device)

        t = flow_time[:, None, None, None]
        noisy_epsilon = (1.0 - t) * noise + t * epsilon
        target_velocity = epsilon - noise
        pred_velocity = self.epsilon_flow_unet(noisy_epsilon, feature, condition, flow_time)
        epsilon_flow_hat = noisy_epsilon + (1.0 - t) * pred_velocity
        return pred_velocity, target_velocity, epsilon_flow_hat

    @torch.no_grad()
    def generate_epsilon_from_feature(
        self,
        feature,
        condition,
        steps=32,
        noise=None,
        clamp=None,
        return_trajectory=False,
    ):
        """
        Generate epsilon from an already-computed encoder feature.

        Args:
            feature: encoder latent feature, shape batch x 64 x height x width.
            condition: case condition tensor, shape batch x cond_dim.
            steps: Euler integration steps from t=0 to t=1.
            noise: optional initial noise, shape batch x 1 x height x width.
            clamp: optional (min, max) tuple applied after each step.
            return_trajectory: if True, also return intermediate generated fields.
        """
        if steps <= 0:
            raise ValueError("steps must be positive")

        batch, _, height, width = feature.shape
        if noise is None:
            z = torch.randn(batch, 1, height, width, device=feature.device, dtype=feature.dtype)
        else:
            z = noise.to(device=feature.device, dtype=feature.dtype)

        dt = 1.0 / steps
        trajectory = [z.clone()] if return_trajectory else None
        for step in range(steps):
            flow_time = torch.full(
                (batch,),
                (step + 0.5) / steps,
                device=feature.device,
                dtype=feature.dtype,
            )
            velocity = self.epsilon_flow_unet(z, feature, condition, flow_time)
            z = z + dt * velocity
            if clamp is not None:
                z = z.clamp(min=clamp[0], max=clamp[1])
            if return_trajectory:
                trajectory.append(z.clone())

        if return_trajectory:
            return z, trajectory
        return z

    @torch.no_grad()
    def sample_epsilon(self, feature, condition, **kwargs):
        """Backward-compatible alias for generate_epsilon_from_feature."""
        return self.generate_epsilon_from_feature(feature, condition, **kwargs)

    @torch.no_grad()
    def generate_epsilon(
        self,
        x,
        condition,
        variable_mask=None,
        steps=32,
        noise=None,
        clamp=None,
        return_attention=False,
    ):
        """
        Convenience wrapper for the full no-epsilon inference path:
            x -> masked-epsilon SV-ViT feature -> epsilon flow sampling.

        The flow head itself never receives raw x. It receives only the encoder
        feature produced from x, plus condition, noise, and flow time.
        """
        was_training = self.training
        self.eval()
        feature, attn, epsilon_masked = self.encode(
            x,
            variable_mask=variable_mask,
            force_mask_epsilon=True,
            return_attention=return_attention,
        )
        epsilon_sample = self.generate_epsilon_from_feature(
            feature,
            condition,
            steps=steps,
            noise=noise,
            clamp=clamp,
        )
        if was_training:
            self.train()
        return {
            "epsilon_sample": epsilon_sample,
            "feature": feature,
            "attention": attn,
            "epsilon_masked": epsilon_masked,
        }

    def forward_flow(self, theta, feature, condition):
        batch = theta.shape[0]
        noise = torch.randn_like(theta)
        flow_time = torch.rand(batch, device=theta.device)

        t = flow_time[:, None, None, None]
        noisy_theta = (1.0 - t) * noise + t * theta
        target_velocity = theta - noise
        pred_velocity = self.flow_unet(noisy_theta, feature, condition, flow_time)
        theta_hat = noisy_theta + (1.0 - t) * pred_velocity
        return pred_velocity, target_velocity, theta_hat
