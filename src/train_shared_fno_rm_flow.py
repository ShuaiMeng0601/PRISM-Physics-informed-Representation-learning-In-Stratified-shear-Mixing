import argparse
import csv
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import KHHolmboeDataset


def first_existing_path(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_H5 = first_existing_path([
    REPO_ROOT / "data" / "kh_holmboe_dataset_keep_epsilon.h5",
    REPO_ROOT / "kh_holmboe_dataset_keep_epsilon.h5",
    REPO_ROOT / "experiment" / "kh_holmboe_dataset_keep_epsilon.h5",
])
DEFAULT_VARIABLES = "buoyancy,reduced_shear,log_epsilon"


def parse_pair(value):
    if isinstance(value, tuple):
        return value
    parts = [int(item.strip()) for item in str(value).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers, e.g. 10,10")
    return tuple(parts)


def default_label_csv(label_csv):
    if label_csv is not None:
        return label_csv
    for candidate in [
        Path(__file__).with_name("RM_summary_table.csv"),
        REPO_ROOT / "data" / "RM_summary_table.csv",
        REPO_ROOT / "RM_summary_table.csv",
        REPO_ROOT / "experiment" / "RM_summary_table.csv",
        Path(__file__).with_name("test_RM_summary_table.csv"),
        REPO_ROOT / "data" / "test_RM_summary_table.csv",
        REPO_ROOT / "test_data" / "test_RM_summary_table.csv",
        REPO_ROOT / "experiment" / "test_data" / "test_RM_summary_table.csv",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def collate_training_batch(items):
    batch = {}
    tensor_keys = ["x", "theta", "ratio", "has_ratio", "variable_mask"]
    for key in tensor_keys:
        batch[key] = torch.stack([item[key] for item in items])
    batch["metadata"] = [item["metadata"] for item in items]
    return batch


def make_loader(h5_path, split, label_csv, batch_size, shuffle, input_variables, num_workers=0):
    dataset = KHHolmboeDataset(
        h5_path,
        split=split,
        label_csv=label_csv,
        input_variables=input_variables,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_training_batch,
    )


def make_group_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class PatchEmbed2D(nn.Module):
    """Old-style Conv2d patch embedding for one variable field."""

    def __init__(self, img_size=(491, 200), patch_size=(10, 10), dim=64):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)

        height, width = self.img_size
        patch_y, patch_t = self.patch_size
        self.pad_y = math.ceil(height / patch_y) * patch_y - height
        self.pad_t = math.ceil(width / patch_t) * patch_t - width
        self.grid_y = (height + self.pad_y) // patch_y
        self.grid_t = (width + self.pad_t) // patch_t
        self.n_patches = self.grid_y * self.grid_t

        self.proj = nn.Conv2d(1, dim, kernel_size=self.patch_size, stride=self.patch_size)

    @property
    def grid_size(self):
        return self.grid_y, self.grid_t

    def forward(self, x):
        x = F.pad(x, (0, self.pad_t, 0, self.pad_y))
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VariableViT(nn.Module):
    """Variable-specific ViT branch, matching the old encoder style."""

    def __init__(
        self,
        img_size=(491, 200),
        patch_size=(10, 10),
        dim=64,
        depth=2,
        heads=4,
        dropout=0.0,
    ):
        super().__init__()
        self.patch = PatchEmbed2D(img_size=img_size, patch_size=patch_size, dim=dim)
        grid_y, grid_t = self.patch.grid_size
        self.pos_2d = nn.Parameter(torch.zeros(1, grid_y, grid_t, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    @property
    def grid_size(self):
        return self.patch.grid_size

    def forward(self, x):
        tokens = self.patch(x)
        pos = self.pos_2d.reshape(1, self.patch.n_patches, -1)
        return self.encoder(tokens + pos)


class CrossVariableAttention(nn.Module):
    """Old-style attention over all variable patch tokens, with missing-variable masking."""

    def __init__(self, dim=64, heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens, var_mask=None, return_attention=False):
        batch, n_vars, n_patches, dim = tokens.shape
        flat = tokens.reshape(batch, n_vars * n_patches, dim)
        key_padding_mask = None
        if var_mask is not None:
            key_padding_mask = ~var_mask[:, :, None].expand(
                batch,
                n_vars,
                n_patches,
            ).reshape(batch, n_vars * n_patches)

        normed = self.norm1(flat)
        attn_out, attn = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        flat = flat + attn_out
        flat = flat + self.mlp(self.norm2(flat))

        out = flat.reshape(batch, n_vars, n_patches, dim)
        if var_mask is not None:
            out = out * var_mask[:, :, None, None].to(dtype=out.dtype)

        if return_attention:
            return out, attn.detach()
        return out, None


class SpectralMixer2D(nn.Module):
    """FNO-style spectral mixing on the patch grid, before channel fusion."""

    def __init__(self, dim=64, modes_y=12, modes_t=8):
        super().__init__()
        self.dim = dim
        self.modes_y = modes_y
        self.modes_t = modes_t
        scale = 1.0 / math.sqrt(dim)
        self.weight_pos = nn.Parameter(scale * torch.randn(dim, dim, modes_y, modes_t, 2))
        self.weight_neg = nn.Parameter(scale * torch.randn(dim, dim, modes_y, modes_t, 2))
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def _complex_mul(self, x_ft, weight):
        return torch.einsum("bihw,iohw->bohw", x_ft, weight)

    def forward(self, tokens, grid_size):
        batch, n_vars, n_patches, dim = tokens.shape
        grid_y, grid_t = grid_size
        x = self.norm(tokens)
        x = x.reshape(batch, n_vars, grid_y, grid_t, dim)
        x = x.permute(0, 1, 4, 2, 3).reshape(batch * n_vars, dim, grid_y, grid_t)
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batch * n_vars,
            dim,
            grid_y,
            grid_t // 2 + 1,
            device=x.device,
            dtype=x_ft.dtype,
        )
        modes_y = min(self.modes_y, grid_y // 2 + 1)
        modes_t = min(self.modes_t, grid_t // 2 + 1)
        weight_pos = torch.view_as_complex(
            self.weight_pos[:, :, :modes_y, :modes_t].contiguous()
        )
        weight_neg = torch.view_as_complex(
            self.weight_neg[:, :, :modes_y, :modes_t].contiguous()
        )
        out_ft[:, :, :modes_y, :modes_t] = self._complex_mul(
            x_ft[:, :, :modes_y, :modes_t],
            weight_pos,
        )
        out_ft[:, :, -modes_y:, :modes_t] = self._complex_mul(
            x_ft[:, :, -modes_y:, :modes_t],
            weight_neg,
        )

        mixed = torch.fft.irfft2(out_ft, s=(grid_y, grid_t), norm="ortho")
        mixed = mixed + self.pointwise(x)
        mixed = mixed.reshape(batch, n_vars, dim, grid_y, grid_t)
        mixed = mixed.permute(0, 1, 3, 4, 2).reshape(batch, n_vars, n_patches, dim)
        return tokens + mixed


class SharedFNOEncoder(nn.Module):
    """
    Old-style encoder with an FNO mixer and a smaller latent output.

    Output:
        feature: batch x latent_channels x patch_y x patch_t
    """

    def __init__(
        self,
        img_size=(491, 200),
        patch_size=(10, 10),
        n_vars=3,
        dim=64,
        vit_depth=2,
        heads=4,
        latent_channels=64,
        fno_modes=(12, 8),
        dropout=0.0,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.dim = dim
        self.latent_channels = latent_channels
        self.variable_vits = nn.ModuleList([
            VariableViT(
                img_size=img_size,
                patch_size=patch_size,
                dim=dim,
                depth=vit_depth,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(n_vars)
        ])
        self.grid_y, self.grid_t = self.variable_vits[0].grid_size
        self.n_patches = self.grid_y * self.grid_t
        self.var_embed = nn.Parameter(torch.zeros(1, n_vars, 1, dim))
        self.availability_embed = nn.Embedding(2, dim)
        self.cross = CrossVariableAttention(dim=dim, heads=heads, dropout=dropout)
        self.fno = SpectralMixer2D(dim=dim, modes_y=fno_modes[0], modes_t=fno_modes[1])
        self.to_feature = nn.Sequential(
            nn.Conv2d(n_vars * dim, latent_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    @property
    def grid_size(self):
        return self.grid_y, self.grid_t

    def forward(self, x, var_mask=None, return_attention=False):
        batch, n_vars, _, _ = x.shape
        if n_vars != self.n_vars:
            raise ValueError(f"expected {self.n_vars} variables, got {n_vars}")

        if var_mask is None:
            var_mask = torch.ones(batch, n_vars, dtype=torch.bool, device=x.device)
        else:
            var_mask = var_mask.to(device=x.device, dtype=torch.bool)

        all_tokens = []
        for var_id, vit in enumerate(self.variable_vits):
            tokens = vit(x[:, var_id:var_id + 1])
            all_tokens.append(tokens)
        tokens = torch.stack(all_tokens, dim=1)

        status = self.availability_embed(var_mask.long()).view(batch, n_vars, 1, self.dim)
        tokens = tokens + self.var_embed + status
        tokens = tokens * var_mask[:, :, None, None].to(dtype=tokens.dtype)

        tokens, attn = self.cross(tokens, var_mask=var_mask, return_attention=return_attention)
        tokens = self.fno(tokens, grid_size=self.grid_size)
        tokens = tokens * var_mask[:, :, None, None].to(dtype=tokens.dtype)

        grid = tokens.reshape(batch, n_vars, self.grid_y, self.grid_t, self.dim)
        grid = grid.permute(0, 1, 4, 2, 3).reshape(
            batch,
            n_vars * self.dim,
            self.grid_y,
            self.grid_t,
        )
        feature = self.to_feature(grid)
        return feature, {"cross_attention": attn} if return_attention else None


class RMHead(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            make_group_norm(hidden_channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, feature):
        return self.net(feature)


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            make_group_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class LowResLatentUNet(nn.Module):
    """Flow velocity decoder. It only uses noisy buoyancy, flow time, and compact latent."""

    def __init__(
        self,
        latent_channels=64,
        latent_map_channels=64,
        base_channels=64,
        channel_mults=(1, 2, 4),
    ):
        super().__init__()
        self.latent_project = nn.Sequential(
            nn.Conv2d(latent_channels, latent_map_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_map_channels, latent_map_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        total_in = 1 + latent_map_channels + 1
        channels = [base_channels * mult for mult in channel_mults]
        self.input = ConvGNAct(total_in, channels[0])
        self.down_blocks = nn.ModuleList()
        for idx in range(1, len(channels)):
            self.down_blocks.append(ConvGNAct(channels[idx - 1], channels[idx]))
        self.mid = ConvGNAct(channels[-1], channels[-1])

        self.up_blocks = nn.ModuleList()
        current = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.up_blocks.append(ConvGNAct(current + skip_channels, skip_channels))
            current = skip_channels
        self.out = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x_tau, tau, feature):
        batch, _, height, width = x_tau.shape
        latent_map = self.latent_project(feature)
        latent_map = F.interpolate(latent_map, size=(height, width), mode="bilinear", align_corners=False)
        time_map = tau[:, None, None, None].expand(batch, 1, height, width)
        x = torch.cat([x_tau, latent_map, time_map], dim=1)

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

        if x.shape[-2:] != (height, width):
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return self.out(x)


class SharedFNOFlowModel(nn.Module):
    def __init__(
        self,
        img_size=(491, 200),
        patch_size=(10, 10),
        n_vars=3,
        dim=64,
        vit_depth=2,
        heads=4,
        latent_channels=64,
        fno_modes=(12, 8),
        dropout=0.0,
        rm_dropout=0.1,
        flow_base_channels=64,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.encoder = SharedFNOEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_vars=n_vars,
            dim=dim,
            vit_depth=vit_depth,
            heads=heads,
            latent_channels=latent_channels,
            fno_modes=fno_modes,
            dropout=dropout,
        )
        self.rm_head = RMHead(in_channels=latent_channels, dropout=rm_dropout)
        self.flow_decoder = LowResLatentUNet(
            latent_channels=latent_channels,
            base_channels=flow_base_channels,
        )

    @property
    def latent_shape(self):
        return self.encoder.latent_channels, self.encoder.grid_y, self.encoder.grid_t

    def encode(self, x, var_mask=None, return_attention=False):
        return self.encoder(x, var_mask=var_mask, return_attention=return_attention)

    def predict_rm(self, feature):
        return self.rm_head(feature)

    def velocity(self, x_tau, tau, feature):
        return self.flow_decoder(x_tau, tau, feature)

    def forward(self, x, var_mask=None, return_attention=False):
        feature, attention = self.encode(x, var_mask=var_mask, return_attention=return_attention)
        rm = self.predict_rm(feature)
        return {"feature": feature, "rm": rm, "attention": attention}

    def forward_flow(self, buoyancy, feature):
        batch = buoyancy.shape[0]
        noise = torch.randn_like(buoyancy)
        tau = torch.rand(batch, device=buoyancy.device, dtype=buoyancy.dtype)
        tau_view = tau[:, None, None, None]
        x_tau = (1.0 - tau_view) * noise + tau_view * buoyancy
        target_velocity = buoyancy - noise
        pred_velocity = self.velocity(x_tau, tau, feature)
        buoyancy_hat = x_tau + (1.0 - tau_view) * pred_velocity
        return pred_velocity, target_velocity, buoyancy_hat

    @torch.no_grad()
    def reconstruct(self, feature, steps=50, noise=None, clamp=None):
        if steps <= 0:
            raise ValueError("steps must be positive")
        batch = feature.shape[0]
        height, width = self.img_size
        if noise is None:
            field = torch.randn(batch, 1, height, width, device=feature.device, dtype=feature.dtype)
        else:
            field = noise.to(device=feature.device, dtype=feature.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            tau = torch.full(
                (batch,),
                (step + 0.5) / steps,
                device=feature.device,
                dtype=feature.dtype,
            )
            field = field + dt * self.velocity(field, tau, feature)
            if clamp is not None:
                field = field.clamp(min=clamp[0], max=clamp[1])
        return field


def random_spatial_patch_mask(x, patch_size=(10, 10), mask_prob=0.25):
    batch, _, height, width = x.shape
    if mask_prob <= 0:
        return torch.zeros(batch, 1, height, width, dtype=torch.bool, device=x.device)
    patch_y, patch_t = patch_size
    grid_y = math.ceil(height / patch_y)
    grid_t = math.ceil(width / patch_t)
    patch_mask = torch.rand(batch, 1, grid_y, grid_t, device=x.device) < mask_prob
    return F.interpolate(patch_mask.float(), size=(height, width), mode="nearest").bool()


def random_variable_dropout(variable_mask, drop_prob=0.25):
    if drop_prob <= 0:
        return variable_mask
    available = variable_mask.to(dtype=torch.bool)
    drop = (torch.rand(available.shape, device=available.device) < drop_prob) & available
    kept = available & ~drop

    empty_rows = torch.where(~kept.any(dim=1))[0]
    for row in empty_rows.tolist():
        candidates = torch.where(available[row])[0]
        if len(candidates) > 0:
            kept[row, candidates[0]] = True
    return kept


def make_self_supervised_input(
    x,
    variable_mask,
    patch_size=(10, 10),
    spatial_mask_prob=0.25,
    variable_drop_prob=0.25,
):
    if variable_mask is None:
        variable_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
    else:
        variable_mask = variable_mask.to(device=x.device, dtype=torch.bool)

    input_var_mask = random_variable_dropout(variable_mask, drop_prob=variable_drop_prob)
    spatial_mask = random_spatial_patch_mask(x, patch_size=patch_size, mask_prob=spatial_mask_prob)
    x_in = x.clone()
    x_in = x_in * input_var_mask[:, :, None, None].to(dtype=x.dtype)
    x_in = x_in.masked_fill(spatial_mask, 0.0)
    return x_in, input_var_mask, spatial_mask


def mse_with_optional_mask(pred, target, mask=None, masked_only=False):
    if mask is None or not masked_only or not bool(mask.any()):
        return F.mse_loss(pred, target)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    sq_error = (pred - target) ** 2
    denom = mask.sum().clamp_min(1.0) * pred.shape[1]
    return (sq_error * mask).sum() / denom


def compute_rm_loss(rm_pred, rm_true, has_rm):
    if has_rm.any():
        return F.mse_loss(rm_pred[has_rm], rm_true[has_rm])
    return torch.zeros((), device=rm_pred.device, dtype=rm_pred.dtype)


def compute_rm_saliency(model, x, var_mask=None):
    """
    RM saliency on the compact latent feature.

    Returns:
        saliency: batch x patch_y x patch_t
        rm_pred: batch x 1
    """
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    feature, _ = model.encode(x, var_mask=var_mask, return_attention=False)
    feature.retain_grad()
    rm_pred = model.predict_rm(feature)
    rm_pred.sum().backward()

    saliency = (feature.grad * feature).abs().mean(dim=1).detach()
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return saliency, rm_pred.detach()


def run_one_epoch(model, loader, optimizer, device, args):
    is_train = optimizer is not None
    model.train(is_train)
    sums = {
        "total": 0.0,
        "rm": 0.0,
        "flow": 0.0,
        "buoyancy_mse": 0.0,
        "rm_mae": 0.0,
        "spatial_mask_frac": 0.0,
        "var_visible_frac": 0.0,
    }
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        buoyancy = batch["theta"].to(device)
        rm_true = batch["ratio"].to(device)
        has_rm = batch["has_ratio"].to(device)
        variable_mask = batch["variable_mask"].to(device)

        x_in, input_var_mask, spatial_mask = make_self_supervised_input(
            x,
            variable_mask,
            patch_size=args.patch_size,
            spatial_mask_prob=args.spatial_mask_prob,
            variable_drop_prob=args.variable_drop_prob,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        out = model(x_in, var_mask=input_var_mask, return_attention=False)
        feature = out["feature"]
        rm_pred = out["rm"]
        rm_loss = compute_rm_loss(rm_pred, rm_true, has_rm)

        pred_velocity, target_velocity, buoyancy_hat = model.forward_flow(buoyancy, feature)
        flow_loss = mse_with_optional_mask(
            pred_velocity,
            target_velocity,
            mask=spatial_mask,
            masked_only=args.flow_masked_only,
        )
        buoyancy_mse = mse_with_optional_mask(
            buoyancy_hat,
            buoyancy,
            mask=spatial_mask,
            masked_only=args.flow_masked_only,
        )
        total_loss = rm_loss + args.lambda_flow * flow_loss

        if is_train:
            total_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if has_rm.any():
            rm_mae = torch.abs(rm_pred[has_rm] - rm_true[has_rm]).mean()
        else:
            rm_mae = torch.zeros((), device=device)

        sums["total"] += float(total_loss.detach())
        sums["rm"] += float(rm_loss.detach())
        sums["flow"] += float(flow_loss.detach())
        sums["buoyancy_mse"] += float(buoyancy_mse.detach())
        sums["rm_mae"] += float(rm_mae.detach())
        sums["spatial_mask_frac"] += float(spatial_mask.float().mean().detach())
        sums["var_visible_frac"] += float(input_var_mask.float().mean().detach())
        n_batches += 1

    return {name: value / max(n_batches, 1) for name, value in sums.items()}


def init_metrics_csv(path):
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_total",
        "train_rm",
        "train_flow",
        "train_buoyancy_mse",
        "train_rm_mae",
        "train_spatial_mask_frac",
        "train_var_visible_frac",
        "val_total",
        "val_rm",
        "val_flow",
        "val_buoyancy_mse",
        "val_rm_mae",
        "val_spatial_mask_frac",
        "val_var_visible_frac",
        "best_val_rm",
        "checkpoint_saved",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return path, fieldnames


def append_metrics_row(path, fieldnames, row):
    with Path(path).open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def plot_loss_curves(metrics_csv, output_path):
    if output_path is None or str(output_path).lower() == "none":
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping loss plot.", flush=True)
        return

    rows = []
    with Path(metrics_csv).open("r", newline="") as f:
        rows.extend(csv.DictReader(f))
    if not rows:
        return

    output_path = Path(output_path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]
    terms = ["total", "rm", "flow", "buoyancy_mse", "rm_mae"]
    fig, axes = plt.subplots(len(terms), 1, figsize=(8, 14), sharex=True)
    for ax, term in zip(axes, terms):
        ax.plot(epochs, [float(row[f"train_{term}"]) for row in rows], label="train")
        ax.plot(epochs, [float(row[f"val_{term}"]) for row in rows], label="val")
        ax.set_ylabel(term)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--input_variables", default=DEFAULT_VARIABLES)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--spatial_mask_prob", type=float, default=0.25)
    parser.add_argument("--variable_drop_prob", type=float, default=0.25)
    parser.add_argument("--flow_masked_only", action="store_true")
    parser.add_argument("--flow_all_pixels", dest="flow_masked_only", action="store_false")
    parser.set_defaults(flow_masked_only=True)
    parser.add_argument("--patch_size", type=parse_pair, default=(10, 10))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--vit_depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--fno_modes", type=parse_pair, default=(12, 8))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rm_dropout", type=float, default=0.1)
    parser.add_argument("--flow_base_channels", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", default="shared_fno_rm_flow_model.pt")
    parser.add_argument("--metrics_csv", default="shared_fno_rm_flow_loss_history.csv")
    parser.add_argument("--loss_plot", default="shared_fno_rm_flow_loss_curves.png")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, args, input_variables, img_size, best_val_rm, epoch):
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "input_variables": input_variables,
            "img_size": img_size,
            "latent_shape": model.latent_shape,
            "best_val_rm": best_val_rm,
            "epoch": epoch,
        },
        path,
    )


def main():
    args = parse_args()
    args.label_csv = default_label_csv(args.label_csv)
    set_seed(args.seed)
    device = torch.device(args.device)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    metrics_csv, metric_fields = init_metrics_csv(args.metrics_csv)

    train_loader = make_loader(
        args.h5,
        "train",
        args.label_csv,
        args.batch_size,
        shuffle=True,
        input_variables=args.input_variables,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        args.h5,
        "val",
        args.label_csv,
        args.batch_size,
        shuffle=False,
        input_variables=args.input_variables,
        num_workers=args.num_workers,
    )

    input_variables = train_loader.dataset.input_variables
    img_size = tuple(train_loader.dataset.x_data.shape[-2:])
    print(f"Using H5: {args.h5}", flush=True)
    print(f"Using label CSV: {args.label_csv}", flush=True)
    print(f"Using input variables: {input_variables}", flush=True)
    print(f"Using image size: {img_size}", flush=True)

    model = SharedFNOFlowModel(
        img_size=img_size,
        patch_size=args.patch_size,
        n_vars=len(input_variables),
        dim=args.dim,
        vit_depth=args.vit_depth,
        heads=args.heads,
        latent_channels=args.latent_channels,
        fno_modes=args.fno_modes,
        dropout=args.dropout,
        rm_dropout=args.rm_dropout,
        flow_base_channels=args.flow_base_channels,
    ).to(device)
    print(f"Compact latent feature per sample: {model.latent_shape}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rm = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, args)
        with torch.no_grad():
            val_metrics = run_one_epoch(model, val_loader, None, device, args)

        checkpoint_saved = False
        if val_metrics["rm"] < best_val_rm:
            best_val_rm = val_metrics["rm"]
            save_checkpoint(
                args.save,
                model,
                args,
                input_variables,
                img_size,
                best_val_rm,
                epoch,
            )
            checkpoint_saved = True

        print(
            f"epoch {epoch:03d} | "
            f"train total {train_metrics['total']:.6f} "
            f"rm {train_metrics['rm']:.6f} "
            f"flow {train_metrics['flow']:.6f} "
            f"buoy_mse {train_metrics['buoyancy_mse']:.6f} "
            f"rm_mae {train_metrics['rm_mae']:.6f} | "
            f"val total {val_metrics['total']:.6f} "
            f"rm {val_metrics['rm']:.6f} "
            f"flow {val_metrics['flow']:.6f} "
            f"buoy_mse {val_metrics['buoyancy_mse']:.6f} "
            f"rm_mae {val_metrics['rm_mae']:.6f}",
            flush=True,
        )
        if checkpoint_saved:
            print(f"saved {args.save}", flush=True)

        row = {
            "epoch": epoch,
            "best_val_rm": best_val_rm,
            "checkpoint_saved": int(checkpoint_saved),
        }
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            for name, value in metrics.items():
                row[f"{split}_{name}"] = value
        append_metrics_row(metrics_csv, metric_fields, row)
        plot_loss_curves(metrics_csv, args.loss_plot)

    train_loader.dataset.close()
    val_loader.dataset.close()


if __name__ == "__main__":
    main()
