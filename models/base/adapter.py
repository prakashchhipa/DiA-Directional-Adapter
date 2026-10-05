#!/usr/bin/env python3
# -----------------------------------------------
# Written by Qizhong Tan
# -----------------------------------------------

import torch
import clip
import math
import utils.logging as logging
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from collections import OrderedDict
import torchvision

from utils.registry import Registry
from models.base.fine_discriminative_adapter import ViT_FineDiscriminativeAdapter

HEAD_REGISTRY = Registry("Head")

logger = logging.get_logger(__name__)


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.norm(x)
        x = rearrange(x, 'b t h w c -> b c t h w')
        return x


def OTAM_dist(dists, lbda=0.5):
    dists = F.pad(dists, (1, 1), 'constant', 0)
    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(- cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(- cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


class ResNet_DeformAttention(nn.Module):
    def __init__(self, dim, heads, groups, kernel_size, stride, padding):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels ** -0.5
        self.groups = groups
        self.group_channels = self.dim // self.groups
        self.group_heads = self.heads // self.groups
        self.factor = 2.0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(in_channels=self.group_channels, out_channels=self.group_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.group_channels),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(in_channels=self.group_channels, out_channels=3, kernel_size=(1, 1, 1), bias=False)
        )

        self.proj_q = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))
        self.proj_k = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))
        self.proj_v = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))
        self.proj_out = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))

    @torch.no_grad()
    def _get_ref_points(self, T, H, W, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 0].div_(T).mul_(2).sub_(1)
        ref[..., 1].div_(H).mul_(2).sub_(1)
        ref[..., 2].div_(W).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)  # B * g T H W 3

        return ref

    def forward(self, x):
        B, C, T, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = rearrange(q, 'b (g c) t h w -> (b g) c t h w', g=self.groups, c=self.group_channels)
        offset = self.conv_offset(q_off)  # B * g 3 Tp Hp Wp
        Tp, Hp, Wp = offset.size(2), offset.size(3), offset.size(4)
        n_sample = Tp * Hp * Wp
        # logger.info('{}x{}x{}={}'.format(Tp, Hp, Wp, n_sample))

        offset_range = torch.tensor([min(1.0, self.factor / Tp), min(1.0, self.factor / Hp), min(1.0, self.factor / Wp)], device=device).reshape(1, 3, 1, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = rearrange(offset, 'b p t h w -> b t h w p')
        reference = self._get_ref_points(Tp, Hp, Wp, B, dtype, device)
        pos = offset + reference

        x_sampled = F.grid_sample(input=x.reshape(B * self.groups, self.group_channels, T, H, W),
                                  grid=pos[..., (2, 1, 0)],  # z, y, x -> x, y, z
                                  mode='bilinear', align_corners=True)  # B * g, Cg, Tp, Hp, Wp

        x_sampled = x_sampled.reshape(B, C, 1, 1, n_sample)
        q = q.reshape(B * self.heads, self.head_channels, T * H * W)
        k = self.proj_k(x_sampled).reshape(B * self.heads, self.head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.heads, self.head_channels, n_sample)

        attn = einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)

        attn = F.softmax(attn, dim=-1)

        out = einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, T, H, W)
        out = self.proj_out(out)

        return out


class ResNet_Vanilla_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        x = self.relu(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


class ResNet_ST_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))
        self.conv = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=self.adapter_channels)
        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        x = self.conv(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


class ResNet_DST_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))

        self.s_conv = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=self.adapter_channels, bias=False)
        self.s_bn = nn.BatchNorm3d(num_features=self.adapter_channels)

        self.t_conv = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=self.adapter_channels, bias=False)
        self.t_bn = nn.BatchNorm3d(num_features=self.adapter_channels)

        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        # Spatial Pathway
        xs = self.s_bn(self.s_conv(x))

        # Temporal Pathway
        xt = self.t_bn(self.t_conv(x))

        x = (xs + xt) / 2
        x = self.relu(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


class ResNet_D2ST_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)
        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))

        self.pos_embed = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=self.adapter_channels)
        self.s_ln = LayerNormProxy(dim=self.adapter_channels)
        self.t_ln = LayerNormProxy(dim=self.adapter_channels)
        if dim == self.args.ADAPTER.WIDTH // 8:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=1, groups=1, kernel_size=(4, 7, 7), stride=(4, 7, 7), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=1, groups=1, kernel_size=(1, 14, 14), stride=(1, 14, 14), padding=(0, 0, 0))
        elif dim == self.args.ADAPTER.WIDTH // 4:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=2, groups=2, kernel_size=(4, 7, 7), stride=(4, 7, 7), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=2, groups=2, kernel_size=(1, 14, 14), stride=(1, 14, 14), padding=(0, 0, 0))
        elif dim == self.args.ADAPTER.WIDTH // 2:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=4, groups=4, kernel_size=(4, 5, 5), stride=(4, 3, 3), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=4, groups=4, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        else:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=8, groups=8, kernel_size=(4, 4, 4), stride=(4, 3, 3), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=8, groups=8, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        self.gelu = nn.GELU()

        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        x = x + self.pos_embed(x)

        # Spatial Deformable Attention
        xs = x + self.s_attn(self.s_ln(x))

        # Temporal Deformable Attention
        xt = x + self.t_attn(self.t_ln(x))

        x = (xs + xt) / 2
        x = self.gelu(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


@HEAD_REGISTRY.register()
class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.args = cfg
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        if self.args.ADAPTER.LAYERS == 18:
            backbone = torchvision.models.resnet18(pretrained=True)
        elif self.args.ADAPTER.LAYERS == 34:
            backbone = torchvision.models.resnet34(pretrained=True)
        elif self.args.ADAPTER.LAYERS == 50:
            backbone = torchvision.models.resnet50(pretrained=True)
        self.stage1 = nn.Sequential(*list(backbone.children())[:5])
        self.Adapter_1 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH // 8)
        self.stage2 = nn.Sequential(*list(backbone.children())[5])
        self.Adapter_2 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH // 4)
        self.stage3 = nn.Sequential(*list(backbone.children())[6])
        self.Adapter_3 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH // 2)
        self.stage4 = nn.Sequential(*list(backbone.children())[7])
        self.Adapter_4 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH)
        self.stage5 = nn.Sequential(*list(backbone.children())[8:-1])
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            # Get dropout rate from config (default to 0.0 if not specified)
            dropout_rate = getattr(cfg.VIDEO.HEAD, "DROPOUT_RATE", 0.0) if hasattr(cfg, "VIDEO") and hasattr(cfg.VIDEO, "HEAD") else 0.0
            # Create dropout layer if dropout_rate > 0
            if dropout_rate > 0.0:
                self.dropout = nn.Dropout(dropout_rate)
            else:
                self.dropout = None
            self.classification_layer = nn.Linear(self.args.ADAPTER.WIDTH, int(self.args.TRAIN.NUM_CLASS))
        self.init_weights()

    def init_weights(self):
        # zero-initialize Adapters
        for n1, m1 in self.named_modules():
            if 'Adapter' in n1:
                for n2, m2 in m1.named_modules():
                    if 'up' in n2:
                        logger.info('init:  {}.{}'.format(n1, n2))
                        if hasattr(m2, 'weight') and m2.weight is not None:
                            nn.init.constant_(m2.weight, 0)
                        if hasattr(m2, 'bias') and m2.bias is not None:
                            nn.init.constant_(m2.bias, 0)

    def get_feat(self, x):
        x = self.stage1(x)
        x = self.Adapter_1(x)
        x = self.stage2(x)
        x = self.Adapter_2(x)
        x = self.stage3(x)
        x = self.Adapter_3(x)
        x = self.stage4(x)
        x = self.Adapter_4(x)
        x = self.stage5(x)
        return x.squeeze()

    def extract_class_indices(self, labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def forward(self, inputs):
        support_images, query_images = inputs['support_set'], inputs['target_set']
        support_features = self.get_feat(support_images)
        query_features = self.get_feat(query_images)
        support_labels = inputs['support_labels']
        unique_labels = torch.unique(support_labels)

        support_features = support_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)
        query_features = query_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)

        class_logits = None
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            class_logits = self.classification_layer(torch.cat([torch.mean(support_features, dim=1), torch.mean(query_features, dim=1)], 0))

        support_features = [torch.mean(torch.index_select(support_features, 0, self.extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_num = support_features.shape[0]
        query_num = query_features.shape[0]

        support_features = support_features.unsqueeze(0).repeat(query_num, 1, 1, 1)
        support_features = rearrange(support_features, 'q s t c -> q (s t) c')

        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(query_features, dim=2).permute(0, 2, 1)).reshape(query_num, support_num, self.num_frames, self.num_frames)
        dist = 1 - frame_sim

        # Bi-MHM
        class_dist = dist.min(3)[0].sum(2) + dist.min(2)[0].sum(2)

        # OTAM
        # class_dist = OTAM_dist(dist) + OTAM_dist(rearrange(dist, 'q s n m -> q s m n'))

        return_dict = {'logits': - class_dist, 'class_logits': class_logits}
        return return_dict

    def forward_classification(self, video, labels=None):
        """
        Forward pass for standard classification.
        Args:
            video: video tensor of shape [B*T, C, H, W]
            labels: optional labels tensor of shape [B]
        Returns:
            dict with 'logits' key containing classification logits
        """
        # Get features: get_feat expects [B*T, C, H, W] and returns [B*T, width]
        features = self.get_feat(video)  # [B*T, width]
        
        # Reshape to [B, T, width] and average over temporal dimension
        B = features.shape[0] // self.num_frames
        features = features.reshape(B, self.num_frames, self.width)
        features = torch.mean(features, dim=1)  # [B, width]
        
        # Apply dropout if enabled (for regularization)
        if hasattr(self, 'dropout') and self.dropout is not None:
            features = self.dropout(features)
        
        # Get classification logits
        if hasattr(self, 'classification_layer'):
            logits = self.classification_layer(features)  # [B, num_classes]
        else:
            raise ValueError("classification_layer not found. Make sure NUM_CLASS is set in config.")
        
        return {'logits': logits}


class ViT_PlainVanilla_Adapter(nn.Module):
    """
    Plain token-wise bottleneck adapter baseline.
    No temporal or spatial mixing: each token is adapted independently.
    """
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.ADAPTER.WIDTH
        self.out_channels = cfg.ADAPTER.WIDTH
        self.adapter_channels = int(cfg.ADAPTER.WIDTH * cfg.ADAPTER.ADAPTER_SCALE)
        act_name = getattr(cfg.ADAPTER, "ACT", "gelu").lower()

        self.down = nn.Linear(self.in_channels, self.adapter_channels)
        self.act = nn.GELU() if act_name == "gelu" else nn.ReLU(inplace=True)
        self.up = nn.Linear(self.adapter_channels, self.out_channels)

        # Keep baseline behavior close to identity at init.
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x):  # [N, BT, C]
        return x + self.up(self.act(self.down(x)))


class ViT_FineGrained_ST_CausalAntiCausal_Adapter(nn.Module):
    """
    Fine-Grained adapter variant with causal + anti-causal temporal modeling.
    """

    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.in_channels = cfg.ADAPTER.WIDTH
        self.out_channels = cfg.ADAPTER.WIDTH
        self.adapter_channels = int(cfg.ADAPTER.WIDTH * cfg.ADAPTER.ADAPTER_SCALE)
        act_name = getattr(cfg.ADAPTER, "ACT", "gelu").lower()
        k_t = getattr(cfg.ADAPTER, "TEMPORAL_K", 5)

        self.down = nn.Conv2d(self.in_channels, self.adapter_channels, kernel_size=1)

        # Temporal-only directional modeling: past-only and future-only paths.
        self.temporal_causal = nn.Conv1d(
            self.adapter_channels,
            self.adapter_channels,
            kernel_size=k_t,
            padding=0,
            groups=self.adapter_channels,
            bias=False,
        )
        self.temporal_anticausal = nn.Conv1d(
            self.adapter_channels,
            self.adapter_channels,
            kernel_size=k_t,
            padding=0,
            groups=self.adapter_channels,
            bias=False,
        )
        self.temporal_direction_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.act = nn.GELU() if act_name == "gelu" else nn.ReLU(inplace=True)

        self.up = nn.Conv2d(self.adapter_channels, self.out_channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias) if self.up.bias is not None else None

    def _causal_anticausal(self, y_flat):
        """Apply directional temporal convs to y_flat shaped (B*H*W, C, T)."""
        T = y_flat.shape[-1]
        k = self.temporal_causal.kernel_size[0]
        pad = k - 1

        y_causal = F.pad(y_flat, (pad, 0), mode="replicate")
        y_causal = self.temporal_causal(y_causal)[:, :, :T]

        y_anticausal = F.pad(y_flat, (0, pad), mode="replicate")
        y_anticausal = self.temporal_anticausal(y_anticausal)[:, :, -T:]

        w = F.softmax(self.temporal_direction_weight, dim=0)
        return w[0] * y_causal + w[1] * y_anticausal

    def forward(self, x):  # [N, BT, C]
        n, bt, c = x.shape
        T = self.args.DATA.NUM_INPUT_FRAMES
        if T <= 1:
            return x
        B = bt // T
        H = round(math.sqrt(n - 1))
        cls, tokens = x[0:1], x[1:]

        tokens_reshaped = rearrange(
            tokens, "(h w) (b t) c -> b t h w c", h=H, w=H, b=B, t=T
        )
        tokens_3d = rearrange(tokens_reshaped, "b t h w c -> b c t h w")

        y = rearrange(tokens_3d, "b c t h w -> (b t) c h w")
        y = self.down(y)
        y = rearrange(y, "(b t) c h w -> b c t h w", b=B, t=T)

        y_flat = rearrange(y, "b c t h w -> (b h w) c t")
        y_temporal = self._causal_anticausal(y_flat)
        y = rearrange(y_temporal, "(b h w) c t -> b c t h w", b=B, h=H, w=H)
        y = self.act(y)

        y = rearrange(y, "b c t h w -> (b t) c h w")
        y = self.up(y)
        y = rearrange(y, "(b t) c h w -> b c t h w", b=B, t=T)

        y = rearrange(y, "b c t h w -> b t h w c")
        y = rearrange(y, "b t h w c -> (h w) (b t) c")
        tokens = tokens + y
        out = torch.cat([cls, tokens], dim=0)
        return out


def build_vit_adapter(cfg, layer_idx: int, total_layers: int) -> nn.Module:
    apply_last_n = getattr(cfg.ADAPTER, "APPLY_LAST_N", cfg.ADAPTER.LAYERS)
    use_adapter = (layer_idx >= total_layers - apply_last_n)
    if not use_adapter:
        return nn.Identity()
    atype = getattr(cfg.ADAPTER, "TYPE", "finegrained_causal_anticausal").lower()
    if atype in ("plain_vanilla", "vit_plain_vanilla", "plain"):
        return ViT_PlainVanilla_Adapter(cfg)
    if atype in (
        "finegrained_causal_anticausal",
        "finegrained_st_ca",
    ):
        return ViT_FineGrained_ST_CausalAntiCausal_Adapter(cfg)
    raise ValueError(
        f"Unknown ADAPTER.TYPE '{atype}'. Use 'finegrained_causal_anticausal' (DiA) "
        f"or 'plain_vanilla' (token-wise bottleneck baseline)."
    )

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, cfg, layer_idx: int, total_layers: int):
        super().__init__()
        d_model = cfg.ADAPTER.WIDTH
        n_head = cfg.ADAPTER.HEADS
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.Adapter = build_vit_adapter(cfg, layer_idx, total_layers)

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        # x shape [HW+1, BT, C]
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = self.Adapter(x)
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        L = cfg.ADAPTER.LAYERS
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(cfg, i, L) for i in range(L)])

    def forward(self, x):
        return self.resblocks(x)


@HEAD_REGISTRY.register()
class ViT_CLIP(nn.Module):
    def __init__(self, cfg):
        super(ViT_CLIP, self).__init__()
        self.args = cfg
        self.pretrained = cfg.ADAPTER.PRETRAINED
        self.width = cfg.ADAPTER.WIDTH
        self.patch_size = cfg.ADAPTER.PATCH_SIZE
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.width, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        scale = self.width ** -0.5
        self.layers = cfg.ADAPTER.LAYERS
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((cfg.DATA.TRAIN_CROP_SIZE // self.patch_size) ** 2 + 1, self.width))
        self.ln_pre = LayerNorm(self.width)
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, self.width))
        self.transformer = Transformer(cfg)
        self.ln_post = LayerNorm(self.width)
        # Always create classification layer for standard classification
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE") or getattr(cfg, "TASK_TYPE", "") in ["classification", "full_supervised"]:
            # Get dropout rate from config (default to 0.0 if not specified)
            dropout_rate = getattr(cfg.VIDEO.HEAD, "DROPOUT_RATE", 0.0) if hasattr(cfg, "VIDEO") and hasattr(cfg.VIDEO, "HEAD") else 0.0
            # Create dropout layer if dropout_rate > 0
            if dropout_rate > 0.0:
                self.dropout = nn.Dropout(dropout_rate)
            else:
                self.dropout = None
            self.classification_layer = nn.Linear(self.width, int(self.args.TRAIN.NUM_CLASS))
        self.init_weights()

    def init_weights(self):
        logger.info(f'load model from: {self.pretrained}')
        # Load OpenAI CLIP pretrained weights
        clip_model, _ = clip.load(self.pretrained, device="cpu")
        pretrain_dict = clip_model.visual.state_dict()
        del clip_model
        del pretrain_dict['proj']
        msg = self.load_state_dict(pretrain_dict, strict=False)
        logger.info('Missing keys: {}'.format(msg.missing_keys))
        logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        torch.cuda.empty_cache()
        # zero-initialize Adapters
        for n1, m1 in self.named_modules():
            if 'Adapter' in n1:
                for n2, m2 in m1.named_modules():
                    if 'up' in n2:
                        logger.info('init:  {}.{}'.format(n1, n2))
                        if hasattr(m2, 'weight') and m2.weight is not None:
                            nn.init.constant_(m2.weight, 0)
                        if hasattr(m2, 'bias') and m2.bias is not None:
                            nn.init.constant_(m2.bias, 0)

    def extract_class_indices(self, labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def get_feat(self, x):
        x = self.conv1(x)  # b*t c h w
        x = rearrange(x, 'b c h w -> b (h w) c')
        # b*t h*w+1 c
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        # n = h*w+1
        n = x.shape[1]

        x = rearrange(x, '(b t) n c -> (b n) t c', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t c -> (b t) n c', n=n)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x[:, 0, :]
        return x

    def forward(self, inputs):
        support_images, query_images = inputs['support_set'], inputs['target_set']
        support_features = self.get_feat(support_images)
        query_features = self.get_feat(query_images)
        support_labels = inputs['support_labels']
        unique_labels = torch.unique(support_labels)

        support_features = support_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)
        query_features = query_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)

        class_logits = None
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            class_logits = self.classification_layer(torch.cat([torch.mean(support_features, dim=1), torch.mean(query_features, dim=1)], 0))

        support_features = [torch.mean(torch.index_select(support_features, 0, self.extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_num = support_features.shape[0]
        query_num = query_features.shape[0]

        support_features = support_features.unsqueeze(0).repeat(query_num, 1, 1, 1)
        support_features = rearrange(support_features, 'q s t c -> q (s t) c')

        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(query_features, dim=2).permute(0, 2, 1)).reshape(query_num, support_num, self.num_frames, self.num_frames)
        dist = 1 - frame_sim

        # Bi-MHM
        class_dist = dist.min(3)[0].sum(2) + dist.min(2)[0].sum(2)

        # OTAM
        # class_dist = OTAM_dist(dist) + OTAM_dist(rearrange(dist, 'q s n m -> q s m n'))

        return_dict = {'logits': - class_dist, 'class_logits': class_logits}
        return return_dict

    def forward_classification(self, video, labels=None):
        """
        Forward pass for standard classification.
        Args:
            video: video tensor of shape [B*T, C, H, W]
            labels: optional labels tensor of shape [B]
        Returns:
            dict with 'logits' key containing classification logits
        """
        # Get features: get_feat expects [B*T, C, H, W] and returns [B*T, width]
        features = self.get_feat(video)  # [B*T, width]
        
        # Reshape to [B, T, width] and average over temporal dimension
        B = features.shape[0] // self.num_frames
        features = features.reshape(B, self.num_frames, self.width)
        features = torch.mean(features, dim=1)  # [B, width]
        
        # Apply dropout if enabled (for regularization)
        if hasattr(self, 'dropout') and self.dropout is not None:
            features = self.dropout(features)
        
        # Get classification logits
        if hasattr(self, 'classification_layer'):
            logits = self.classification_layer(features)  # [B, num_classes]
        else:
            raise ValueError("classification_layer not found. Make sure NUM_CLASS is set in config.")
        
        return {'logits': logits}
