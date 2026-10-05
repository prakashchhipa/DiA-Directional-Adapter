"""
Lightweight Fine-Discriminative Adapter for ViT CLIP.

Designed for fine-grained action recognition (e.g. Diving48, FineGym99).
- Temporal difference stream: frame-to-frame differences capture fine motion/discriminative cues (TDN-style).
- Multi-scale temporal convolution: depthwise 1D convs at multiple dilations capture short and long temporal context.
- No attention; all operations are linear + depthwise conv for minimal parameters.
- Zero-initialized output projection so the adapter starts as identity (residual).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ViT_FineDiscriminativeAdapter(nn.Module):
    """
    Lightweight adapter that captures fine discriminative temporal patterns for fine-grained action recognition.
    Uses:
    1. Temporal difference (local motion): diff along time -> depthwise 1D conv.
    2. Multi-scale temporal conv: parallel depthwise 1D convs with different dilations.
    No attention; fits onto frozen ViT CLIP with minimal parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        c = cfg.ADAPTER.WIDTH
        r = max(1, int(c * cfg.ADAPTER.ADAPTER_SCALE))

        self.down = nn.Linear(c, r, bias=False)

        # --- Temporal difference branch (local motion / fine discriminative) ---
        k_diff = getattr(cfg.ADAPTER, "FINE_DIFF_K", 3)
        self.diff_conv = nn.Conv1d(r, r, kernel_size=k_diff, padding=(k_diff - 1) // 2, groups=r, bias=False)

        # --- Multi-scale temporal context (short + long range, no attention) ---
        dilations = getattr(cfg.ADAPTER, "FINE_MS_DILATIONS", [1, 2, 3])
        k_t = getattr(cfg.ADAPTER, "TCONV_K", 5)
        self.ms_convs = nn.ModuleList()
        for d in dilations:
            pad = ((k_t - 1) * d) // 2
            self.ms_convs.append(
                nn.Conv1d(r, r, kernel_size=k_t, padding=pad, dilation=d, groups=r, bias=False)
            )
        self.ms_fusion = nn.Parameter(torch.ones(len(dilations)) / len(dilations))

        # Fuse diff + multi-scale (learnable balance)
        self.fuse_diff_scale = nn.Parameter(torch.tensor(0.5))
        self.fuse_ms_scale = nn.Parameter(torch.tensor(0.5))

        self.up = nn.Linear(r, c, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        # x: [N, BT, C], N = 1 + H*W, BT = B*T
        n, bt, c = x.shape
        T = self.args.DATA.NUM_INPUT_FRAMES
        if T <= 1:
            return x
        B = bt // T
        H = round(math.sqrt(n - 1))
        cls, tokens = x[0:1], x[1:]
        # (B*N, T, C)
        tokens_bt = rearrange(tokens, "n (b t) c -> (b n) t c", b=B, t=T, n=H * H)

        y = self.down(tokens_bt)  # (B*N, T, r)
        y_t = y.transpose(1, 2)   # (B*N, r, T)

        # --- Branch 1: Temporal difference (local motion) ---
        diff = y[:, 1:, :] - y[:, :-1, :]  # (B*N, T-1, r)
        diff = F.pad(diff, (0, 0, 0, 1), mode="replicate")  # (B*N, T, r); replicate last for T-1 -> T
        diff_t = diff.transpose(1, 2)  # (B*N, r, T)
        out_diff = self.diff_conv(diff_t)  # (B*N, r, T)

        # --- Branch 2: Multi-scale temporal conv ---
        ms_list = [conv(y_t) for conv in self.ms_convs]
        w = F.softmax(self.ms_fusion, dim=0)
        out_ms = sum(wi * m for wi, m in zip(w, ms_list))  # (B*N, r, T)

        # Fuse: scale both branches and add (bounded scales for stability)
        scale_diff = torch.sigmoid(self.fuse_diff_scale)
        scale_ms = torch.sigmoid(self.fuse_ms_scale)
        out_fused = scale_diff * out_diff + scale_ms * out_ms  # (B*N, r, T)
        out_fused = out_fused.transpose(1, 2)  # (B*N, T, r)

        y_out = self.up(out_fused)  # (B*N, T, c)
        res = rearrange(tokens_bt, "(b n) t c -> n (b t) c", b=B, n=H * H)
        y_out = rearrange(y_out, "(b n) t c -> n (b t) c", b=B, n=H * H)
        out = torch.cat([cls, res + y_out], dim=0)
        return out
