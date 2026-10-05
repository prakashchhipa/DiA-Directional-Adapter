#!/usr/bin/env python3

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


class CausalAntiCausalProbe:
    """
    Runtime probe for ViT_FineGrained_ST_CausalAntiCausal_Adapter.

    This class monkey-patches the target adapter's `_causal_anticausal` method to
    collect per-frame branch energies (causal, anti-causal, fused) and fusion
    weights, without modifying `models/base/adapter.py`.
    """

    def __init__(self, adapter_module, keep_history: bool = False):
        self.adapter = adapter_module
        self.keep_history = keep_history
        self.latest: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self._original_method = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return

        self._original_method = self.adapter._causal_anticausal
        probe = self

        def _wrapped_causal_anticausal(y_flat):
            T = y_flat.shape[-1]
            k = probe.adapter.temporal_causal.kernel_size[0]
            pad = k - 1

            y_causal = F.pad(y_flat, (pad, 0), mode="replicate")
            y_causal = probe.adapter.temporal_causal(y_causal)[:, :, :T]

            y_anticausal = F.pad(y_flat, (0, pad), mode="replicate")
            y_anticausal = probe.adapter.temporal_anticausal(y_anticausal)[:, :, -T:]

            w = F.softmax(probe.adapter.temporal_direction_weight, dim=0)
            y_fused = w[0] * y_causal + w[1] * y_anticausal

            with torch.no_grad():
                payload = {
                    "causal_weight": float(w[0].detach().cpu().item()),
                    "anticausal_weight": float(w[1].detach().cpu().item()),
                    "causal_energy_per_frame": y_causal.detach().abs().mean(dim=(0, 1)).cpu(),
                    "anticausal_energy_per_frame": y_anticausal.detach().abs().mean(dim=(0, 1)).cpu(),
                    "fused_energy_per_frame": y_fused.detach().abs().mean(dim=(0, 1)).cpu(),
                }
                probe.latest = payload
                if probe.keep_history:
                    probe.history.append(payload)

            return y_fused

        self.adapter._causal_anticausal = _wrapped_causal_anticausal
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.adapter._causal_anticausal = self._original_method
        self._original_method = None
        self._installed = False

    def get_latest(self, clear: bool = False) -> Optional[Dict[str, Any]]:
        payload = self.latest
        if clear:
            self.latest = None
        return payload

    def get_history(self, clear: bool = False) -> List[Dict[str, Any]]:
        payload = self.history
        if clear:
            self.history = []
        return payload
