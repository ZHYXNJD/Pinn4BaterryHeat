from __future__ import annotations

import math
import torch
from torch import nn


def sinusoidal_time_embedding(t: torch.Tensor, num_freqs: int = 4) -> torch.Tensor:
    """显式时间编码：sin/cos 多尺度，便于模型学习 T(t) 的时间依赖。t 已归一化到 [-1,1]。"""
    freqs = 2.0 * math.pi * torch.arange(1, num_freqs + 1, device=t.device, dtype=t.dtype)
    t_flat = t.squeeze(-1)
    args = freqs * t_flat.unsqueeze(-1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_features: int, sigma: float = 3.0):
        super().__init__()
        self.register_buffer("B", sigma * torch.randn(in_dim, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ConditionEncoder(nn.Module):
    def __init__(self, cond_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class PINNModel(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        hidden_layers: list[int],
        activation: str = "sine",
        fourier_features: int = 6,
        fourier_sigma: float = 3.0,
        time_embed_freqs: int = 0,
    ):
        super().__init__()
        act = self._make_activation(activation)
        self._is_sine = isinstance(act, Sine)
        self.cond_encoder = ConditionEncoder(cond_dim)
        self.use_fourier = fourier_features > 0
        self.fourier = FourierFeatures(4, fourier_features, sigma=fourier_sigma) if self.use_fourier else None
        self.time_embed_freqs = max(0, time_embed_freqs)

        input_dim = 4 + (2 * fourier_features if self.use_fourier else 0) + 64
        if self.time_embed_freqs > 0:
            input_dim += 2 * self.time_embed_freqs
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(act)
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        if self._is_sine:
            self._init_sine_weights(w0=30.0)

    def _make_activation(self, name: str):
        if name.lower() == "sine":
            return Sine()
        if name.lower() == "gelu":
            return nn.GELU()
        if name.lower() == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation {name}")

    def _init_sine_weights(self, w0: float = 30.0):
        first_linear = True
        for module in self.net:
            if isinstance(module, nn.Linear):
                in_dim = module.in_features
                if first_linear:
                    bound = 1.0 / in_dim
                    first_linear = False
                else:
                    bound = math.sqrt(6.0 / in_dim) / w0
                nn.init.uniform_(module.weight, -bound, bound)
                nn.init.zeros_(module.bias)

    def forward(self, coords: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_embed = self.cond_encoder(cond)
        parts = [coords]
        if self.use_fourier:
            parts.append(self.fourier(coords))
        if self.time_embed_freqs > 0:
            t = coords[:, 3:4]
            parts.append(sinusoidal_time_embedding(t, self.time_embed_freqs))
        parts.append(cond_embed)
        x = torch.cat(parts, dim=-1)
        return self.net(x)


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)
