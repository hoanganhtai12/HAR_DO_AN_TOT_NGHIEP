"""model.py -- WavDualMamba, s4.nogn_gate configuration ONLY (minimal).

This is the trimmed model used by the standalone sshar package. It implements
exactly ONE architecture -- the "s4.nogn_gate" rung -- with every ablation branch
removed (no LL subband, no ECA / FreqMix / positional-embedding / shared
backbone / final attention / post-fusion projection / GAP / convex|concat
fusion / FFN / stem GroupNorm). The remaining math is byte-identical to the full
WavDualMamba run with:

    subbands=('HL','LH'), stem_norm=False, fusion='gate', pool='attnstat',
    bidirectional=True, n_mamba_layers=2, d_model=64, d_stem=16, d_state=32.

Forward path (per the two Haar detail subbands HL, LH):
    packed input X (B, 2*n_per_sub, T2, F2) = [HL | LH]   (n_per_sub = n_links*n_antennas)
      per subband s:
        SubbandStem_s   Conv2d(n_per_sub -> d_stem, kernel_s) + SiLU       (no GroupNorm)
        TFBlock x3      dilated axial-depthwise (dilations 1,2,4)
        flatten F x C   (B, d_stem, T2, F2) -> (B, T2, d_stem*F2)          (KEEP frequency)
        Linear embed    d_stem*F2 -> d_model + SiLU + Dropout
        BiMamba x2      gated fwd/bwd merge (per-channel zero-init gate)
      AdaptiveFusionGate  per-channel softmax over the 2 branches (zero-init -> mean)
      AttnStatPool        ECAPA attentive mean||std over time -> 2*d_model
      Classifier          LayerNorm -> Dropout -> Linear -> logits

Kernels (physically motivated): HL=(3,7) temporal burst onsets, LH=(7,3) Doppler.
Requires mamba_ssm (CUDA).
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    _MAMBA_IMPORT_ERROR = None
except Exception as e:                                   # pragma: no cover
    Mamba = None
    HAS_MAMBA = False
    _MAMBA_IMPORT_ERROR = e


# s4.nogn_gate uses only the two Haar detail subbands, in this canonical order.
_SUBBANDS = ('HL', 'LH')
_SUBBAND_KERNEL = {'HL': (3, 7), 'LH': (7, 3)}
#   HL (3,7) temporal burst onsets  ·  LH (7,3) Doppler-like


def _gn_groups(d: int) -> int:
    """Largest divisor of d that is <= d//8 (>= ~8 channels/group); 1 if none."""
    for g in range(max(1, d // 8), 0, -1):
        if d % g == 0:
            return g
    return 1


# --- Stochastic depth ----------------------------------------------------------

class DropPath(nn.Module):
    """Drop whole samples from a residual branch during training."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


# --- CNN front-end -------------------------------------------------------------

class SubbandStem(nn.Module):
    """Conv2d(in_ch -> d_stem) + SiLU, one per subband (no GroupNorm: stem_norm=False)."""

    def __init__(self, in_ch: int, d_stem: int = 16, kernel=(5, 5)):
        super().__init__()
        kt, kf = kernel
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d_stem, (kt, kf), padding=(kt // 2, kf // 2)),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TFBlock(nn.Module):
    """Pre-norm axial-depthwise block: GN -> dw_t -> dw_f -> pw(SiLU) -> residual."""

    def __init__(self, d: int, k_t: int = 7, k_f: int = 3,
                 dilation: int = 1, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(_gn_groups(d), d)
        self.dw_t = nn.Conv2d(d, d, (k_t, 1),
                              padding=(k_t // 2 * dilation, 0),
                              dilation=(dilation, 1), groups=d)
        self.dw_f = nn.Conv2d(d, d, (1, k_f), padding=(0, k_f // 2), groups=d)
        self.act  = nn.SiLU()
        self.pw   = nn.Conv2d(d, d, 1)
        self.dp   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dw_f(self.dw_t(y))
        y = self.pw(self.act(y))
        return x + self.dp(y)


# --- Bidirectional gated Mamba -------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps)


class BiMambaLayer(nn.Module):
    """Bidirectional Mamba with per-channel zero-init gated fwd/bwd merge.

        h = RMSNorm(x); f = Mamba_fwd(h); b = flip(Mamba_bwd(flip(h)))
        g = sigmoid(W[f||b]+c)   (W=0,c=0 at init -> g==0.5)
        y = g*f + (1-g)*b;  x = x + DropPath(y)
    """

    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4,
                 expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is required (CUDA).\n"
                "Install: pip install mamba-ssm[causal-conv1d] --no-build-isolation\n"
                f"Original import error: {_MAMBA_IMPORT_ERROR}")
        self.norm = RMSNorm(d_model)
        self.fwd  = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd  = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.gate.weight)       # g == 0.5 at init -> 1/2 (fwd+bwd)
        nn.init.zeros_(self.gate.bias)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        f = self.fwd(h)
        b = self.bwd(h.flip(1)).flip(1)
        g = torch.sigmoid(self.gate(torch.cat([f, b], dim=-1)))
        return x + self.dp(g * f + (1.0 - g) * b)


class BiMamba(nn.Module):
    """Stack of gated bidirectional Mamba layers + final RMSNorm."""

    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 32,
                 d_conv: int = 4, expand: int = 2, drop_path_rates=(0.0, 0.10)):
        super().__init__()
        if len(drop_path_rates) != n_layers:
            raise ValueError("len(drop_path_rates) must equal n_layers")
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         drop_path=drop_path_rates[i])
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# --- Per-subband branch backbone ----------------------------------------------

class BranchBackbone(nn.Module):
    """TFBlock x3 (dilation 1,2,4) -> flatten F x C -> Linear embed -> BiMamba.

    Input  : (B, d_stem, T2, F2)    Output: (B, T2, d_model)
    """

    def __init__(self, d_stem: int, f2: int, d_model: int = 64,
                 dp_cnn=(0.0, 0.05, 0.1), dilations=(1, 2, 4),
                 n_mamba_layers: int = 2, d_state: int = 32, d_conv: int = 4,
                 expand: int = 2, dp_mamba=(0.0, 0.10), embed_drop: float = 0.1):
        super().__init__()
        if len(dp_cnn) != len(dilations):
            raise ValueError("len(dp_cnn) must equal len(dilations)")
        self.blocks = nn.ModuleList([
            TFBlock(d_stem, dilation=dilations[i], drop_path=dp_cnn[i])
            for i in range(len(dilations))
        ])
        self.embed = nn.Sequential(
            nn.Linear(d_stem * f2, d_model),
            nn.SiLU(),
            nn.Dropout(embed_drop),
        )
        self.mamba = BiMamba(d_model, n_layers=n_mamba_layers, d_state=d_state,
                             d_conv=d_conv, expand=expand, drop_path_rates=dp_mamba)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)                                   # (B, d_stem, T2, F2)
        B, C, T, Fd = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fd)  # flatten F x C -> feature
        x = self.embed(x)                                # (B, T2, d_model)
        return self.mamba(x)                             # (B, T2, d_model)


# --- Per-channel gated fusion (fusion='gate') ----------------------------------

class AdaptiveFusionGate(nn.Module):
    """Merge the 2 branch streams by per-channel convex routing (zero-init -> mean).

        a   = Linear_{2d->2d}(concat(S_HL, S_LH)) -> (B, T, 2, d)
        alpha = softmax(a, dim=branch)             per token, per CHANNEL
        out = sum_i alpha_i * S_i                  (alpha >= 0, sum_i alpha_i = 1)

    Linear is zero-init so softmax -> 1/2 per channel at step 0 (== mean of the
    two branches). Identical to WavDualMamba.AdaptiveFusion(mode='gate', n=2).
    """

    def __init__(self, d_model: int, n_branches: int = 2):
        super().__init__()
        self.n_branches = n_branches
        self.d_model = d_model
        self.proj = nn.Linear(n_branches * d_model, n_branches * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, streams):
        cat = torch.cat(streams, dim=-1)                 # (B, T, N*d)
        B, T, _ = cat.shape
        w = self.proj(cat).view(B, T, self.n_branches, self.d_model)
        w = w.softmax(dim=2)                             # (B, T, N, d)
        s = torch.stack(streams, dim=2)                  # (B, T, N, d)
        return (w * s).sum(dim=2)                        # (B, T, d)


# --- Temporal attentive statistics pooling (ECAPA-style, zero-init) ------------

class AttnStatPool(nn.Module):
    """Per-channel temporal attention -> [weighted mean || weighted std].

    Input : (B, T, d)    Output: (B, 2*d)
    Score sees [x || mu || sigma] (full ECAPA context, pool_context=True).
    """

    def __init__(self, dim: int, bn: int = None):
        super().__init__()
        bn = bn or max(8, dim // 2)
        self.score = nn.Sequential(
            nn.Linear(3 * dim, bn), nn.Tanh(), nn.Linear(bn, dim))
        nn.init.zeros_(self.score[-1].weight)            # uniform attention init
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True).expand_as(x)
        sg = x.var(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6).sqrt().expand_as(x)
        h = torch.cat([x, mu, sg], dim=-1)               # (B, T, 3*d)
        w = self.score(h).softmax(dim=1)                 # (B, T, d)
        mean = (w * x).sum(dim=1)                        # (B, d)
        var = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        return torch.cat([mean, var.clamp(min=1e-6).sqrt()], dim=-1)


# --- Classifier head -----------------------------------------------------------

class Classifier(nn.Module):
    """LayerNorm -> Dropout -> Linear."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# --- Full model (s4.nogn_gate) -------------------------------------------------

class WavDualMamba(nn.Module):
    """s4.nogn_gate: per-subband (HL, LH) CNN + gated BiMamba + per-channel gate
    fusion + attentive-stat pooling. Minimal -- no ablation flags.

    Args:
        num_classes : output classes.
        n_links     : Wi-Fi receivers stacked (sshar: 3).
        n_antennas  : antennas per receiver (ESP=1, ASUS=4).
        f2          : subcarrier axis after Haar (sshar 56-sub -> 28).
        d_model, d_stem, d_state, n_mamba_layers : capacity knobs (defaults match
                      the trained checkpoint).
        dropout     : classifier dropout.

    The other constructor args (subbands/pool/stem_norm/fusion) are accepted for
    bundle-kwargs compatibility but FIXED -- passing anything else raises.

    Input  : X (B, 2*n_per_sub, T2, F2) packed [HL | LH] (n_per_sub = n_links*n_antennas).
    Output : logits (B, num_classes).
    """

    def __init__(self, num_classes: int, n_links: int = 3, n_antennas: int = 1,
                 f2: int = 28, d_model: int = 64, d_stem: int = 16,
                 d_state: int = 32, n_mamba_layers: int = 2, dropout: float = 0.2,
                 subbands=('HL', 'LH'), pool='attnstat', stem_norm=False,
                 fusion='gate'):
        super().__init__()
        # Locked configuration -- this minimal model implements only s4.nogn_gate.
        if tuple(subbands) != _SUBBANDS:
            raise ValueError(f"this build is fixed to subbands={_SUBBANDS}, got {subbands!r}")
        if pool != 'attnstat':
            raise ValueError(f"this build is fixed to pool='attnstat', got {pool!r}")
        if stem_norm:
            raise ValueError("this build is fixed to stem_norm=False (s4.nogn)")
        if fusion != 'gate':
            raise ValueError(f"this build is fixed to fusion='gate', got {fusion!r}")

        self.subbands  = _SUBBANDS
        self.f2        = f2
        self.n_per_sub = n_links * n_antennas            # channels per subband

        self.stems = nn.ModuleDict({
            s: SubbandStem(self.n_per_sub, d_stem, kernel=_SUBBAND_KERNEL[s])
            for s in self.subbands
        })
        self.backbones = nn.ModuleDict({
            s: BranchBackbone(d_stem=d_stem, f2=f2, d_model=d_model,
                              n_mamba_layers=n_mamba_layers, d_state=d_state)
            for s in self.subbands
        })
        self.fusion = AdaptiveFusionGate(d_model, n_branches=len(self.subbands))
        self.tpool  = AttnStatPool(d_model)
        self.head   = Classifier(2 * d_model, num_classes=num_classes, dropout=dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 4:
            raise ValueError(f"expected 4-D input (B, C, T2, F2), got {tuple(X.shape)}")
        packed_c = self.n_per_sub * len(self.subbands)
        if X.shape[1] != packed_c:
            raise ValueError(
                f"expected {packed_c} channels (packed [HL|LH], n_per_sub={self.n_per_sub}), "
                f"got {X.shape[1]}")
        if X.shape[-1] != self.f2:
            raise ValueError(
                f"expected F2={self.f2} subcarriers, got {X.shape[-1]}")

        streams = []
        for k, s in enumerate(self.subbands):
            sb = X[:, k * self.n_per_sub:(k + 1) * self.n_per_sub]   # (B, n_per_sub, T2, F2)
            streams.append(self.backbones[s](self.stems[s](sb)))     # (B, T2, d_model)
        z = self.fusion(streams)                                     # (B, T2, d_model)
        z = self.tpool(z)                                            # (B, 2*d_model)
        return self.head(z)                                          # (B, num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
