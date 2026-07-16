"""preprocess.py -- canonical sshar -> WavDualMamba(s4.nogn_gate) transform.

This single module is the ONE source of truth for turning a captured sample
(3 receivers of raw CSI amplitude) into the packed Haar tensor the model eats.
BOTH training (train.py / train.ipynb) and real-time prediction (run.py) call
build_sample() here, so the channel layout is guaranteed identical end to end.

Pipeline (raw mode -- NO Hampel / NO low-pass filter; fs is unknown):
    rx0,rx1,rx2 each (A, 56, 1000) or (A*56, 1000)   # A = antennas per receiver
      -> stack along axis 0 (FIXED order rx0,rx1,rx2) -> (3A*56, 1000)
      -> 2-D Haar DWT (periodization)                 -> HL, LH each (500, 3A*28)
      -> pack [HL | LH] via to_maps(., 3A)            -> (6A, 500, 28)
      -> per-position z-score with stats saved at train time

Shapes:  ESP  (A=1) -> ( 6, 500, 28)
         ASUS (A=4) -> (24, 500, 28)

The model is then WavDualMamba(n_links=3, n_antennas=A, f2=28, subbands=('HL','LH')),
so n_per_sub = 3*A and the packed channel count 2*3*A matches exactly.

Subband convention is IDENTICAL to the repo build (xrf55_bench/preprocessing/
multi_dataset.py): HL = cV.T, LH = cH.T from pywt.dwt2(flat,'haar','periodization').
Only HL and LH are packed (s4.nogn_gate drops LL).
"""
from __future__ import annotations

import numpy as np
import pywt

# --- Fixed dataset / model constants (do NOT change without retraining) --------
SUB        = 56               # subcarriers per receiver-antenna (must be even);
                              # ESP32 HT20 keeps 52 data + 4 pilot, nulls dropped
TIME       = 1000             # packets per 5s capture window
F2         = SUB // 2         # 28  -- subcarrier axis after Haar
T2         = TIME // 2        # 500 -- time axis after Haar
N_LINKS    = 3               # 3 receivers, stacked -> treated as model "n_links"
SUBBANDS   = ('HL', 'LH')    # s4.nogn_gate: HL + LH only (no LL)
RX_ORDER   = ('rx0', 'rx1', 'rx2')   # FIXED stacking order, train == predict


def to_maps(a, n_per_sub):
    """(T, n_per_sub*f2) -> (n_per_sub, T, f2). Unflatten link-major feature axis.

    Byte-identical to xrf55_bench.preprocessing.multi_dataset.to_maps.
    """
    T, M = a.shape
    f2 = M // n_per_sub
    return a.reshape(T, n_per_sub, f2).transpose(1, 0, 2)


def _to_amplitude(arr):
    """Return real non-negative amplitude. abs() only if the file is complex."""
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr.astype(np.float32, copy=False)


def _orient_feat_time(arr):
    """Force layout (feature=A*SUB rows, time cols).

    Accepts either a 2-D rx array (feat,time)/(time,feat) or a 3-D rx array
    (antenna, subcarrier, time) as stored by the 2slab capture system -- in the
    3-D case the antenna axis is merged into the feature axis: (A,SUB,T)->(A*SUB,T).

    The feature axis length is a multiple of SUB; the time axis (~1000) is not.
    We pick the feature axis as the one divisible by SUB; if ambiguous we fall
    back to 'time is the longer axis'.
    """
    if arr.ndim == 3:
        a0, a1, a2 = arr.shape           # (antenna, subcarrier, time)
        if a1 % SUB == 0:                # (A, SUB, T) -> (A*SUB, T)
            arr = arr.reshape(a0 * a1, a2)
        elif a2 % SUB == 0:              # (A, T, SUB) -> (A*SUB, T)
            arr = arr.transpose(0, 2, 1).reshape(a0 * a2, a1)
        else:
            arr = arr.reshape(a0 * a1, a2)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (feat,time) or 3-D (A,SUB,time) rx array, got {arr.shape}")
    r, c = arr.shape
    feat0 = (r % SUB == 0)
    feat1 = (c % SUB == 0)
    if feat0 and not feat1:
        return arr
    if feat1 and not feat0:
        return arr.T
    # ambiguous (rare) -> assume time is the longer axis
    return arr if r <= c else arr.T


def _fix_time(arr, time=TIME):
    """Center-crop / symmetric zero-pad the time axis (cols) to `time`.

    The model itself is length-agnostic (Mamba + AttnStatPool), but it was
    trained on 1000-packet windows, so we normalise the window length to keep
    train/predict on the same distribution. Used by csi_cut_tool semantics.
    """
    f, t = arr.shape
    if t == time:
        return arr
    if t > time:                                  # center crop
        s = (t - time) // 2
        return arr[:, s:s + time]
    pad = time - t                                # symmetric zero-pad
    left = pad // 2
    return np.pad(arr, ((0, 0), (left, pad - left)), mode='constant')


def infer_antennas(rx_arr):
    """Antennas per receiver A from one rx file (= feature rows / SUB)."""
    a = _orient_feat_time(_to_amplitude(rx_arr))
    rows = a.shape[0]
    if rows % SUB != 0:
        raise ValueError(f"rx feature rows {rows} not a multiple of {SUB} subcarriers")
    return rows // SUB


def haar_HL_LH(flat):
    """(3A*56, 1000) amplitude -> (HL, LH), each (500, 3A*28). Raw, no filter.

    Identical math to multi_dataset.haar3_subbands(do_filter=False), HL/LH only.
    Haar is 2-tap and every receiver-antenna block has SUB (even) subcarriers, so
    the DWT on the merged axis never mixes receivers/antennas.
    """
    flat = np.asarray(flat, dtype=np.float32)
    cA, (cH, cV, _) = pywt.dwt2(flat, 'haar', mode='periodization')
    HL = cV.T.astype(np.float32)
    LH = cH.T.astype(np.float32)
    return HL, LH


def build_sample(rx_list, n_antennas):
    """3 receiver arrays -> packed model input (2*3*A, 500, 28) float32.

    rx_list   : [rx0, rx1, rx2] raw amplitude arrays (any orientation, ~1000 time).
    n_antennas: A (1 for ESP, 4 for ASUS).

    The 3 receivers are stacked in FIXED order (rx0,rx1,rx2); this defines the
    channel identity and MUST be the same at train and predict time.
    """
    if len(rx_list) != N_LINKS:
        raise ValueError(f"need exactly {N_LINKS} receivers (rx0,rx1,rx2), got {len(rx_list)}")
    n_per_sub = N_LINKS * n_antennas          # 3A
    blocks = []
    for k, rx in enumerate(rx_list):
        a = _fix_time(_orient_feat_time(_to_amplitude(rx)))      # (A*56, 1000)
        if a.shape[0] != n_antennas * SUB:
            raise ValueError(
                f"{RX_ORDER[k]}: feature rows {a.shape[0]} != A*{SUB}={n_antennas * SUB} "
                f"(mismatched antenna count / device)")
        blocks.append(a)
    flat = np.concatenate(blocks, axis=0)                        # (3A*56, 1000)
    HL, LH = haar_HL_LH(flat)                                    # each (500, 3A*28)
    x = np.concatenate(
        [to_maps(HL, n_per_sub), to_maps(LH, n_per_sub)], axis=0
    ).astype(np.float32, copy=False)                            # (6A, 500, 28)
    return x


def zscore(x, mean, std):
    """Per-position z-score. mean/std are (C,T2,F2) or (C,1,F2) (broadcast)."""
    return ((x - mean) / std).astype(np.float32, copy=False)


def fit_stats(X, chunk=256):
    """Per-position mean/std over samples. X: (N, C, T2, F2) -> each (C, T2, F2).

    Streaming float64 accumulation (sum + sum-of-squares) in chunks, so peak
    memory is ~one chunk in float64 instead of a full float64 copy of X. Result
    is numerically identical to the all-at-once E[X^2]-E[X]^2 build (per-position,
    std floored 1e-6; matches scripts/10_build_multi.py).
    """
    X = np.asarray(X)
    n = X.shape[0]
    s1 = np.zeros(X.shape[1:], dtype=np.float64)             # sum_i x
    s2 = np.zeros(X.shape[1:], dtype=np.float64)             # sum_i x^2
    for i in range(0, n, chunk):
        xb = X[i:i + chunk].astype(np.float64)
        s1 += xb.sum(axis=0)
        s2 += (xb * xb).sum(axis=0)
    mean = s1 / n                                            # (C,T2,F2)
    var = np.maximum(s2 / n - mean ** 2, 0.0)                # E[X^2]-E[X]^2
    std = np.maximum(np.sqrt(var), 1e-6)                     # floor like the build
    return mean.astype(np.float32), std.astype(np.float32)
