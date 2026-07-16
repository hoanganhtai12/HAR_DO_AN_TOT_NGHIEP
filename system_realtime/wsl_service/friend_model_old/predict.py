"""predict.py -- load a trained bundle and predict one capture.

Implements the API-spec contract (mo_ta_api, 2026-06-24) for the AI side, with
the activity-name lookup folded in so the IoT side gets it for free. IoT gọi
thẳng hàm theo thiết bị của mình:

    from predict import predict_esp     # ESP:  mỗi rx shape (1, 56, 1000)
    label, name, percent = predict_esp(rx1, rx2, rx3)    # 1..8, str, [0,1]

    from predict import predict_asus    # ASUS: mỗi rx shape (4, 56, 1000)
    label, name, percent = predict_asus(rx1, rx2, rx3)

`predict()` (auto-detect thiết bị theo số anten) vẫn còn cho code cũ.

`rx1, rx2, rx3` are the three receivers in order (on disk: rx_00, rx_01, rx_02).
A bundle (bundle/esp.pt or bundle/asus.pt) is fully self-contained: weights +
z-score stats + the exact model kwargs + class names. chuyen_nhan_tu_so_sang_string(label)
is kept as a standalone helper for callers that already hold a label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import WavDualMamba
from .preprocess import build_sample, infer_antennas, zscore


# Bundle nằm cạnh package WSL này, không phụ thuộc thư mục terminal hiện tại.
BUNDLE_DIR = Path(__file__).resolve().parent / "bundle"


def _name_of(class_names, label):
    """1-based label (1..8) -> activity name; safe fallback to 'act<n>'."""
    i = int(label) - 1
    return class_names[i] if 0 <= i < len(class_names) else f"act{int(label)}"


class Predictor:
    def __init__(self, bundle_dir='bundle', device=None):
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device.type != 'cuda':
            print("[!] CUDA not available -- mamba_ssm needs a GPU; prediction will fail on CPU.")
        self._cache = {}                       # dev_name -> dict(model, mean, std, names)

    def _load(self, dev_name):
        if dev_name in self._cache:
            return self._cache[dev_name]
        path = self.bundle_dir / f"{dev_name}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"bundle '{path}' not found. Train the {dev_name.upper()} model first "
                f"(train.ipynb) so this device can be predicted.")
        b = torch.load(path, map_location=self.device, weights_only=False)
        model = WavDualMamba(**b['model_kwargs']).to(self.device).eval()
        model.load_state_dict(b['state_dict'])
        meta = b['meta']
        # mean/std load onto self.device (CUDA) via map_location; z-score runs in
        # numpy on the host, so move them to CPU before converting.
        mean, std = b['mean'], b['std']
        if torch.is_tensor(mean):
            mean = mean.cpu()
        if torch.is_tensor(std):
            std = std.cpu()
        entry = {
            'model': model,
            'mean': np.asarray(mean, dtype=np.float32),
            'std': np.asarray(std, dtype=np.float32),
            'n_antennas': meta['n_antennas'],
            'names': meta.get('class_names', []),
        }
        self._cache[dev_name] = entry
        print(f"  loaded bundle: {path.name}  (acc_test={meta.get('test_acc_last', '?')})")
        return entry

    # số anten mỗi thiết bị: ESP = 1 anten/receiver, ASUS = 4 anten/receiver.
    _A_OF = {'esp': 1, 'asus': 4}

    @torch.no_grad()
    def predict_device(self, dev_name, rx1, rx2, rx3):
        """Predict cho MỘT thiết bị đã biết ('esp'/'asus'): (label 1..8, name, percent 0..1).

        Chỉ định thẳng thiết bị (không đoán), kiểm tra số anten input có khớp
        thiết bị không, rồi chạy transform + model. Đây là phần lõi mà
        predict_esp/predict_asus và predict() (auto-detect) đều gọi vào.
        """
        if dev_name not in self._A_OF:
            raise ValueError(f"unknown device '{dev_name}' (expected 'esp' or 'asus')")
        A = infer_antennas(rx1)
        want = self._A_OF[dev_name]
        if A != want:
            raise ValueError(
                f"predict_{dev_name}() cần {want} anten/receiver nhưng input có A={A} "
                f"({'ESP=1' if dev_name == 'esp' else 'ASUS=4'}) -- gọi nhầm hàm?")
        entry = self._load(dev_name)
        x = build_sample([rx1, rx2, rx3], A)               # (C,T2,F2)
        x = zscore(x, entry['mean'], entry['std'])
        xb = torch.from_numpy(x[None]).to(self.device)     # (1,C,T2,F2)
        prob = entry['model'](xb).softmax(1)[0].cpu().numpy()
        idx = int(prob.argmax())
        return idx + 1, _name_of(entry['names'], idx + 1), float(prob[idx])

    def predict(self, rx1, rx2, rx3):
        """Auto-detect thiết bị theo số anten (1->esp, 4->asus) rồi predict.

        Giữ cho code cũ; nếu đã biết thiết bị thì gọi thẳng predict_device().
        """
        A = infer_antennas(rx1)
        dev_name = 'esp' if A == 1 else ('asus' if A == 4 else None)
        if dev_name is None:
            raise ValueError(f"unsupported antenna count A={A} (expected 1=ESP or 4=ASUS)")
        return self.predict_device(dev_name, rx1, rx2, rx3)

    def label_to_string(self, label):
        """1-based label (1..8) -> activity name, from whichever bundle is loaded."""
        names = next((e['names'] for e in self._cache.values() if e['names']), None)
        if not names:
            try:
                from data import CLASS_NAMES
                names = CLASS_NAMES
            except Exception:
                names = []
        return _name_of(names, label)


# --- Module-level API exactly as the spec pseudocode expects --------------------
# (a process-wide default Predictor, lazily created from ./bundle)

_DEFAULT = None


def get_default_predictor(bundle_dir=BUNDLE_DIR):
    """Lazily build (once) and return the process-wide default Predictor."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Predictor(bundle_dir)
    return _DEFAULT


def predict(rx1, rx2, rx3):
    """API: label, name, percent = predict(rx1, rx2, rx3). label 1..8, percent in [0,1].

    Auto-detect thiết bị theo số anten. Nếu đã biết thiết bị thì dùng
    predict_esp / predict_asus cho rõ ràng.
    """
    return get_default_predictor().predict(rx1, rx2, rx3)


def predict_esp(rx1, rx2, rx3):
    """ESP (1 anten/receiver): mỗi rx shape (1, 56, 1000) -> (label 1..8, name, percent)."""
    return get_default_predictor().predict_device('esp', rx1, rx2, rx3)


def predict_asus(rx1, rx2, rx3):
    """ASUS (4 anten/receiver): mỗi rx shape (4, 56, 1000) -> (label 1..8, name, percent)."""
    return get_default_predictor().predict_device('asus', rx1, rx2, rx3)


def chuyen_nhan_tu_so_sang_string(label):
    """Helper: 1-based label (1..8) -> activity name string."""
    return get_default_predictor().label_to_string(label)
