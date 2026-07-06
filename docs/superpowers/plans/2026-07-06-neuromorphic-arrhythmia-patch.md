# Neuromorphic Arrhythmia Patch — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a spiking-neural-network (SNN) arrhythmia detector on single-lead ECG — training/evaluating on public PhysioNet datasets under the AAMI inter-patient paradigm, beating or matching CNN/LSTM baselines, with a hardware-faithful spike encoder and an energy proxy — as the de-risking software core of a non-invasive neuromorphic cardiac-monitoring patch.

**Architecture:** A `src/`-layout Python package (`neurocardio`). ECG → bandpass/normalize → R-peak-centered beat windows → AAMI 5-class labels (N/SVEB/VEB/F/Q) → a level-crossing **delta spike encoder** (mirrors a hardware analog front-end) → a small **LIF SNN** trained with surrogate-gradient BPTT (snnTorch). Everything is validated on a strict **inter-patient split** (de Chazal DS1 train / DS2 test) to prevent the beat-leakage that inflates most published ECG numbers. Baselines (1D-CNN, LSTM) share the same pipeline. A deployment abstraction plus a synaptic-operations energy proxy make the "low-power neuromorphic" claim measurable now and portable to MCU/Loihi later. A streaming detector runs the trained model over a continuous record and logs timestamped anomaly detections.

**Tech Stack:** Python 3.11+, PyTorch ≥2.2, snnTorch ≥0.9 (LIF neurons + surrogate gradients), wfdb ≥4.1 (PhysioNet I/O), NumPy, SciPy (filtering, peak finding), scikit-learn (metrics), PyYAML (config), pytest + ruff, `uv` for env/deps.

---

## Why this is Phase 1 (and what it de-risks)

The single largest technical risk in this startup is not the patch, the regulator, or the pilot — it is **whether an SNN small enough to run on a sub-milliwatt chip can detect arrhythmias at clinically credible sensitivity on data it has never seen from patients it has never seen.** Everything downstream (hardware BOM, IEC 62304 file, clinical pilots, fundraising) is wasted motion until that is answered with honest numbers. This plan builds exactly that answer in software, using free public data, with methodology strict enough that the results survive regulatory and investor scrutiny.

**Two methodological guardrails are baked into the plan and are non-negotiable:**

1. **Inter-patient split (DS1/DS2).** Beats from one patient must never appear in both train and test. Intra-patient (random) splits routinely report 99% accuracy that collapses to ~85% or worse inter-patient. We report inter-patient from day one. Investors and notified bodies will ask; we answer honestly.
2. **AAMI EC57 class grouping + per-class metrics.** Overall accuracy is meaningless under ~90% normal-beat prevalence. We report per-class **sensitivity** and **positive predictivity** for VEB and SVEB — the numbers a cardiologist and a regulator actually read.

---

## File Structure

```
neurocardio/                          # repo root (C:\Users\valer\neuromorphic)
  pyproject.toml                      # deps, build backend, ruff + pytest config
  README.md                           # quickstart, methodology notes, results table
  configs/
    default.yaml                      # canonical experiment config
  src/neurocardio/
    __init__.py
    config.py                         # dataclass config + YAML loader
    data/
      __init__.py
      download.py                     # PhysioNet fetch wrappers (mitdb, ptbdb)
      records.py                      # load_record: wfdb -> Record dataclass + AAMI map
      preprocess.py                   # bandpass_filter, normalize
      segment.py                      # AAMI symbol map + segment_beats
      splits.py                       # DS1/DS2/paced record lists + get_split
      dataset.py                      # ECGBeatDataset (torch), build_dataset
    encoding/
      __init__.py
      delta.py                        # delta_encode / delta_decode (level-crossing)
      rate.py                         # rate_encode (baseline encoder)
    models/
      __init__.py
      snn.py                          # SNNClassifier (LIF, surrogate grad)
      baselines.py                    # CNN1D, LSTMClassifier
    train/
      __init__.py
      loop.py                         # set_seed, train
    eval/
      __init__.py
      metrics.py                      # confusion, aami_metrics
      evaluate.py                     # evaluate(model, loader)
    deploy/
      __init__.py
      energy.py                       # synaptic_operations, spike_stats
    stream/
      __init__.py
      qrs.py                          # find_r_peaks (online R-peak detector)
      detector.py                     # Detection, StreamDetector
    cli.py                            # download / prepare / train / evaluate / stream
  tests/
    test_config.py
    test_records.py
    test_preprocess.py
    test_segment.py
    test_splits.py
    test_dataset.py
    test_delta.py
    test_rate.py
    test_snn.py
    test_baselines.py
    test_train.py
    test_metrics.py
    test_evaluate.py
    test_energy.py
    test_qrs.py
    test_detector.py
    test_cli_smoke.py
  docs/superpowers/plans/2026-07-06-neuromorphic-arrhythmia-patch.md   # this file
```

Each module has one responsibility. Data lives together (I/O, preprocess, segment, split, dataset). Encoding, models, training, eval, deploy, and streaming are separate so a single file stays holdable in context. Tests mirror the module tree 1:1.

---

## Conventions used by every task

- **Run tests with:** `uv run pytest tests/<file>::<test> -v` (from repo root).
- **Run all tests:** `uv run pytest -q`.
- **Lint before commit:** `uv run ruff check . && uv run ruff format --check .`
- **AAMI class order is fixed everywhere:** `AAMI_CLASSES = ["N", "SVEB", "VEB", "F", "Q"]` → indices `0..4`. Never reorder.
- **Beat window length** `L = window_before + window_after = 256` (128 + 128 at fs=360, ≈0.71 s).
- **Delta encoder output shape** is `[L, 2]` (column 0 = up spikes, column 1 = down spikes), dtype float32, values in {0,1}.
- **SNN input shape** is `[B, T, C]` with `T = L = 256`, `C = 2`. Model output is spike-count logits `[B, 5]`.
- Tests must be **hermetic**: no network, no multi-GB downloads. Real ECG loading is tested by *writing a tiny wfdb fixture* into a temp dir. Actual dataset download has one integration test marked `@pytest.mark.slow` (skipped by default).

---

### Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/neurocardio/__init__.py`
- Create: `tests/test_config.py` (placeholder smoke, replaced in Task 1)
- Create: `configs/default.yaml`
- Create: `README.md`

- [ ] **Step 1: Initialize repo and env**

The working directory `C:\Users\valer\neuromorphic` is not yet a git repo. Run:

```bash
cd /c/Users/valer/neuromorphic
git init
uv venv --python 3.11
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "neurocardio"
version = "0.1.0"
description = "SNN arrhythmia detector on single-lead ECG (Phase 1 PoC)"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "snntorch>=0.9",
    "wfdb>=4.1",
    "numpy>=1.26",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5"]

[project.scripts]
neurocardio = "neurocardio.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/neurocardio"]

[tool.pytest.ini_options]
markers = ["slow: requires network / large download (deselected by default)"]
addopts = "-m 'not slow'"

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

- [ ] **Step 3: Create package init and config placeholders**

`src/neurocardio/__init__.py`:
```python
__version__ = "0.1.0"
```

`configs/default.yaml`:
```yaml
data:
  data_dir: data/mitdb
  fs: 360
  lead_index: 0
  window_before: 128
  window_after: 128
  bandpass_low: 0.5
  bandpass_high: 40.0
  filter_order: 4
encoder:
  kind: delta
  delta_threshold: 0.1
  rate_num_steps: 256
model:
  kind: snn
  hidden: 128
  beta: 0.9
  n_classes: 5
train:
  epochs: 20
  batch_size: 128
  lr: 0.001
  seed: 1337
  device: cpu
```

`README.md` (minimal — expanded in the final task):
```markdown
# neurocardio

SNN arrhythmia detector on single-lead ECG. Phase 1 proof-of-concept.

## Setup
    uv venv --python 3.11
    uv pip install -e ".[dev]"
    uv run pytest -q
```

`tests/test_config.py` (temporary smoke, replaced in Task 1):
```python
def test_placeholder():
    import neurocardio
    assert neurocardio.__version__ == "0.1.0"
```

- [ ] **Step 4: Install and verify the smoke test passes**

Run:
```bash
uv pip install -e ".[dev]"
uv run pytest -q
```
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/neurocardio/__init__.py configs/default.yaml README.md tests/test_config.py .gitignore
git commit -m "chore: scaffold neurocardio package"
```

(Create a `.gitignore` with `.venv/`, `__pycache__/`, `data/`, `*.egg-info/`, `runs/` before committing.)

---

### Task 1: Config module

**Files:**
- Create: `src/neurocardio/config.py`
- Test: `tests/test_config.py` (replace placeholder)

- [ ] **Step 1: Write the failing test**

Replace `tests/test_config.py`:
```python
from pathlib import Path
from neurocardio.config import Config, load_config


def test_defaults():
    cfg = Config()
    assert cfg.data.fs == 360
    assert cfg.data.window_before + cfg.data.window_after == 256
    assert cfg.model.n_classes == 5
    assert cfg.encoder.kind == "delta"


def test_yaml_override_preserves_untouched_defaults(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  hidden: 64\ntrain:\n  epochs: 3\n")
    cfg = load_config(p)
    assert cfg.model.hidden == 64          # overridden
    assert cfg.train.epochs == 3           # overridden
    assert cfg.model.beta == 0.9           # default preserved
    assert cfg.data.fs == 360              # default preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'Config'`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/config.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data/mitdb"
    fs: int = 360
    lead_index: int = 0
    window_before: int = 128
    window_after: int = 128
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    filter_order: int = 4


@dataclass
class EncoderConfig:
    kind: str = "delta"        # "delta" | "rate" | "none"
    delta_threshold: float = 0.1
    rate_num_steps: int = 256


@dataclass
class ModelConfig:
    kind: str = "snn"          # "snn" | "cnn" | "lstm"
    hidden: int = 128
    beta: float = 0.9
    n_classes: int = 5


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 1337
    device: str = "cpu"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        data=DataConfig(**raw.get("data", {})),
        encoder=EncoderConfig(**raw.get("encoder", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/config.py tests/test_config.py
git commit -m "feat: dataclass config with yaml override"
```

---

### Task 2: PhysioNet download wrappers

**Files:**
- Create: `src/neurocardio/data/__init__.py` (empty)
- Create: `src/neurocardio/data/download.py`
- Test: `tests/test_records.py` (the slow integration test lives here; see Task 3)

- [ ] **Step 1: Write the failing test**

Create `tests/test_download.py`:
```python
import inspect
from neurocardio.data import download


def test_download_functions_exist_and_signatures():
    assert callable(download.download_mitdb)
    assert callable(download.download_ptbdb)
    sig = inspect.signature(download.download_mitdb)
    assert "dest" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: neurocardio.data.download`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/data/__init__.py`: empty file.

`src/neurocardio/data/download.py`:
```python
from pathlib import Path

import wfdb


def download_mitdb(dest: str = "data/mitdb") -> Path:
    """Download the MIT-BIH Arrhythmia Database from PhysioNet."""
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(out))
    return out


def download_ptbdb(dest: str = "data/ptbdb") -> Path:
    """Download the PTB Diagnostic ECG Database (extension dataset)."""
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("ptbdb", str(out))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_download.py -v`
Expected: PASS.

Optionally, add a real download integration test (network, ~100 MB), marked slow so it is skipped by default:
```python
# append to tests/test_download.py
import pytest


@pytest.mark.slow
def test_download_mitdb_real(tmp_path):
    from neurocardio.data.download import download_mitdb
    out = download_mitdb(tmp_path / "mitdb")
    assert (out / "100.dat").exists()
```
Run it explicitly with: `uv run pytest -m slow tests/test_download.py -v` (requires internet).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/data/__init__.py src/neurocardio/data/download.py tests/test_download.py
git commit -m "feat: PhysioNet download wrappers (mitdb, ptbdb)"
```

---

### Task 3: Record loading + AAMI symbol map

**Files:**
- Create: `src/neurocardio/data/records.py`
- Test: `tests/test_records.py`

- [ ] **Step 1: Write the failing test**

`tests/test_records.py` — writes a tiny hermetic wfdb record + annotation, then reads it back:
```python
import numpy as np
import wfdb

from neurocardio.data.records import Record, load_record


def _write_fixture(dirpath):
    fs = 360
    sig = np.zeros((fs, 1), dtype=np.float64)
    sig[100, 0] = 1.0
    sig[250, 0] = 1.0
    wfdb.wrsamp(
        "rec1", fs=fs, units=["mV"], sig_name=["MLII"],
        p_signal=sig, write_dir=str(dirpath),
    )
    wfdb.wrann(
        "rec1", "atr",
        sample=np.array([100, 250]),
        symbol=["N", "V"],
        write_dir=str(dirpath),
    )


def test_load_record_returns_signal_fs_and_annotations(tmp_path):
    _write_fixture(tmp_path)
    rec = load_record(tmp_path, "rec1", lead_index=0)
    assert isinstance(rec, Record)
    assert rec.fs == 360
    assert rec.signal.shape == (360,)
    assert list(rec.ann_samples) == [100, 250]
    assert rec.ann_symbols == ["N", "V"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_records.py -v`
Expected: FAIL with `ModuleNotFoundError: neurocardio.data.records`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/data/records.py`:
```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb


@dataclass
class Record:
    record_id: str
    signal: np.ndarray      # 1-D single lead, shape [n_samples]
    fs: int
    ann_samples: np.ndarray  # int sample indices of beat annotations
    ann_symbols: list[str]   # wfdb beat symbols aligned to ann_samples


def load_record(record_dir, record_id: str, lead_index: int = 0) -> Record:
    base = str(Path(record_dir) / record_id)
    rec = wfdb.rdrecord(base)
    ann = wfdb.rdann(base, "atr")
    signal = np.asarray(rec.p_signal[:, lead_index], dtype=np.float64)
    return Record(
        record_id=record_id,
        signal=signal,
        fs=int(rec.fs),
        ann_samples=np.asarray(ann.sample, dtype=int),
        ann_symbols=list(ann.symbol),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_records.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/data/records.py tests/test_records.py
git commit -m "feat: load_record with wfdb + Record dataclass"
```

---

### Task 4: Preprocessing (bandpass + normalize)

**Files:**
- Create: `src/neurocardio/data/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write the failing test**

`tests/test_preprocess.py`:
```python
import numpy as np

from neurocardio.data.preprocess import bandpass_filter, normalize


def test_bandpass_removes_dc_offset():
    fs = 360
    t = np.arange(fs * 2) / fs
    clean = np.sin(2 * np.pi * 10 * t)      # 10 Hz, in-band
    with_dc = clean + 5.0                    # DC offset (0 Hz, out-of-band)
    out = bandpass_filter(with_dc, fs=fs, low=0.5, high=40.0, order=4)
    assert abs(out.mean()) < 0.05            # DC essentially removed


def test_bandpass_attenuates_out_of_band():
    fs = 360
    t = np.arange(fs * 2) / fs
    inband = np.sin(2 * np.pi * 10 * t)
    highfreq = np.sin(2 * np.pi * 120 * t)   # above 40 Hz cutoff
    out = bandpass_filter(inband + highfreq, fs=fs, low=0.5, high=40.0, order=4)
    # in-band amplitude largely preserved, high-freq suppressed
    assert out.std() < (inband + highfreq).std()


def test_normalize_zero_mean_unit_std():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = normalize(x)
    assert abs(out.mean()) < 1e-9
    assert abs(out.std() - 1.0) < 1e-6


def test_normalize_constant_signal_is_safe():
    out = normalize(np.ones(10))
    assert np.all(np.isfinite(out))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/data/preprocess.py`:
```python
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs: int, low: float = 0.5, high: float = 40.0,
                    order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, np.asarray(signal, dtype=np.float64))


def normalize(signal) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/data/preprocess.py tests/test_preprocess.py
git commit -m "feat: bandpass filter + normalize"
```

---

### Task 5: Beat segmentation + AAMI class mapping

**Files:**
- Create: `src/neurocardio/data/segment.py`
- Test: `tests/test_segment.py`

- [ ] **Step 1: Write the failing test**

`tests/test_segment.py`:
```python
import numpy as np

from neurocardio.data.segment import AAMI_CLASSES, segment_beats, symbol_to_aami


def test_aami_class_order_is_fixed():
    assert AAMI_CLASSES == ["N", "SVEB", "VEB", "F", "Q"]


def test_symbol_mapping():
    assert symbol_to_aami("N") == "N"
    assert symbol_to_aami("L") == "N"
    assert symbol_to_aami("A") == "SVEB"
    assert symbol_to_aami("V") == "VEB"
    assert symbol_to_aami("E") == "VEB"
    assert symbol_to_aami("F") == "F"
    assert symbol_to_aami("/") == "Q"
    assert symbol_to_aami("+") is None      # rhythm marker, not a beat


def test_segment_windows_and_labels():
    fs = 360
    signal = np.arange(2000, dtype=np.float64)
    ann_samples = np.array([500, 1000, 1500])
    ann_symbols = ["N", "V", "A"]
    beats, labels = segment_beats(
        signal, ann_samples, ann_symbols, window_before=128, window_after=128
    )
    assert beats.shape == (3, 256)
    # window is centered on the annotation sample
    assert beats[0, 128] == 500.0
    assert list(labels) == [0, 2, 1]        # N=0, VEB=2, SVEB=1


def test_segment_drops_edge_and_nonbeat_annotations():
    signal = np.zeros(1000, dtype=np.float64)
    ann_samples = np.array([10, 500, 995])       # 10 and 995 too close to edges
    ann_symbols = ["N", "+", "N"]                # 500 is a non-beat marker
    beats, labels = segment_beats(
        signal, ann_samples, ann_symbols, window_before=128, window_after=128
    )
    assert beats.shape[0] == 0                     # all dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_segment.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/data/segment.py`:
```python
import numpy as np

AAMI_CLASSES = ["N", "SVEB", "VEB", "F", "Q"]
_CLASS_INDEX = {c: i for i, c in enumerate(AAMI_CLASSES)}

# MIT-BIH beat symbol -> AAMI class (per EC57). Non-beat markers map to None.
_SYMBOL_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "SVEB", "a": "SVEB", "J": "SVEB", "S": "SVEB",
    "V": "VEB", "E": "VEB",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


def symbol_to_aami(symbol: str):
    return _SYMBOL_MAP.get(symbol)


def segment_beats(signal, ann_samples, ann_symbols, window_before: int = 128,
                  window_after: int = 128):
    n = len(signal)
    beats, labels = [], []
    for s, sym in zip(ann_samples, ann_symbols):
        cls = symbol_to_aami(sym)
        if cls is None:
            continue
        start, end = s - window_before, s + window_after
        if start < 0 or end > n:
            continue
        beats.append(signal[start:end])
        labels.append(_CLASS_INDEX[cls])
    if not beats:
        return (np.zeros((0, window_before + window_after), dtype=np.float64),
                np.zeros((0,), dtype=np.int64))
    return np.asarray(beats, dtype=np.float64), np.asarray(labels, dtype=np.int64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_segment.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/data/segment.py tests/test_segment.py
git commit -m "feat: AAMI beat segmentation and class mapping"
```

---

### Task 6: Inter-patient split (DS1/DS2)

**Files:**
- Create: `src/neurocardio/data/splits.py`
- Test: `tests/test_splits.py`

- [ ] **Step 1: Write the failing test**

`tests/test_splits.py`:
```python
import pytest

from neurocardio.data.splits import (
    DS1_RECORDS, DS2_RECORDS, PACED_RECORDS, get_split,
)


def test_ds1_ds2_sizes():
    assert len(DS1_RECORDS) == 22
    assert len(DS2_RECORDS) == 22


def test_ds1_ds2_disjoint():
    assert set(DS1_RECORDS).isdisjoint(set(DS2_RECORDS))


def test_paced_records_excluded_from_both():
    both = set(DS1_RECORDS) | set(DS2_RECORDS)
    assert both.isdisjoint(set(PACED_RECORDS))
    assert set(PACED_RECORDS) == {"102", "104", "107", "217"}


def test_get_split_names():
    assert get_split("train") == DS1_RECORDS
    assert get_split("test") == DS2_RECORDS
    with pytest.raises(ValueError):
        get_split("bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/data/splits.py`:
```python
# de Chazal et al. (2004) inter-patient split. Paced records (102, 104, 107,
# 217) are excluded per AAMI EC57. Keeping patients disjoint across train/test
# is the guardrail against beat-level leakage.
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]
PACED_RECORDS = ["102", "104", "107", "217"]


def get_split(name: str) -> list[str]:
    if name == "train":
        return DS1_RECORDS
    if name == "test":
        return DS2_RECORDS
    raise ValueError(f"unknown split: {name!r} (expected 'train' or 'test')")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_splits.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/data/splits.py tests/test_splits.py
git commit -m "feat: DS1/DS2 inter-patient split"
```

---

### Task 7: Torch dataset + build pipeline

**Files:**
- Create: `src/neurocardio/data/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:
```python
import numpy as np
import torch

from neurocardio.data.dataset import ECGBeatDataset


def test_dataset_returns_beat_and_label():
    beats = np.random.randn(5, 256).astype(np.float64)
    labels = np.array([0, 1, 2, 3, 4])
    ds = ECGBeatDataset(beats, labels)
    x, y = ds[2]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (256,)
    assert int(y) == 2
    assert len(ds) == 5


def test_dataset_applies_transform():
    beats = np.zeros((2, 4), dtype=np.float64)
    labels = np.array([0, 1])

    def to_two_channels(beat):
        return torch.zeros((beat.shape[0], 2), dtype=torch.float32)

    ds = ECGBeatDataset(beats, labels, transform=to_two_channels)
    x, _ = ds[0]
    assert x.shape == (4, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/data/dataset.py`:
```python
import numpy as np
import torch
from torch.utils.data import Dataset


class ECGBeatDataset(Dataset):
    """Holds pre-segmented beats [N, L] and integer AAMI labels [N].

    If `transform` is given it is applied to each beat (numpy [L]) and should
    return a tensor; otherwise the beat is returned as a float32 tensor [L].
    """

    def __init__(self, beats: np.ndarray, labels: np.ndarray, transform=None):
        assert len(beats) == len(labels)
        self.beats = np.asarray(beats, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        beat = self.beats[idx]
        y = int(self.labels[idx])
        if self.transform is not None:
            x = self.transform(beat)
        else:
            x = torch.from_numpy(beat)
        return x, y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/data/dataset.py tests/test_dataset.py
git commit -m "feat: ECGBeatDataset with optional transform"
```

---

### Task 8: Delta (level-crossing) spike encoder

**Files:**
- Create: `src/neurocardio/encoding/__init__.py` (empty)
- Create: `src/neurocardio/encoding/delta.py`
- Test: `tests/test_delta.py`

This is the neuromorphic heart of the pipeline: it mirrors what a hardware analog front-end + comparator would emit (up/down threshold crossings), so the SNN trains on the same representation the eventual chip produces.

- [ ] **Step 1: Write the failing test**

`tests/test_delta.py`:
```python
import numpy as np

from neurocardio.encoding.delta import delta_decode, delta_encode


def test_flat_signal_produces_no_spikes():
    sig = np.full(10, 0.3)
    spikes = delta_encode(sig, threshold=0.1)
    assert spikes.shape == (10, 2)
    assert spikes.sum() == 0.0


def test_step_up_produces_up_spikes():
    sig = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    spikes = delta_encode(sig, threshold=0.5)
    # ref=0 -> t2 crosses (up, ref=0.5) -> t3 crosses (up, ref=1.0) -> t4 none
    assert list(spikes[:, 0]) == [0.0, 0.0, 1.0, 1.0, 0.0]   # up channel
    assert spikes[:, 1].sum() == 0.0                          # no down spikes


def test_step_down_produces_down_spikes():
    sig = np.array([1.0, 1.0, 0.0, 0.0])
    spikes = delta_encode(sig, threshold=0.5)
    assert spikes[:, 1].sum() == 2.0        # two down crossings
    assert spikes[:, 0].sum() == 0.0


def test_reconstruction_error_bounded_by_threshold():
    rng = np.random.default_rng(0)
    # scale kept well below threshold so per-step change < threshold (no slope
    # overload): single-spike-per-step delta only tracks a slope of `threshold`
    # per timestep, so the error bound holds only when the signal is slew-limited.
    sig = np.cumsum(rng.standard_normal(500)) * 0.02
    thr = 0.1
    spikes = delta_encode(sig, threshold=thr)
    recon = delta_decode(spikes, threshold=thr, initial=sig[0])
    assert np.max(np.abs(recon - sig)) <= thr + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_delta.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/encoding/__init__.py`: empty file.

`src/neurocardio/encoding/delta.py`:
```python
import numpy as np


def delta_encode(signal, threshold: float) -> np.ndarray:
    """Level-crossing (delta) encoder. Emits at most one up or down spike per
    timestep when the signal has moved >= threshold from the running reference.
    Returns float32 array [L, 2]: column 0 = up spikes, column 1 = down spikes.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    up = np.zeros(n, dtype=np.float32)
    down = np.zeros(n, dtype=np.float32)
    ref = x[0]
    for t in range(1, n):
        if x[t] - ref >= threshold:
            up[t] = 1.0
            ref += threshold
        elif ref - x[t] >= threshold:
            down[t] = 1.0
            ref -= threshold
    return np.stack([up, down], axis=1)


def delta_decode(spikes, threshold: float, initial: float = 0.0) -> np.ndarray:
    """Inverse of delta_encode: cumulative reconstruction from up/down spikes."""
    steps = (spikes[:, 0] - spikes[:, 1]) * threshold
    return initial + np.cumsum(steps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_delta.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/encoding/__init__.py src/neurocardio/encoding/delta.py tests/test_delta.py
git commit -m "feat: delta level-crossing spike encoder"
```

---

### Task 9: Rate encoder (baseline)

**Files:**
- Create: `src/neurocardio/encoding/rate.py`
- Test: `tests/test_rate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rate.py`:
```python
import numpy as np

from neurocardio.encoding.rate import rate_encode


def test_rate_encode_shape_and_range():
    sig = np.linspace(0.0, 1.0, 8)
    spikes = rate_encode(sig, num_steps=50, seed=0)
    assert spikes.shape == (50, 8)
    assert set(np.unique(spikes)).issubset({0.0, 1.0})


def test_higher_value_fires_more_often():
    sig = np.array([0.05, 0.95])
    spikes = rate_encode(sig, num_steps=2000, seed=1)
    rate_low = spikes[:, 0].mean()
    rate_high = spikes[:, 1].mean()
    assert rate_high > rate_low
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/encoding/rate.py`:
```python
import numpy as np


def rate_encode(signal, num_steps: int, seed: int = 0) -> np.ndarray:
    """Poisson-style rate encoder. Min-max scales the signal to [0, 1] firing
    probabilities and samples spikes over `num_steps`. Returns [num_steps, L]."""
    x = np.asarray(signal, dtype=np.float64)
    lo, hi = x.min(), x.max()
    p = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    rng = np.random.default_rng(seed)
    draws = rng.random((num_steps, len(x)))
    return (draws < p).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/encoding/rate.py tests/test_rate.py
git commit -m "feat: rate encoder baseline"
```

---

### Task 10: SNN classifier (LIF + surrogate gradient)

**Files:**
- Create: `src/neurocardio/models/__init__.py` (empty)
- Create: `src/neurocardio/models/snn.py`
- Test: `tests/test_snn.py`

- [ ] **Step 1: Write the failing test**

`tests/test_snn.py`:
```python
import torch

from neurocardio.models.snn import SNNClassifier


def test_forward_output_shape():
    model = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    x = torch.randint(0, 2, (4, 256, 2)).float()   # [B, T, C]
    out = model(x)
    assert out.shape == (4, 5)


def test_forward_is_deterministic_with_fixed_seed():
    torch.manual_seed(0)
    m1 = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    torch.manual_seed(0)
    m2 = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    x = torch.ones(2, 32, 2)
    assert torch.allclose(m1(x), m2(x))


def test_gradients_flow():
    model = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    x = torch.randint(0, 2, (3, 64, 2)).float()
    out = model(x)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_snn.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/models/__init__.py`: empty file.

`src/neurocardio/models/snn.py`:
```python
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate


class SNNClassifier(nn.Module):
    """Two-layer feedforward LIF network trained with surrogate gradients.
    Input x: [B, T, C] spike trains. Output: spike-count logits [B, n_classes]."""

    def __init__(self, in_features: int = 2, hidden: int = 128,
                 n_classes: int = 5, beta: float = 0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(in_features, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        mem1 = self.lif1.reset_mem()   # snntorch 1.0: init_leaky() is deprecated
        mem2 = self.lif2.reset_mem()
        out_sum = torch.zeros(b, self.fc2.out_features, device=x.device)
        for step in range(t):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            # Read out the output-layer MEMBRANE POTENTIAL, not its spike count.
            # A spike-count readout has a dead-neuron trap: loss is minimised by
            # the network going silent (all-zero logits -> uniform softmax ->
            # ln(n_classes)) and the surrogate gradient is too weak to escape.
            # Integrating the continuous membrane gives an always-nonzero
            # gradient so the classifier trains. The hidden layer still spikes
            # (that is where the sparse-compute/energy benefit lives).
            out_sum = out_sum + mem2
        return out_sum
```

> **Discovered during integration (Task 12):** the original spike-count readout (`out_sum + spk2`) does not train — even on dense, cleanly separable input the SNN collapses to `ln(n_classes)` loss and chance accuracy. The membrane-potential readout above fixes it (verified: loss → 0, accuracy → 100% on a separable toy). `snntorch` 1.0.0 also renamed `init_leaky()` → `reset_mem()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_snn.py -v`
Expected: PASS (3 passed).

> Note: `test_gradients_flow` may occasionally see all-zero spike outputs for a random init (no spikes → no gradient through the surrogate). If it flakes, seed the test with `torch.manual_seed(0)` at the top and scale the input up (`x = torch.ones(...)`). The delta-encoded real data is dense enough that this is not an issue in training.

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/models/__init__.py src/neurocardio/models/snn.py tests/test_snn.py
git commit -m "feat: LIF SNN classifier with surrogate gradients"
```

---

### Task 11: CNN and LSTM baselines

**Files:**
- Create: `src/neurocardio/models/baselines.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing test**

`tests/test_baselines.py`:
```python
import torch

from neurocardio.models.baselines import CNN1D, LSTMClassifier


def test_cnn_output_shape():
    model = CNN1D(n_classes=5)
    x = torch.randn(4, 256)          # [B, L] raw beat
    out = model(x)
    assert out.shape == (4, 5)


def test_lstm_output_shape():
    model = LSTMClassifier(n_classes=5, hidden=16)
    x = torch.randn(4, 256)          # [B, L] raw beat
    out = model(x)
    assert out.shape == (4, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/models/baselines.py`:
```python
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1-D CNN baseline on raw beats [B, L]."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.unsqueeze(1))       # [B, 1, L]
        return self.head(h.squeeze(-1))


class LSTMClassifier(nn.Module):
    """LSTM baseline on raw beats [B, L] (treated as length-L, 1-feature seq)."""

    def __init__(self, n_classes: int = 5, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))   # [B, L, 1]
        return self.head(out[:, -1, :])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/models/baselines.py tests/test_baselines.py
git commit -m "feat: CNN1D and LSTM baselines"
```

---

### Task 12: Training loop + seeding

**Files:**
- Create: `src/neurocardio/train/__init__.py` (empty)
- Create: `src/neurocardio/train/loop.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

The canonical TDD test for a training loop is "can it overfit a tiny batch." If it can drive loss down and hit high accuracy on 8 samples, the wiring (forward, loss, backward, step) is correct.

`tests/test_train.py`:
```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from neurocardio.data.dataset import ECGBeatDataset
from neurocardio.models.baselines import CNN1D
from neurocardio.train.loop import set_seed, train


def test_set_seed_is_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_overfits_tiny_batch():
    # Task 12 tests the TRAINING LOOP (forward -> loss -> backward -> step). Use
    # the CNN1D baseline on raw, linearly separable beats: a conventional model
    # that overfits 8 samples monotonically and deterministically, so the loop's
    # mechanics are what is under test, not SNN spike-learning dynamics. The
    # SNN + delta + train integration (and that it learns) is exercised in the
    # end-to-end smoke test (Task 18).
    set_seed(0)
    beats = np.zeros((8, 64), dtype=np.float32)
    labels = np.zeros(8, dtype=np.int64)
    for i in range(8):
        cls = i % 2
        start = 10 + cls * 30
        beats[i, start : start + 10] = 1.0  # early bump vs late bump
        labels[i] = cls

    ds = ECGBeatDataset(beats, labels)  # raw beats, no spike encoding
    loader = DataLoader(ds, batch_size=8)
    model = CNN1D(n_classes=2)
    history = train(model, loader, loader, epochs=100, lr=0.01, device="cpu")
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_acc"][-1] >= 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/train/__init__.py`: empty file.

`src/neurocardio/train/loop.py`:
```python
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += len(y)
    return correct / max(total, 1)


def train(model, train_loader, val_loader, epochs: int = 20, lr: float = 1e-3,
          device: str = "cpu", class_weights=None) -> dict:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weight = None if class_weights is None else torch.tensor(
        class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    history = {"train_loss": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss) * len(y)
        history["train_loss"].append(epoch_loss / len(train_loader.dataset))
        history["val_acc"].append(_accuracy(model, val_loader, device))
    return history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (2 passed). If `test_overfits_tiny_batch` doesn't reach 0.99, raise `epochs` to 100 or `lr` to 0.05 — it must overfit 8 samples; if it can't, the wiring is broken.

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/train/__init__.py src/neurocardio/train/loop.py tests/test_train.py
git commit -m "feat: training loop with seeding and class weights"
```

---

### Task 13: Metrics (confusion matrix + AAMI sensitivity/PPV)

**Files:**
- Create: `src/neurocardio/eval/__init__.py` (empty)
- Create: `src/neurocardio/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import numpy as np

from neurocardio.eval.metrics import aami_metrics, confusion


def test_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0])
    cm = confusion(y_true, y_pred, n_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1     # one N->N, one N->SVEB
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1 and cm[2, 0] == 1
    assert cm.sum() == 5


def test_aami_metrics_known_values():
    # Perfect classifier over 3 classes -> sensitivity=PPV=1, acc=1
    cm = np.array([[10, 0, 0], [0, 5, 0], [0, 0, 2]])
    m = aami_metrics(cm, classes=["N", "SVEB", "VEB"])
    assert abs(m["overall_accuracy"] - 1.0) < 1e-9
    assert abs(m["per_class"]["VEB"]["sensitivity"] - 1.0) < 1e-9
    assert abs(m["per_class"]["VEB"]["ppv"] - 1.0) < 1e-9


def test_aami_metrics_partial():
    # VEB: 8 true; 6 correctly caught (sens=0.75); of 7 predicted VEB, 6 right (ppv≈0.857)
    cm = np.array([
        [90, 0, 1],     # N row
        [0, 10, 0],     # SVEB row
        [2, 0, 6],      # VEB row: 6 TP, 2 FN
    ])
    m = aami_metrics(cm, classes=["N", "SVEB", "VEB"])
    veb = m["per_class"]["VEB"]
    assert abs(veb["sensitivity"] - 6 / 8) < 1e-9
    assert abs(veb["ppv"] - 6 / 7) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/eval/__init__.py`: empty file.

`src/neurocardio/eval/metrics.py`:
```python
import numpy as np


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def aami_metrics(cm: np.ndarray, classes: list[str]) -> dict:
    """Per-class sensitivity (recall) and positive predictivity (precision),
    plus overall accuracy. Rows = true, columns = predicted."""
    total = cm.sum()
    per_class = {}
    for i, name in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class[name] = {
            "sensitivity": float(sens),
            "ppv": float(ppv),
            "support": int(cm[i, :].sum()),
        }
    return {
        "overall_accuracy": float(np.trace(cm) / total) if total else 0.0,
        "per_class": per_class,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/eval/__init__.py src/neurocardio/eval/metrics.py tests/test_metrics.py
git commit -m "feat: confusion matrix and AAMI sensitivity/PPV metrics"
```

---

### Task 14: Evaluate over a loader

**Files:**
- Create: `src/neurocardio/eval/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate.py`:
```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from neurocardio.data.dataset import ECGBeatDataset
from neurocardio.eval.evaluate import evaluate


class _ConstModel(torch.nn.Module):
    """Always predicts class 0 (for a deterministic test)."""

    def forward(self, x):
        b = x.shape[0]
        out = torch.zeros(b, 3)
        out[:, 0] = 1.0
        return out


def test_evaluate_returns_cm_and_metrics():
    beats = np.zeros((6, 8), dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2])
    ds = ECGBeatDataset(beats, labels)
    loader = DataLoader(ds, batch_size=3)
    result = evaluate(_ConstModel(), loader, classes=["N", "SVEB", "VEB"])
    assert result["confusion"].shape == (3, 3)
    # all predicted as N -> N sensitivity 1.0, VEB sensitivity 0.0
    assert result["metrics"]["per_class"]["N"]["sensitivity"] == 1.0
    assert result["metrics"]["per_class"]["VEB"]["sensitivity"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/eval/evaluate.py`:
```python
import numpy as np
import torch

from neurocardio.eval.metrics import aami_metrics, confusion


def evaluate(model, loader, classes, device: str = "cpu") -> dict:
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(np.asarray(y).tolist())
    cm = confusion(y_true, y_pred, n_classes=len(classes))
    return {"confusion": cm, "metrics": aami_metrics(cm, classes)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/eval/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluate model over a DataLoader"
```

---

### Task 15: Energy proxy (synaptic operations)

**Files:**
- Create: `src/neurocardio/deploy/__init__.py` (empty)
- Create: `src/neurocardio/deploy/energy.py`
- Test: `tests/test_energy.py`

The neuromorphic value proposition is energy. We cannot measure joules in simulation, but we can count **synaptic operations (SynOps)** — the standard neuromorphic proxy — and total spikes. A layer's SynOps = (number of presynaptic spikes) × (postsynaptic fan-out). This makes the "low-power" claim a number, not an adjective, and it is the metric that translates to a Loihi/Akida power estimate later.

- [ ] **Step 1: Write the failing test**

`tests/test_energy.py`:
```python
import torch

from neurocardio.deploy.energy import spike_stats, synaptic_operations
from neurocardio.models.snn import SNNClassifier


def test_synaptic_operations_pure_function():
    # 5 presynaptic spikes into a layer with fan-out 10 = 50 SynOps
    spike_counts = {"fc1": 5, "fc2": 3}
    fan_out = {"fc1": 10, "fc2": 4}
    assert synaptic_operations(spike_counts, fan_out) == 5 * 10 + 3 * 4


def test_spike_stats_runs_model_and_counts():
    torch.manual_seed(0)
    model = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    x = torch.ones(1, 64, 2)
    stats = spike_stats(model, x)
    assert stats["total_spikes"] >= 0
    assert "synops" in stats
    assert stats["synops"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_energy.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/deploy/__init__.py`: empty file.

`src/neurocardio/deploy/energy.py`:
```python
import torch

from neurocardio.models.snn import SNNClassifier


def synaptic_operations(spike_counts: dict, fan_out: dict) -> int:
    """Pure SynOps proxy: sum over layers of presynaptic_spikes * fan_out."""
    return int(sum(spike_counts[k] * fan_out[k] for k in spike_counts))


def spike_stats(model: SNNClassifier, x: torch.Tensor) -> dict:
    """Run one forward pass, counting input spikes to each Linear layer and the
    resulting SynOps. Assumes the SNNClassifier fc1/fc2/lif structure."""
    model.eval()
    b, t, _ = x.shape
    mem1 = model.lif1.reset_mem()
    mem2 = model.lif2.reset_mem()
    fc1_in_spikes = 0
    fc2_in_spikes = 0
    total_spikes = 0
    with torch.no_grad():
        for step in range(t):
            inp = x[:, step, :]
            fc1_in_spikes += int(inp.sum())
            spk1, mem1 = model.lif1(model.fc1(inp), mem1)
            fc2_in_spikes += int(spk1.sum())
            spk2, mem2 = model.lif2(model.fc2(spk1), mem2)
            total_spikes += int(spk1.sum()) + int(spk2.sum())
    counts = {"fc1": fc1_in_spikes, "fc2": fc2_in_spikes}
    fan_out = {"fc1": model.fc1.out_features, "fc2": model.fc2.out_features}
    return {
        "total_spikes": total_spikes,
        "synops": synaptic_operations(counts, fan_out),
        "spike_counts": counts,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_energy.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/deploy/__init__.py src/neurocardio/deploy/energy.py tests/test_energy.py
git commit -m "feat: synaptic-operations energy proxy"
```

---

### Task 16: Streaming R-peak detector + online detector

**Files:**
- Create: `src/neurocardio/stream/__init__.py` (empty)
- Create: `src/neurocardio/stream/qrs.py`
- Create: `src/neurocardio/stream/detector.py`
- Test: `tests/test_qrs.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: Write the failing test (QRS)**

`tests/test_qrs.py`:
```python
import numpy as np

from neurocardio.stream.qrs import find_r_peaks


def test_finds_peaks_near_known_positions():
    fs = 360
    n = fs * 4
    sig = np.zeros(n)
    true_peaks = [300, 700, 1100, 1500]
    for p in true_peaks:
        # sharp QRS-like deflection
        sig[p - 2:p + 3] += np.array([0.2, 0.6, 1.0, 0.6, 0.2])
    peaks = find_r_peaks(sig, fs=fs)
    assert len(peaks) == len(true_peaks)
    for detected, expected in zip(sorted(peaks), true_peaks):
        assert abs(int(detected) - expected) <= 5


def test_refractory_prevents_double_counting():
    fs = 360
    sig = np.zeros(fs)
    sig[100:103] = [0.5, 1.0, 0.5]
    sig[105:108] = [0.5, 1.0, 0.5]   # 5 samples later, within refractory
    peaks = find_r_peaks(sig, fs=fs, refractory_s=0.2)
    assert len(peaks) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qrs.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation (QRS)**

`src/neurocardio/stream/__init__.py`: empty file.

`src/neurocardio/stream/qrs.py`:
```python
import numpy as np
from scipy.signal import find_peaks


def find_r_peaks(signal, fs: int, refractory_s: float = 0.2,
                 threshold_frac: float = 0.3) -> np.ndarray:
    """Lightweight Pan-Tompkins-style detector: derivative -> square -> moving
    integration -> peak pick with a refractory period. For the online path only;
    training uses ground-truth annotations."""
    x = np.asarray(signal, dtype=np.float64)
    diff = np.diff(x, prepend=x[0])
    squared = diff ** 2
    win = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    peak_max = integrated.max()
    if peak_max <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(
        integrated,
        height=threshold_frac * peak_max,
        distance=max(1, int(refractory_s * fs)),
    )
    return peaks
```

- [ ] **Step 4: Run test to verify it passes (QRS)**

Run: `uv run pytest tests/test_qrs.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the failing test (detector)**

`tests/test_detector.py`:
```python
import numpy as np
import torch

from neurocardio.config import Config
from neurocardio.stream.detector import Detection, StreamDetector


class _Const2Model(torch.nn.Module):
    def forward(self, x):
        b = x.shape[0]
        out = torch.zeros(b, 5)
        out[:, 2] = 1.0        # always "VEB"
        return out


def test_stream_detector_emits_one_detection_per_beat():
    cfg = Config()
    fs = cfg.data.fs
    n = fs * 4
    sig = np.zeros(n)
    true_peaks = [400, 800, 1200, 1600]
    for p in true_peaks:
        sig[p - 2:p + 3] += np.array([0.2, 0.6, 1.0, 0.6, 0.2])
    det = StreamDetector(_Const2Model(), cfg)
    detections = det.process(sig)
    assert len(detections) == len(true_peaks)
    assert all(isinstance(d, Detection) for d in detections)
    assert all(d.label == "VEB" for d in detections)
    # detections are ordered by sample index
    idxs = [d.sample_index for d in detections]
    assert idxs == sorted(idxs)
```

- [ ] **Step 6: Run test to verify it fails (detector)**

Run: `uv run pytest tests/test_detector.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 7: Write minimal implementation (detector)**

`src/neurocardio/stream/detector.py`:
```python
from dataclasses import dataclass

import numpy as np
import torch

from neurocardio.config import Config
from neurocardio.data.preprocess import bandpass_filter, normalize
from neurocardio.data.segment import AAMI_CLASSES
from neurocardio.encoding.delta import delta_encode
from neurocardio.stream.qrs import find_r_peaks


@dataclass
class Detection:
    sample_index: int
    label: str
    score: float


class StreamDetector:
    """Runs a trained model over a continuous single-lead signal, emitting one
    Detection per detected beat. Mirrors the on-patch anomaly-logging path."""

    def __init__(self, model, config: Config):
        self.model = model
        self.cfg = config

    def _window(self, signal, peak) -> np.ndarray | None:
        before = self.cfg.data.window_before
        after = self.cfg.data.window_after
        start, end = peak - before, peak + after
        if start < 0 or end > len(signal):
            return None
        return signal[start:end]

    def process(self, signal) -> list[Detection]:
        sig = normalize(bandpass_filter(
            signal, fs=self.cfg.data.fs,
            low=self.cfg.data.bandpass_low, high=self.cfg.data.bandpass_high,
            order=self.cfg.data.filter_order,
        ))
        peaks = find_r_peaks(sig, fs=self.cfg.data.fs)
        self.model.eval()
        detections: list[Detection] = []
        with torch.no_grad():
            for p in sorted(peaks):
                beat = self._window(sig, int(p))
                if beat is None:
                    continue
                spikes = delta_encode(beat, self.cfg.encoder.delta_threshold)
                x = torch.from_numpy(spikes).unsqueeze(0)   # [1, T, 2]
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                idx = int(probs.argmax(dim=1))
                detections.append(Detection(
                    sample_index=int(p),
                    label=AAMI_CLASSES[idx],
                    score=float(probs[0, idx]),
                ))
        return detections
```

- [ ] **Step 8: Run test to verify it passes (detector)**

Run: `uv run pytest tests/test_detector.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/neurocardio/stream/ tests/test_qrs.py tests/test_detector.py
git commit -m "feat: streaming R-peak detector and online StreamDetector"
```

---

### Task 17: Dataset builder (wire records → beats → dataset)

**Files:**
- Modify: `src/neurocardio/data/dataset.py` (add `build_split`)
- Test: `tests/test_dataset.py` (add a build test using a wfdb fixture)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset.py`:
```python
import numpy as np
import wfdb

from neurocardio.config import Config
from neurocardio.data.dataset import build_split


def _write_two_beat_record(dirpath, record_id):
    fs = 360
    n = fs * 3
    sig = np.zeros((n, 1))
    for center in (500, 1500):
        sig[center - 2:center + 3, 0] = [0.2, 0.6, 1.0, 0.6, 0.2]
    wfdb.wrsamp(record_id, fs=fs, units=["mV"], sig_name=["MLII"],
                p_signal=sig, write_dir=str(dirpath))
    wfdb.wrann(record_id, "atr", sample=np.array([500, 1500]),
               symbol=["N", "V"], write_dir=str(dirpath))


def test_build_split_produces_beats_and_labels(tmp_path):
    _write_two_beat_record(tmp_path, "900")
    cfg = Config()
    cfg.data.data_dir = str(tmp_path)
    beats, labels = build_split(cfg, record_ids=["900"])
    assert beats.shape[1] == 256
    assert beats.shape[0] == 2
    assert set(labels.tolist()) == {0, 2}      # N and VEB
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py::test_build_split_produces_beats_and_labels -v`
Expected: FAIL with `ImportError: cannot import name 'build_split'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/neurocardio/data/dataset.py`:
```python
from neurocardio.config import Config
from neurocardio.data.preprocess import bandpass_filter, normalize
from neurocardio.data.records import load_record
from neurocardio.data.segment import segment_beats


def build_split(config: Config, record_ids):
    """Load each record, preprocess, segment into AAMI beats, and concatenate."""
    all_beats, all_labels = [], []
    for rid in record_ids:
        rec = load_record(config.data.data_dir, rid, config.data.lead_index)
        sig = normalize(bandpass_filter(
            rec.signal, fs=rec.fs,
            low=config.data.bandpass_low, high=config.data.bandpass_high,
            order=config.data.filter_order,
        ))
        beats, labels = segment_beats(
            sig, rec.ann_samples, rec.ann_symbols,
            window_before=config.data.window_before,
            window_after=config.data.window_after,
        )
        if len(beats):
            all_beats.append(beats)
            all_labels.append(labels)
    if not all_beats:
        return (np.zeros((0, config.data.window_before + config.data.window_after)),
                np.zeros((0,), dtype=np.int64))
    return np.concatenate(all_beats), np.concatenate(all_labels)
```

(Move the `import numpy as np` to the top of the file if not already present.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS (all dataset tests).

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/data/dataset.py tests/test_dataset.py
git commit -m "feat: build_split assembles beats from records"
```

---

### Task 18: CLI + end-to-end smoke on synthetic data

**Files:**
- Create: `src/neurocardio/cli.py`
- Test: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write the failing test**

The smoke test runs the *entire* pipeline (build → encode → train briefly → evaluate → energy) on a synthetic in-memory dataset, asserting it produces a metrics dict and a SynOps number. This is the integration guard that proves the parts fit together.

`tests/test_cli_smoke.py`:
```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from neurocardio.data.dataset import ECGBeatDataset
from neurocardio.data.segment import AAMI_CLASSES
from neurocardio.deploy.energy import spike_stats
from neurocardio.encoding.delta import delta_encode
from neurocardio.eval.evaluate import evaluate
from neurocardio.models.snn import SNNClassifier
from neurocardio.train.loop import set_seed, train


def test_end_to_end_synthetic_pipeline():
    set_seed(0)
    # synthetic separable beats: class 0 flat, class 2 has a step (VEB stand-in)
    n = 40
    beats = np.zeros((n, 256), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if i % 2 == 0:
            beats[i, 120:136] = 1.0
            labels[i] = 0
        else:
            beats[i, 120:200] = 1.0
            labels[i] = 2

    def transform(beat):
        return torch.from_numpy(delta_encode(beat, threshold=0.5))

    ds = ECGBeatDataset(beats, labels, transform=transform)
    loader = DataLoader(ds, batch_size=10, shuffle=True)
    model = SNNClassifier(in_features=2, hidden=32, n_classes=5)
    history = train(model, loader, loader, epochs=30, lr=0.02)
    # the SNN + delta + train integration must actually LEARN (loss decreases) --
    # this is the guard that the membrane-potential readout works end-to-end
    assert history["train_loss"][-1] < history["train_loss"][0]
    result = evaluate(model, loader, classes=AAMI_CLASSES)
    assert 0.0 <= result["metrics"]["overall_accuracy"] <= 1.0
    stats = spike_stats(model, next(iter(loader))[0][:1])
    assert stats["synops"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_smoke.py -v`
Expected: FAIL only if imports are wrong; if all prior tasks passed it may already pass. If it fails on `neurocardio.cli` import, that's expected before Step 3.

- [ ] **Step 3: Write minimal implementation**

`src/neurocardio/cli.py`:
```python
import argparse

import torch
from torch.utils.data import DataLoader

from neurocardio.config import load_config
from neurocardio.data.dataset import ECGBeatDataset, build_split
from neurocardio.data.segment import AAMI_CLASSES
from neurocardio.data.splits import get_split
from neurocardio.deploy.energy import spike_stats
from neurocardio.encoding.delta import delta_encode
from neurocardio.eval.evaluate import evaluate
from neurocardio.models.baselines import CNN1D, LSTMClassifier
from neurocardio.models.snn import SNNClassifier
from neurocardio.train.loop import set_seed, train


def _make_model(cfg):
    if cfg.model.kind == "snn":
        return SNNClassifier(in_features=2, hidden=cfg.model.hidden,
                             n_classes=cfg.model.n_classes, beta=cfg.model.beta)
    if cfg.model.kind == "cnn":
        return CNN1D(n_classes=cfg.model.n_classes)
    if cfg.model.kind == "lstm":
        return LSTMClassifier(n_classes=cfg.model.n_classes, hidden=cfg.model.hidden)
    raise ValueError(f"unknown model kind: {cfg.model.kind}")


def _make_dataset(cfg, record_ids):
    beats, labels = build_split(cfg, record_ids)
    if cfg.model.kind == "snn" and cfg.encoder.kind == "delta":
        thr = cfg.encoder.delta_threshold

        def transform(beat):
            return torch.from_numpy(delta_encode(beat, thr))
        return ECGBeatDataset(beats, labels, transform=transform)
    return ECGBeatDataset(beats, labels)


def cmd_download(args):
    from neurocardio.data.download import download_mitdb
    download_mitdb(args.dest)


def cmd_train(args):
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    train_ds = _make_dataset(cfg, get_split("train"))
    test_ds = _make_dataset(cfg, get_split("test"))
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size)
    model = _make_model(cfg)
    train(model, train_loader, test_loader, epochs=cfg.train.epochs,
          lr=cfg.train.lr, device=cfg.train.device)
    result = evaluate(model, test_loader, classes=AAMI_CLASSES, device=cfg.train.device)
    print("Inter-patient (DS2) metrics:", result["metrics"])
    torch.save(model.state_dict(), args.out)


def cmd_evaluate(args):
    cfg = load_config(args.config)
    test_ds = _make_dataset(cfg, get_split("test"))
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size)
    model = _make_model(cfg)
    model.load_state_dict(torch.load(args.weights))
    result = evaluate(model, loader, classes=AAMI_CLASSES)
    print(result["metrics"])
    if cfg.model.kind == "snn":
        print("Energy proxy:", spike_stats(model, next(iter(loader))[0][:1]))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="neurocardio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download")
    p_dl.add_argument("--dest", default="data/mitdb")
    p_dl.set_defaults(func=cmd_download)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--config", default="configs/default.yaml")
    p_tr.add_argument("--out", default="runs/model.pt")
    p_tr.set_defaults(func=cmd_train)

    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--config", default="configs/default.yaml")
    p_ev.add_argument("--weights", required=True)
    p_ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_smoke.py -v && uv run pytest -q`
Expected: smoke test PASSES; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/neurocardio/cli.py tests/test_cli_smoke.py
git commit -m "feat: CLI (download/train/evaluate) + end-to-end smoke test"
```

---

### Task 19: Real-data experiment run + results table in README

**Files:**
- Modify: `README.md`
- Create: `configs/snn.yaml`, `configs/cnn.yaml`, `configs/lstm.yaml`

This task produces the actual Phase-1 deliverable: the honest inter-patient results table comparing SNN vs CNN vs LSTM, with the energy proxy. It requires the real MIT-BIH download, so it is a manual/experimental task rather than a unit test.

- [ ] **Step 1: Download the data (one-time, ~100 MB)**

Run:
```bash
uv run neurocardio download --dest data/mitdb
```
Expected: `data/mitdb/100.dat` ... `234.dat` present.

- [ ] **Step 2: Create per-model configs**

`configs/snn.yaml`: copy `default.yaml`, keep `model.kind: snn`, `encoder.kind: delta`.
`configs/cnn.yaml`: copy `default.yaml`, set `model.kind: cnn`, `encoder.kind: none`.
`configs/lstm.yaml`: copy `default.yaml`, set `model.kind: lstm`, `encoder.kind: none`.

(For all three keep `train.seed: 1337` so runs are reproducible.)

- [ ] **Step 3: Train each model on DS1, evaluate on DS2**

Run:
```bash
uv run neurocardio train --config configs/snn.yaml  --out runs/snn.pt
uv run neurocardio train --config configs/cnn.yaml  --out runs/cnn.pt
uv run neurocardio train --config configs/lstm.yaml --out runs/lstm.pt
```
Record the printed DS2 metrics (VEB/SVEB sensitivity + PPV, overall accuracy) and, for the SNN, the SynOps from `evaluate`.

- [ ] **Step 4: Add class weighting if VEB/SVEB sensitivity is poor**

If VEB sensitivity is low (expected under heavy N-class imbalance), pass inverse-frequency `class_weights` into `train()` via a config field or a quick code edit, retrain the SNN, and record the improved numbers. Document what you changed. This is a genuine iteration point — the plan's job is to make the number visible and improvable, not to guarantee a specific value.

- [ ] **Step 5: Write the results table into README and commit**

Fill this table in `README.md` with your real numbers:
```markdown
## Phase 1 results (MIT-BIH, inter-patient DS1→DS2)

| Model | Params | VEB Sens | VEB PPV | SVEB Sens | SVEB PPV | Overall Acc | SynOps/beat |
|-------|--------|----------|---------|-----------|----------|-------------|-------------|
| SNN (delta) |  |  |  |  |  |  |  |
| CNN1D       |  |  |  |  |  |  | n/a |
| LSTM        |  |  |  |  |  |  | n/a |

Split: de Chazal DS1 train / DS2 test. Paced records excluded. Metrics per AAMI EC57.
```

```bash
git add README.md configs/snn.yaml configs/cnn.yaml configs/lstm.yaml
git commit -m "docs: phase-1 inter-patient results (SNN vs CNN/LSTM)"
```

- [ ] **Step 6: Final full-suite verification**

Run:
```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```
Expected: lint clean, all tests pass. This is the Phase-1 software core, done.

---

## Self-Review (author's pass against the spec)

**Spec coverage:**
- Spec step 1 (clinical problem selection) → resolved to arrhythmia/ECG; MIT-BIH primary (Tasks 2–7), PTB available via `download_ptbdb` (Task 2) and flagged as extension.
- Spec step 2 (AFE + spike encoder → SNN on low-power target, log anomalies) → delta encoder mirrors AFE comparator (Task 8); LIF SNN (Task 10); energy proxy for the low-power claim (Task 15); streaming anomaly logging (Task 16). Physical low-power chip is Phase-2 roadmap below (software-first per your decision).
- Spec step 3 (retrospective study on public datasets, CNN/LSTM baselines) → Tasks 11, 13, 14, 19; inter-patient methodology (Task 6) makes it credible. Retrospective *hospital* data is Phase-3 roadmap.
- Spec step 4 (regulatory: IEC 62304, MDR Class IIa, AI Act) → roadmap Phase 4 below; the codebase already lays groundwork (deterministic seeds, config-as-record, test suite = verification evidence).
- Spec step 5 (pilots) → roadmap Phase 5.

**Placeholder scan:** every code step contains complete, runnable code; every run step names an exact command and expected result. No TBDs.

**Type consistency:** `AAMI_CLASSES`, `Record`, `Config`/sub-configs, `delta_encode([L,2])`, `SNNClassifier([B,T,C]→[B,5])`, `build_split`, `evaluate`, `spike_stats`, `StreamDetector`/`Detection` names are used identically across Tasks 5–19.

---

## Phase 2+ Program Roadmap (context, not TDD tasks)

The software core above is the only part that is honestly "code you write and test today." The rest of the startup is program work — engineering procurement, clinical operations, and regulatory drafting — with different gates and deliverables. Sequenced so each phase only starts once the prior gate is cleared, to conserve capital.

### Phase 2 — Hardware realization (gated on Phase-1 SNN clearing a sensitivity bar, e.g. VEB Sens ≥ 0.90 inter-patient)
- **AFE selection.** Evaluate a low-power ECG analog front-end (e.g. TI ADS129x / Maxim MAX30003) feeding a level-crossing ADC or a firmware delta encoder — the hardware analogue of `delta_encode`.
- **Compute target trade study.** Benchmark the trained SNN on (a) an ARM Cortex-M MCU (quantized, e.g. via CMSIS-NN / a spiking kernel), (b) BrainChip Akida, (c) Intel Loihi 2. Use the Task-15 SynOps figure to seed power estimates before buying boards.
- **Deployment backend.** Extend `deploy/` with a concrete backend that exports the trained weights to the chosen target and reproduces the DS2 metrics on-device (bit-accuracy check vs the PyTorch model).
- **Patch form factor.** Adhesive single-lead electrode + AFE + MCU + BLE, battery-life bench measurement vs. the software power estimate.
- **Gate:** on-device DS2 metrics within tolerance of simulation; measured power supports a multi-day patch.

### Phase 3 — Retrospective clinical validation
- **Ethics / data governance.** Research ethics approval and a GDPR-compliant data processing agreement with a partner hospital; document data provenance (this becomes the AI Act training-data record).
- **Retrospective study.** Re-train/validate on de-identified hospital Holter/telemetry data; compare against the existing device's outputs and cardiologist over-reads. Report the same AAMI metrics plus a prospective-style alarm-burden analysis.
- **Dataset/versioning discipline.** Every model tied to a config hash + data snapshot (the Task-1 config and seeding make this reproducible) — the traceability a notified body expects.
- **Gate:** performance holds on real hospital data; false-alarm rate acceptable to clinicians.

### Phase 4 — Regulatory & quality groundwork ("more compliant than required, early")
- **Classification.** Position as MDR Class IIa medical device software (MDSW); confirm intended use (detection/early-warning, not diagnosis) and the corresponding claims.
- **IEC 62304 lifecycle.** Stand up the software safety classification, SDLC, risk management (ISO 14971), and a traceability matrix. The existing test suite, commit history, and config records are the first artifacts of this file.
- **AI Act readiness.** Build the technical documentation set (data governance, risk management, transparency, human oversight, robustness/accuracy) even ahead of full obligation — the differentiator you flagged.
- **QMS.** ISO 13485 quality management system scoped to software.
- **Gate:** a defensible technical file and a pre-submission meeting with a notified body / competent authority.

### Phase 5 — Prospective pilots
- **Site selection.** 1–2 epilepsy-adjacent cardiology or arrhythmia clinics; small prospective cohort.
- **Endpoints.** Earlier detection and/or lower false-alarm burden and/or battery life vs. incumbent wearables — the metrics your simulation already instruments.
- **Evidence loop.** Pilot data feeds back into training and the clinical evidence dossier; results become the fundraising and reimbursement narrative.
- **Gate:** prospective evidence supporting a regulatory submission and a Series-A story.

### Cross-cutting risks to watch
- **Inter-patient generalization** is the make-or-break; guard it in every phase (never report intra-patient numbers as headline).
- **Class imbalance** (rare arrhythmias) — carry class weighting / focal loss / resampling forward; report per-class, never just accuracy.
- **Regulatory drift** — the AI Act and MDR timelines move; keep the technical file living, not a one-time deliverable.
- **Motion/noise robustness** — real ambulatory ECG is far noisier than MIT-BIH; budget a dedicated noise-robustness study before pilots.
```
