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
