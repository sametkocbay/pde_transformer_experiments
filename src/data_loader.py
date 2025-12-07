import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any


def find_first_h5(dataset_dir: str) -> Path:
    p = Path(dataset_dir)
    files = list(p.glob('*.h5'))
    if not files:
        raise FileNotFoundError(f"Keine .h5-Datei in {dataset_dir} gefunden.")
    return files[0]


def load_shear_velocity(filename: str) -> Dict[str, Any]:
    f = h5py.File(filename, 'r')
    try:
        data = {
            'file': f,
            'time': f['scales']['sim_time'][:],
            'vel': f['tasks']['shear_velocity'][:],
            'x': f['scales']['x']['1.0'][:] if 'x' in f['scales'] else None,
            'z': f['scales']['z']['1.0'][:] if 'z' in f['scales'] else None,
        }
    except KeyError:
        f.close()
        raise
    return data

