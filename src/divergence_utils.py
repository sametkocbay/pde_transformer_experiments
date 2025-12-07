# Hilfsfunktionen zur Berechnung von Divergenz für (u,w) auf (x,z)
import numpy as np
from typing import Optional, Tuple


def compute_divergence(u: np.ndarray, w: np.ndarray, x: Optional[np.ndarray]=None, z: Optional[np.ndarray]=None) -> np.ndarray:
    """
    Berechnet die Divergenz div(u) = du/dx + dw/dz.
    Erwartet Felder mit Dimensionen (X, Z).
    Wenn x/z angegeben, wird np.gradient mit Koordinaten genutzt.
    """
    if x is not None and z is not None:
        du_dx = np.gradient(u, x, axis=0)
        dw_dz = np.gradient(w, z, axis=1)
    else:
        du_dx = np.gradient(u, axis=0)
        dw_dz = np.gradient(w, axis=1)
    return du_dx + dw_dz


def divergence_stats(div: np.ndarray) -> Tuple[float, float]:
    """Gibt (mean, max_abs) zurück."""
    return float(np.mean(div)), float(np.max(np.abs(div)))

