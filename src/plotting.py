import numpy as np
import matplotlib.pyplot as plt
from orszag_simulator import simulate_orszag_tang  # Annahme: Simulator ist in dieser Datei


def compute_derived_fields(trajectory, L=2 * np.pi):
    """
    Berechnet Vortizität (omega) und Stromdichte (current) aus den Rohdaten.
    trajectory: [Time, 4, N, N] -> (u, v, Bx, By)
    """
    # Grid spacing
    N = trajectory.shape[2]
    dx = L / N

    # Entpacken
    u = trajectory[:, 0]
    v = trajectory[:, 1]
    Bx = trajectory[:, 2]
    By = trajectory[:, 3]

    # Gradienten berechnen (np.gradient nutzt zentrale Differenzen)
    # axis=1 ist y-Richtung (Zeilen), axis=2 ist x-Richtung (Spalten)
    du_dy = np.gradient(u, dx, axis=1)
    dv_dx = np.gradient(v, dx, axis=2)
    dBx_dy = np.gradient(Bx, dx, axis=1)
    dBy_dx = np.gradient(By, dx, axis=2)

    # Physik:
    # Vortizität omega = dv/dx - du/dy
    omega = dv_dx - du_dy

    # Stromdichte J = dBy/dx - dBx/dy (in 2D z-Komponente des Stroms)
    current = dBy_dx - dBx_dy

    # Kinetische und Magnetische Energie (Integral über Fläche)
    # E ~ sum(u^2 + v^2)
    E_kin = 0.5 * np.mean(u ** 2 + v ** 2, axis=(1, 2))
    E_mag = 0.5 * np.mean(Bx ** 2 + By ** 2, axis=(1, 2))

    return omega, current, E_kin, E_mag


def visualize_simulation(trajectory, time_points):
    """
    Erstellt Plots für Zeitentwicklung und Felder.
    """
    omega, current, E_kin, E_mag = compute_derived_fields(trajectory)

    # 1. Energie-Plot (Erhaltungs-Check)
    plt.figure(figsize=(10, 5))
    plt.plot(time_points, E_kin, label='$E_{kin}$ (Kinetisch)')
    plt.plot(time_points, E_mag, label='$E_{mag}$ (Magnetisch)')
    plt.plot(time_points, E_kin + E_mag, 'k--', label='$E_{tot}$ (Gesamt)')
    plt.title('Energieerhaltung und -umwandlung')
    plt.xlabel('Zeit')
    plt.ylabel('Energie (normiert)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Feld-Visualisierung (Snapshots)
    # Wir zeigen t=0 (Start), t=Mitte, t=Ende
    indices = [0, len(time_points) // 2, len(time_points) - 1]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i, idx in enumerate(indices):
        t = time_points[idx]

        # Zeile i: Spalte 0=Vortizität, Spalte 1=Stromdichte, Spalte 2=Magnetfeld-Stärke

        # Vortizität
        im0 = axes[i, 0].imshow(omega[idx], cmap='RdBu_r', origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi])
        axes[i, 0].set_title(f'Vortizität $\omega$ (t={t:.2f})')
        fig.colorbar(im0, ax=axes[i, 0])

        # Stromdichte (zeigt Schockwellen/Rekonnexion)
        im1 = axes[i, 1].imshow(current[idx], cmap='inferno', origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi])
        axes[i, 1].set_title(f'Stromdichte $J$ (t={t:.2f})')
        fig.colorbar(im1, ax=axes[i, 1])

        # Magnetfeld B (Betrag)
        B_mag = np.sqrt(trajectory[idx, 2] ** 2 + trajectory[idx, 3] ** 2)
        im2 = axes[i, 2].imshow(B_mag, cmap='viridis', origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi])
        axes[i, 2].set_title(f'|B| Feldstärke (t={t:.2f})')
        fig.colorbar(im2, ax=axes[i, 2])

    plt.tight_layout()
    plt.show()

