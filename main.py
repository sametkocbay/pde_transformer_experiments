
from src.orzag_tang import simulate_orszag_tang
from src.plotting import visualize_simulation

if __name__ == "__main__":
    # Simulation laufen lassen
    print("Starte Simulation...")
    traj, t = simulate_orszag_tang(
        reynolds=1000.0,
        N=128,  # Höhere Auflösung für bessere Visualisierung der Schocks
        stop_time=2.0
    )

    if traj is not None:
        print(f"Daten generiert: {traj.shape}")
        visualize_simulation(traj, t)