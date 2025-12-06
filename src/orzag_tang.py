import numpy as np
import dedalus.public as d3
import logging
import warnings

# Suppress extensive logging from Dedalus for the loop
logger = logging.getLogger(__name__)
logging.getLogger('dedalus').setLevel(logging.WARNING)


def simulate_orszag_tang(
        reynolds=1000.0,
        magnetic_reynolds=1000.0,
        N=64,  # Spatial resolution (Nx = Ny = N)
        L=2 * np.pi,  # Domain size
        stop_time=2.0,  # How long to simulate
        snapshot_freq=0.1  # Time delta between saved frames
):
    """
    Runs a single 2D Orszag-Tang Vortex simulation.

    Returns:
        trajectory (np.array): Shape [Time, 4, N, N].
                               Channels are [u, v, Bx, By].
        time_points (np.array): Shape [Time].
    """

    # --- 1. Domain & Basis Setup ---
    # We recreate the basis inside the function to allow changing N
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=N, bounds=(0, L), dealias=3 / 2)
    ybasis = d3.RealFourier(coords['y'], size=N, bounds=(0, L), dealias=3 / 2)

    # Fields
    psi = dist.Field(name='psi', bases=(xbasis, ybasis))
    A = dist.Field(name='A', bases=(xbasis, ybasis))
    omega = dist.Field(name='omega', bases=(xbasis, ybasis))
    j = dist.Field(name='j', bases=(xbasis, ybasis))

    # Grid access
    x, y = dist.local_grids(xbasis, ybasis)

    # --- 2. Equations (Incompressible Viscous MHD) ---
    nu = 1 / reynolds
    eta = 1 / magnetic_reynolds

    # Derivatives
    dx = lambda f: d3.Differentiate(f, coords['x'])
    dy = lambda f: d3.Differentiate(f, coords['y'])
    lap = lambda f: d3.Laplacian(f)

    # Problem Definition
    problem = d3.IVP([psi, omega, A, j], namespace=locals())

    # Poisson equations
    problem.add_equation("lap(psi) + omega = 0")
    problem.add_equation("lap(A) + j = 0")

    # Evolution equations (Vorticity & Induction)
    problem.add_equation(
        "dt(omega) - nu*lap(omega) = - (dy(psi)*dx(omega) - dx(psi)*dy(omega)) + (dy(A)*dx(j) - dx(A)*dy(j))")
    problem.add_equation("dt(A) - eta*lap(A) = - (dy(psi)*dx(A) - dx(psi)*dy(A))")

    # --- 3. Initial Conditions (Orszag-Tang) ---
    # Potentials that generate the standard v and B fields
    psi['g'] = 2 * (np.cos(y) + np.cos(x))
    A['g'] = 2 * (np.cos(y) + 0.5 * np.cos(2 * x))

    # Initialize auxiliary fields (vorticity/current) consistent with potentials
    omega_op = -lap(psi)
    j_op = -lap(A)
    omega['g'] = omega_op.evaluate()['g']
    j['g'] = j_op.evaluate()['g']

    # --- 4. Solver Execution ---
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = stop_time

    # Storage for the trajectory
    # Channels: 0:u, 1:v, 2:Bx, 3:By
    snapshots = []
    times = []

    # Helper to extract physical fields from potentials
    def get_fields():
        # u = dy(psi), v = -dx(psi)
        u = dy(psi).evaluate()['g']
        v = -dx(psi).evaluate()['g']
        # Bx = dy(A), By = -dx(A)
        Bx = dy(A).evaluate()['g']
        By = -dx(A).evaluate()['g']
        # Stack into shape [4, N, N]
        return np.stack([u, v, Bx, By], axis=0)

    # Save initial state (t=0)
    snapshots.append(get_fields())
    times.append(solver.sim_time)

    # Main Loop
    try:
        while solver.proceed:
            solver.step(snapshot_freq)
            if solver.iteration > 0:  # Avoid duplicating t=0 if step logic overlaps
                snapshots.append(get_fields())
                times.append(solver.sim_time)

    except Exception as e:
        print(f"Simulation failed at Re={reynolds}: {e}")
        return None, None

    # Convert list to one big numpy array [Time, 4, N, N]
    trajectory = np.stack(snapshots, axis=0)
    time_points = np.array(times)

    return trajectory, time_points


