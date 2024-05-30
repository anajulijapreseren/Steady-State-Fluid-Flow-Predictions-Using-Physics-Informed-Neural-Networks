import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
L = 1.0  # Length of the domain
H = 1.0  # Height of the domain
N = 100  # Number of grid points
nu = 0.003  # Kinematic viscosity
rho = 1.0  # Density
total_time = 20.0  # Total simulation time
dt = 0.001  # Time step

# Discretize the domain
dx = L / (N - 1)
dy = H / (N - 1)
x = np.linspace(0, L, N)
y = np.linspace(0, H, N)
X, Y = np.meshgrid(x, y)

# Initialize the velocity and pressure fields
u = np.zeros((N, N))
v = np.zeros((N, N))
p = np.zeros((N, N))

# Helper functions
def build_up_b(rho, dt, dx, dy, u, v):
    b = np.zeros((N, N))
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                     ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) + 
                      (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                     ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                       2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                            (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    
    for q in range(50):  # Number of iterations
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                         b[1:-1, 1:-1])

        # Boundary conditions
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = L
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = H
        
    return p

def generate_structured_boundary_and_internal_points(num_boundary, num_internal):
    boundary_points = []
    boundary_values = []

    # Calculate the number of points along each boundary
    num_inlet = int(num_boundary // 20)
    num_left = int(num_boundary // 20) * 4
    num_right = num_left
    num_other = num_boundary - num_inlet - num_left - num_right  # Distribute the rest evenly

    # Inlet on the left
    inlet_ys = np.linspace(0, 0.2, num_inlet)
    for y in inlet_ys:
        u = 0.3 * (4 / 0.2**2) * y * (0.2 - y)  # Using alpha = 0.3, H = 0.2
        boundary_points.append([0, y])
        boundary_values.append([u, 0])

    # Remaining part of left wall with zero velocities
    left_ys = np.linspace(0.2, 1.0, num_left)
    for y in left_ys:
        boundary_points.append([0, y])
        boundary_values.append([0, 0])

    # # Outlet on the right with the same profile as inlet on the left(ideally we dont need it)
    # outlet_ys = np.linspace(0.8, 1., num_right)
    # for y in outlet_ys:
    #     u = 0.3 * (4 / 0.2**2) * (y-1.) * (0.8 - y)  # Using alpha = 0.3, H = 0.2
    #     boundary_points.append([1, y])
    #     boundary_values.append([u, 0])

    # Right wall with zero velocities
    outlet_ys = np.linspace(0.0, 0.8, num_right)
    for y in outlet_ys:
        boundary_points.append([1, y])
        boundary_values.append([0, 0])

    # Other boundaries, equally spaced
    other_xs_ys = np.linspace(0, 1, num_other // 2)
    for x in other_xs_ys:
        boundary_points.append([x, 0])
        boundary_values.append([0, 0])
        boundary_points.append([x, 1])
        boundary_values.append([0, 0])

    # Internal points on a structured grid
    sqrt_internal = int(np.sqrt(num_internal))
    internal_xs = np.linspace(0, 1, sqrt_internal)
    internal_ys = np.linspace(0, 1, sqrt_internal)
    internal_points = np.array(np.meshgrid(internal_xs, internal_ys)).T.reshape(-1, 2)

    return np.array(boundary_points), np.array(boundary_values), internal_points

def apply_boundary_conditions(u, v, boundary_points, boundary_values):
    for (x, y), (u_val, v_val) in zip(boundary_points, boundary_values):
        i = int(x * (N - 1))
        j = int(y * (N - 1))
        u[j, i] = u_val
        v[j, i] = v_val

def structured_flow(u, v, dt, dx, dy, p, rho, nu, nit, boundary_points, boundary_values):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((N, N))

    for n in range(int(total_time / dt)):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(rho, dt, dx, dy, u, v)
        p = pressure_poisson(p, dx, dy, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) *
                        (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) *
                        (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                         dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        # Apply boundary conditions
        apply_boundary_conditions(u, v, boundary_points, boundary_values)

        # Plot every 1000 steps
        if n % 1000 == 0:
            plot_fields(u, v, p, n)
        
    return u, v, p

def compute_stream_function(u, v):
    psi = np.zeros_like(u)
    for i in range(1, N):
        for j in range(1, N):
            psi[i, j] = psi[i, j-1] + v[i, j]*dx
    for j in range(1, N):
        for i in range(1, N):
            psi[i, j] = psi[i-1, j] - u[i, j]*dy
    return psi

def plot_fields(u, v, p, step):
    psi = compute_stream_function(u, v)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    cp = axs[0, 0].contourf(X, Y, psi, levels=50, cmap='viridis')
    plt.colorbar(cp, ax=axs[0, 0])
    axs[0, 0].set_title(f'Stream Function ψ at step {step}')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')

    cp = axs[0, 1].contourf(X, Y, p, levels=50, cmap='viridis')
    plt.colorbar(cp, ax=axs[0, 1])
    axs[0, 1].set_title(f'Pressure p at step {step}')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')

    cp = axs[1, 0].contourf(X, Y, u, levels=50, cmap='viridis')
    plt.colorbar(cp, ax=axs[1, 0])
    axs[1, 0].set_title(f'Velocity u at step {step}')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')

    cp = axs[1, 1].contourf(X, Y, v, levels=50, cmap='viridis')
    plt.colorbar(cp, ax=axs[1, 1])
    axs[1, 1].set_title(f'Velocity v at step {step}')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')

    plt.tight_layout()
    output_dir = 'FD_inlet'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'output_{step}.png'))
    plt.close()

# Generate boundary and internal points
boundary_points, boundary_values, internal_points = generate_structured_boundary_and_internal_points(2000, 2000)

# Solve the structured flow problem
u, v, p = structured_flow(u, v, dt, dx, dy, p, rho, nu, 50, boundary_points, boundary_values)
