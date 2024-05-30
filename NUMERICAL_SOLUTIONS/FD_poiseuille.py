import numpy as np
import matplotlib.pyplot as plt
import os
#neki

# Parameters
L = 2.0  # Length of the domain
H = 0.5
N = 100  # Number of grid points
nu = 0.01  # Kinematic viscosity
u0 = 1.0  # unit flow velocity
rho = 1.0  # Density
total_time = 5.0  # Total simulation time
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

# Boundary conditions
u[:, 0] = u0  # Left wall (uniform flow u = u0)
u[:, -1] = u[:, -2]  # Right wall (zero gradient)
u[0, :] = 0  # Bottom wall (y = 0, u = 0)
u[-1, :] = 0  # Top wall (y = H, u = 0)

v[:, 0] = 0  # Left wall (v = 0)
v[:, -1] = 0  # Right wall (v = 0)
v[0, :] = 0  # Bottom wall (y = 0, v = 0)
v[-1, :] = 0  # Top wall (y = H, v = 0)

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
        p[-1, :] = 0         # p = 0 at y = L
        
    return p

def cavity_flow(u, v, dt, dx, dy, p, rho, nu, nit):
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

        # Boundary conditions
        u[:, 0] = u0  # Left wall (uniform flow u = u0)
        u[:, -1] = u[:, -2]  # Right wall (zero gradient)
        u[0, :] = 0  # Bottom wall (y = 0, u = 0)
        u[-1, :] = 0  # Top wall (y = H, u = 0)

        v[:, 0] = 0  # Left wall (v = 0)
        v[:, -1] = 0  # Right wall (v = 0)
        v[0, :] = 0  # Bottom wall (y = 0, v = 0)
        v[-1, :] = 0  # Top wall (y = H, v = 0)

        # Plot every 100 steps
        if n % 100 == 0:
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
    axs[0, 0].set_xlim([0, L])
    axs[0, 0].set_ylim([0, H])
    axs[0, 0].set_aspect(aspect='equal')

    cp = axs[0, 1].contourf(X, Y, p, levels=50, cmap='viridis')
    plt.colorbar(cp, ax=axs[0, 1])
    axs[0, 1].set_title(f'Pressure p at step {step}')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_xlim([0, L])
    axs[0, 1].set_ylim([0, H])
    axs[0, 1].set_aspect(aspect='equal')

    cp = axs[1, 0].contourf(X, Y, u, levels=50, cmap='viridis')
    plt.colorbar(cp, ax=axs[1, 0])
    axs[1, 0].set_title(f'Velocity u at step {step}')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_xlim([0, L])
    axs[1, 0].set_ylim([0, H])
    axs[1, 0].set_aspect(aspect='equal')

    cp = axs[1, 1].contourf(X, Y, v, levels=50, cmap='viridis')
    plt.colorbar(cp, ax=axs[1, 1])
    axs[1, 1].set_title(f'Velocity v at step {step}')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_xlim([0, L])
    axs[1, 1].set_ylim([0, H])
    axs[1, 1].set_aspect(aspect='equal')


    plt.tight_layout()
    output_dir = 'FD_poiseuille'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'output_{step}.png'))
    plt.close()


# Solve the cavity flow problem
u, v, p = cavity_flow(u, v, dt, dx, dy, p, rho, nu, 50)
