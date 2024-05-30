import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from lib_torch.pinn import NavierStokesGradientLayer
from lib_torch.advnetwork import Network

config = {
    'num_inputs': 2,
    'num_layers': 8,
    'neurons_per_layer': 40,
    'num_outputs': 2,
    'activation': 'tanh',
    'dropout_rate': 0.0,
    'use_skip_connections': False,
    'init_type': 'kaiming',
    'alpha': 1,  # Weight for boundary loss
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_structured_boundary_and_internal_points(num_boundary, num_internal):
    boundary_points = []
    boundary_values = []

    # Calculate the number of points along each boundary
    num_inlet = int(num_boundary // 19)
    num_left = int(num_boundary // 19) * 4
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

    # Outlet on the right with the same profile as inlet on the left
    outlet_ys = np.linspace(0.8, 1., num_right)
    for y in outlet_ys:
        u = 0.3 * (4 / 0.2**2) * (y-1.) * (0.8 - y)  # Using alpha = 0.3, H = 0.2
        boundary_points.append([1, y])
        boundary_values.append([u, 0])

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

class L_BFGS_B:
    def __init__(self, model, x_train, y_train, rho, nu, max_iter=50000, tol=1e-9, print_interval=100):
        self.model = model.to(device)
        self.x_train = [torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device) for x in x_train]
        self.y_train = [torch.tensor(y, dtype=torch.float32).to(device) for y in y_train]
        self.rho = rho
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.print_interval = print_interval
        self.iteration = 0

    def closure(self):
        self.optimizer.zero_grad()
        loss = self.model.loss_fn(self.x_train, self.y_train, self.rho, self.nu)
        loss.backward()
        if self.iteration % self.print_interval == 0:
            print(f"Iteration {self.iteration}, Loss: {loss.item()}")
            self.print_fields(epoch=self.iteration)
        self.iteration += 1
        return loss

    def print_fields(self, epoch):
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        x, y = np.meshgrid(x, y)
        xy = np.stack([x.flatten(), y.flatten()], axis=-1)
        xy_tensor = torch.tensor(xy, dtype=torch.float32, requires_grad=True).to(device)

        u, v, psi, p = uv(self.model.network, xy_tensor)

        u = u.reshape(100, 100)
        v = v.reshape(100, 100)
        psi, p = psi.reshape(100, 100), p.reshape(100, 100)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        plot_contours(x, y, psi, 'Stream Function ψ', axs[0, 0])
        plot_contours(x, y, p, 'Pressure p', axs[0, 1])
        plot_contours(x, y, u, 'Velocity u', axs[1, 0])
        plot_contours(x, y, v, 'Velocity v', axs[1, 1])
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'plots_inlet/main_inlet_{epoch}.png')

    def fit(self):
        self.optimizer = optim.LBFGS(self.model.parameters(), max_iter=self.max_iter, tolerance_grad=self.tol)
        self.optimizer.step(self.closure)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

class PINN(nn.Module):
    def __init__(self, network, rho=1, nu=0.01):
        super(PINN, self).__init__()
        self.network = network
        self.grads = NavierStokesGradientLayer(network)
        self.rho = rho
        self.nu = nu

    def forward(self, xy_eqn, xy_bnd):
        psi_eqn, p_grads_eqn, u_grads_eqn, v_grads_eqn = self.grads(xy_eqn)
        uv_eqn = torch.column_stack(self.compute_navier_stokes_loss(p_grads_eqn, u_grads_eqn, v_grads_eqn))

        psi_bnd, p_grads_bnd, u_grads_bnd, v_grads_bnd = self.grads(xy_bnd)
        psi_bnd = torch.column_stack([psi_bnd, psi_bnd])
        u_bnd = u_grads_bnd[0]
        v_bnd = v_grads_bnd[0]
        uv_bnd = torch.column_stack([u_bnd, v_bnd])
        return uv_eqn, psi_bnd, uv_bnd

    def compute_navier_stokes_loss(self, p_grads, u_grads, v_grads):
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads

        u_eqn = self.rho * (u * u_x + v * u_y) + p_x - self.nu * (u_xx + u_yy)
        v_eqn = self.rho * (u * v_x + v * v_y) + p_y - self.nu * (v_xx + v_yy)
        return u_eqn, v_eqn

    def loss_fn(self, x_train, y_train, rho, nu):
        xy_eqn, xy_bnd = x_train
        uv_eqn, psi_bnd, uv_bnd = self.forward(xy_eqn, xy_bnd)
        zeros, zeros_bnd, uv_bnd_target = y_train

        loss_eqn = ((uv_eqn - zeros) ** 2).mean()
        loss_bnd = ((uv_bnd - uv_bnd_target) ** 2).mean()
        loss = loss_eqn + config['alpha'] * loss_bnd
        return loss

def uv(network, xy_tensor):
    network.eval()
    psi_p = network(xy_tensor)
    psi, p = psi_p[:, 0], psi_p[:, 1]

    with torch.enable_grad():
        grad_psi = torch.autograd.grad(psi.sum(), xy_tensor, create_graph=True)[0]
        u = grad_psi[:, 1]
        v = -grad_psi[:, 0]

    return u.detach().cpu().numpy(), v.detach().cpu().numpy(), psi.detach().cpu().numpy(), p.detach().cpu().numpy()

def plot_contours(x, y, z, title, ax):
    if z.min() == z.max():
        contour_levels = np.linspace(z.min(), z.max() + 1, num=50)
    else:
        contour_levels = np.linspace(z.min(), z.max(), num=50)
    cp = ax.contourf(x, y, z, levels=contour_levels, cmap='viridis')
    plt.colorbar(cp, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

if __name__ == '__main__':
    num_train_samples = 10000
    num_test_samples = 200
    u0 = 1.

    network = Network(
        num_inputs=config['num_inputs'], 
        num_layers=config['num_layers'], 
        neurons_per_layer=config['neurons_per_layer'], 
        num_outputs=config['num_outputs'],
        activation=config['activation'], 
        dropout_rate=config['dropout_rate'], 
        use_skip_connections=config['use_skip_connections'],
        init_type=config['init_type']
    )

    pinn = PINN(network)

    # Generate structured boundary and internal points
    boundary_points, boundary_values, internal_points = generate_structured_boundary_and_internal_points(num_boundary=400, num_internal=9600)

    x_train = [internal_points, boundary_points]
    y_train = [np.zeros((len(internal_points), 2)), np.zeros((len(boundary_points), 2)), boundary_values]

    # Define different parameters for rho and nu
    parameter_sets = [
        {'rho': 1.0, 'nu': 0.01},
        {'rho': 1.0, 'nu': 0.02},
        {'rho': 1.5, 'nu': 0.01},
        {'rho': 1.5, 'nu': 0.02},
    ]

    for params in parameter_sets:
        rho = params['rho']
        nu = params['nu']
        print(f"Training with rho={rho}, nu={nu}")

        lbfgs = L_BFGS_B(
            model=pinn, 
            x_train=x_train, 
            y_train=y_train, 
            rho=rho,
            nu=nu,
            print_interval=200,
            tol=1e-12
        )
        lbfgs.fit()

        # Save the trained model
        model_filename = f"pinn_model_rho_{rho}_nu_{nu}.pth"
        lbfgs.save_model(model_filename)

        x = np.linspace(0, 1, num_test_samples)
        y = np.linspace(0, 1, num_test_samples)
        x, y = np.meshgrid(x, y)
        xy = np.stack([x.flatten(), y.flatten()], axis=-1)
        xy_tensor = torch.tensor(xy, dtype=torch.float32, requires_grad=True).to(device)

        u, v, psi, p = uv(network, xy_tensor)

        u = u.reshape(num_test_samples, num_test_samples)
        v = v.reshape(num_test_samples, num_test_samples)
        psi, p = psi.reshape(num_test_samples, num_test_samples), p.reshape(num_test_samples, num_test_samples)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        plot_contours(x, y, psi, 'Stream Function ψ', axs[0, 0])
        plot_contours(x, y, p, 'Pressure p', axs[0, 1])
        plot_contours(x, y, u, 'Velocity u', axs[1, 0])
        plot_contours(x, y, v, 'Velocity v', axs[1, 1])
        plt.tight_layout()
        plt.show()
