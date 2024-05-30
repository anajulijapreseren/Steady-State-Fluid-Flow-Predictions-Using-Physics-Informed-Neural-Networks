import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from lib_torch.NSGLayer import NavierStokesGradientLayer
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

def load_true_data(filename, num_samples=1600):
    data = pd.read_csv(filename)
    sampled_data = data.sample(n=num_samples)
    xy_true = sampled_data[['x', 'y']].values
    uv_true = sampled_data[['u', 'v']].values
    return xy_true, uv_true

class L_BFGS_B:
    def __init__(self, model, x_train, y_train, true_data, max_iter=50000, tol=1e-9, print_interval=100):
        self.model = model.to(device)
        self.x_train = [torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device) for x in x_train]
        self.y_train = [torch.tensor(y, dtype=torch.float32).to(device) for y in y_train]
        self.xy_true = torch.tensor(true_data[0], dtype=torch.float32, requires_grad=True).to(device)
        self.uv_true = torch.tensor(true_data[1], dtype=torch.float32).to(device)
        self.max_iter = max_iter
        self.tol = tol
        self.print_interval = print_interval
        self.iteration = 0

    def closure(self):
        self.optimizer.zero_grad()
        loss = self.model.loss_fn(self.x_train, self.y_train, self.xy_true, self.uv_true)
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
        self.rho = rho
        self.nu = nu
        self.grads = NavierStokesGradientLayer(network)

    def forward(self, xy_eqn, xy_bnd, xy_true=None):
        psi_eqn, p_grads_eqn, u_grads_eqn, v_grads_eqn = self.grads(xy_eqn)
        u_eqn, v_eqn = self.compute_navier_stokes_loss(p_grads_eqn, u_grads_eqn, v_grads_eqn)
        uv_eqn = torch.column_stack((u_eqn, v_eqn))

        psi_bnd, p_grads_bnd, u_grads_bnd, v_grads_bnd = self.grads(xy_bnd)
        psi_bnd = torch.column_stack([psi_bnd, psi_bnd])
        u_bnd = u_grads_bnd[0]
        v_bnd = v_grads_bnd[0]
        uv_bnd = torch.column_stack([u_bnd, v_bnd])

        if xy_true is not None:
            #psi_p_true = self.network(xy_true)
            psi_true, p_true = self.network(xy_true).chunk(2, dim=-1)
            #psi_true = psi_p_true[:, 0]

            with torch.enable_grad():
                #grad_psi_true = torch.autograd.grad(psi_true.sum(), xy_true, create_graph=True)#[0]
                grad_psi_true = torch.autograd.grad(psi_true, xy_true, grad_outputs=torch.ones_like(psi_true) ,create_graph=True)[0]
                #psi_grads = torch.autograd.grad(psi, xy, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)
                u_true = grad_psi_true[:, 1]
                v_true = -grad_psi_true[:, 0]
            uv_true = torch.column_stack([u_true, v_true])
            return uv_eqn, psi_bnd, uv_bnd, uv_true
        else:
            return uv_eqn, psi_bnd, uv_bnd

    def compute_navier_stokes_loss(self, p_grads, u_grads, v_grads):
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads

        u_eqn = self.rho * (u * u_x + v * u_y) + p_x - self.nu * (u_xx + u_yy)
        v_eqn = self.rho * (u * v_x + v * v_y) + p_y - self.nu * (v_xx + v_yy)
        return u_eqn, v_eqn

    def loss_fn(self, x_train, y_train, xy_true, uv_true):
        xy_eqn, xy_bnd = x_train
        uv_eqn, psi_bnd, uv_bnd, uv_pred_true = self.forward(xy_eqn, xy_bnd, xy_true)
        zeros, zeros_bnd, uv_bnd_target = y_train

        loss_eqn = ((uv_eqn - zeros) ** 2).mean()
        loss_bnd = ((uv_bnd - uv_bnd_target) ** 2).mean()
        loss_true = ((uv_pred_true - uv_true) ** 2).mean()
        loss = loss_eqn + config['alpha'] * loss_bnd + config['alpha'] * 10 * loss_true
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
    rho = 1.
    nu = 0.01

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

    #network = torch.load('pinn_model.pth')

    pinn = PINN(network, rho=rho, nu=nu)

    # Generate structured boundary and internal points
    boundary_points, boundary_values, internal_points = generate_structured_boundary_and_internal_points(num_boundary=1000, num_internal=8000)

    x_train = [internal_points, boundary_points]
    y_train = [np.zeros((len(internal_points), 2)), np.zeros((len(boundary_points), 2)), boundary_values]

    # Load true data
    xy_true, uv_true = load_true_data('combined_data.csv', num_samples=1600)

    lbfgs = L_BFGS_B(
        model=pinn, 
        x_train=x_train, 
        y_train=y_train, 
        true_data=(xy_true, uv_true),
        print_interval=200,
        tol=1e-14
    )

    lbfgs.fit()
    # Save the trained model
    lbfgs.save_model("pinn_model.pth")

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
