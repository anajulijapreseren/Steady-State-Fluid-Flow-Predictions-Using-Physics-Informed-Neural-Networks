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

class L_BFGS_B:
    def __init__(self, model, x_train, y_train, max_iter=50000, tol=1e-9, print_interval=100):
        self.model = model
        self.x_train = [torch.tensor(x, dtype=torch.float32, requires_grad=True) for x in x_train]
        self.y_train = [torch.tensor(y, dtype=torch.float32) for y in y_train]
        self.max_iter = max_iter
        self.tol = tol
        self.print_interval = print_interval
        self.iteration = 0

    def closure(self):
        self.optimizer.zero_grad()
        loss = self.model.loss_fn(self.x_train, self.y_train)
        loss.backward()
        if self.iteration % self.print_interval == 0:
            print(f"Iteration {self.iteration}, Loss: {loss.item()}")
            self.print_fields()
        self.iteration += 1
        return loss

    def print_fields(self):
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        x, y = np.meshgrid(x, y)
        xy = np.stack([x.flatten(), y.flatten()], axis=-1)
        xy_tensor = torch.tensor(xy, dtype=torch.float32, requires_grad=True)

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
        plt.show()

    def fit(self):
        self.optimizer = optim.LBFGS(self.model.parameters(), max_iter=self.max_iter, tolerance_grad=self.tol)
        self.optimizer.step(self.closure)

class PINN(nn.Module):
    def __init__(self, network, rho=1, nu=0.01):
        super(PINN, self).__init__()
        self.network = network
        self.rho = rho
        self.nu = nu
        self.grads = NavierStokesGradientLayer(network)

    def forward(self, xy_eqn, xy_bnd):
        psi_eqn, p_grads_eqn, u_grads_eqn, v_grads_eqn = self.grads(xy_eqn)
        u_eqn, v_eqn = self.compute_navier_stokes_loss(p_grads_eqn, u_grads_eqn, v_grads_eqn)
        uv_eqn = torch.column_stack((u_eqn, v_eqn))

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

    def loss_fn(self, x_train, y_train):
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
    contour_levels = np.linspace(z.min(), z.max(), num=50)
    cp = ax.contourf(x, y, z, levels=contour_levels, cmap='inferno')
    plt.colorbar(cp, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

if __name__ == '__main__':
    num_train_samples = 10000
    num_test_samples = 2000
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

    pinn = PINN(network, rho=rho, nu=nu)

    xy_eqn = np.random.rand(num_train_samples, 2)
    xy_ub = np.random.rand(num_train_samples//2, 2)  
    xy_ub[..., 1] = np.round(xy_ub[..., 1])          
    xy_lr = np.random.rand(num_train_samples//2, 2)  
    xy_lr[..., 0] = np.round(xy_lr[..., 0])          
    xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr]))
    x_train = [xy_eqn, xy_bnd]

    zeros = np.zeros((num_train_samples, 2))
    uv_bnd = np.zeros((num_train_samples, 2))
    uv_bnd[..., 0] = u0 * np.floor(xy_bnd[..., 1])
    y_train = [zeros, zeros, uv_bnd]

    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train, print_interval=100)
    lbfgs.fit()

    x = np.linspace(0, 1, num_test_samples)
    y = np.linspace(0, 1, num_test_samples)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    xy_tensor = torch.tensor(xy, dtype=torch.float32, requires_grad=True)

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
