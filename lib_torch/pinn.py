import torch
import torch.nn as nn

from lib_torch.NSGLayer import NavierStokesGradientLayer

"""inputs=[xy_eqn, xy_bnd], outputs=[uv_eqn, psi_bnd, uv_bnd]"""
# uv_eqn returns the NS loss (how much left and right side differ)
# psi_bnd: value of the stream function on boundary points
# uv_bnd: predicted velocities for boundary points


import torch
import torch.nn as nn

from lib_torch.NSGLayer import NavierStokesGradientLayer

class PINN(nn.Module):
    def __init__(self, network, rho=1, nu=0.01):
        super(PINN, self).__init__()
        self.network = network
        self.rho = rho
        self.nu = nu
        self.grads = NavierStokesGradientLayer(network)

    def forward(self, xy_eqn, xy_bnd):
        # Compute outputs and derivatives for equation points
        psi_eqn, p_grads_eqn, u_grads_eqn, v_grads_eqn = self.grads(xy_eqn)
        u_eqn, v_eqn = self.compute_navier_stokes_loss(p_grads_eqn, u_grads_eqn, v_grads_eqn)
        uv_eqn = torch.column_stack((u_eqn, v_eqn))

        # Compute outputs for boundary points
        psi_bnd, p_grads_bnd, u_grads_bnd, v_grads_bnd = self.grads(xy_bnd)
        psi_bnd=psi_bnd
        psi_bnd = torch.column_stack([psi_bnd, psi_bnd])
        # Directly use boundary outputs
        u_bnd = u_grads_bnd[0]
        v_bnd = v_grads_bnd[0]
        uv_bnd = torch.column_stack([u_bnd, v_bnd])
        return uv_eqn, psi_bnd, uv_bnd #u_eqn, v_eqn are losses based on NS equation, u_bnd, v_bnd are actual velocity predictions on boundary

    def compute_navier_stokes_loss(self, p_grads, u_grads, v_grads):
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads

        u_eqn = self.rho * (u * u_x + v * u_y) + p_x - self.nu * (u_xx + u_yy)
        v_eqn = self.rho * (u * v_x + v * v_y) + p_y - self.nu * (v_xx + v_yy)
        return u_eqn, v_eqn
