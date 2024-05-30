import torch
import torch.nn as nn

class NavierStokesGradientLayer(nn.Module):
    """
    Custom layer to compute derivatives for the steady Navier-Stokes equations using a given neural network model that predicts the stream function psi and pressure p.

    Attributes:
        model: A PyTorch neural network model that outputs [psi, p] given [x, y].
    """

    def __init__(self, model):
        """
        Initialize the NavierStokesGradientLayer with a pre-trained model.

        Args:
            model: A PyTorch neural network model.
        """
        super(NavierStokesGradientLayer, self).__init__()
        self.model = model

    def forward(self, xy):
        xy.requires_grad_(True)
        psi_p = self.model(xy)  # model outputs psi and p 
        psi, p = psi_p[:, 0], psi_p[:, 1]#psi_p[..., 0], psi_p[..., 1]

        #torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, 
        #only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False)

        # First derivatives with retain_graph=True for subsequent second derivative calculations
        psi_grads = torch.autograd.grad(psi, xy, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)
        dpsi_dx, dpsi_dy = psi_grads[0][:, 0], psi_grads[0][:, 1]
        p_grads = torch.autograd.grad(p, xy, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)
        dp_dx, dp_dy = p_grads[0][:, 0], p_grads[0][:, 1]

        u, v = dpsi_dy, -dpsi_dx#-dpsi_dy, dpsi_dx

        # Second derivatives(for psi, first for uv) with retain_graph=True for third derivative calculations
        du_dxy = torch.autograd.grad(outputs=u, inputs=xy, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        du_dx, du_dy = du_dxy[:, 0], du_dxy[:, 1]
        dv_dxy = torch.autograd.grad(outputs=v, inputs=xy, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        dv_dx, dv_dy = dv_dxy[:, 0], dv_dxy[:, 1]

        # Third derivatives(for psi, second for uv)
        du_dx2 = torch.autograd.grad(du_dx, xy, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0]
        du_dy2 = torch.autograd.grad(du_dy, xy, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0][:, 1]#[:, 0] worked better with this shit but i do not get why
        dv_dx2 = torch.autograd.grad(dv_dx, xy, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0][:, 0]
        dv_dy2 = torch.autograd.grad(dv_dy, xy, grad_outputs=torch.ones_like(dv_dy), create_graph=True)[0][:, 1]#[:, 0]

        # Pack gradients
        p_grads = (p, dp_dx, dp_dy)
        u_grads = (u, du_dx, du_dy, du_dx2, du_dy2)
        v_grads = (v, dv_dx, dv_dy, dv_dx2, dv_dy2)

        return psi, p_grads, u_grads, v_grads




    # def forward(self, xy):
    #     """
    #     Apply the layer to input tensors to compute derivatives necessary for the Navier-Stokes equations.

    #     Args:
    #         xy: Input tensor with shape (batch_size, 2), where each entry is [x, y].

    #     Returns:
    #         psi: The stream function tensor.
    #         p_grads: Tuple containing pressure and its gradients (p, dp/dx, dp/dy).
    #         u_grads: Tuple containing u and its gradients (u, du/dx, du/dy, d^2u/dx^2, d^2u/dy^2).
    #         v_grads: Tuple containing v and its gradients (v, dv/dx, dv/dy, d^2v/dx^2, d^2v/dy^2).
    #     """
    #     xy.requires_grad_(True)
    #     psi_p = self.model(xy)  # Model predicts [psi, p]
    #     psi, p = psi_p[..., 0], psi_p[..., 1]

    #     # First derivatives
    #     psi_grads = torch.autograd.grad(psi, xy, grad_outputs=torch.ones_like(psi), create_graph=True)
    #     p_grads = torch.autograd.grad(p, xy, grad_outputs=torch.ones_like(p), create_graph=True)
    #     dpsi_dx, dpsi_dy = psi_grads[0][:, 0], psi_grads[0][:, 1]
    #     dp_dx, dp_dy = p_grads[0][:, 0], p_grads[0][:, 1]

    #     # Velocity components
    #     u, v = dpsi_dy, -dpsi_dx

    #     # Second derivatives
    #     # du_dx, du_dy = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)
    #     # dv_dx, dv_dy = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(v), create_graph=True)
    #     # Assuming u and v are properly shaped vectors (e.g., [batch_size, 1])
    #     du_dx, du_dy = torch.autograd.grad(outputs=u,
    #                                     inputs=xy,
    #                                     grad_outputs=torch.ones_like(u),
    #                                     create_graph=True)[0].T  # Transpose if necessary

    #     dv_dx, dv_dy = torch.autograd.grad(outputs=v,
    #                                     inputs=xy,
    #                                     grad_outputs=torch.ones_like(v),
    #                                     create_graph=True)[0].T  # Transpose if necessary


    #     # Third derivatives
    #     du_dx2, du_dy2 = torch.autograd.grad(du_dx, xy, grad_outputs=torch.ones_like(du_dx)), torch.autograd.grad(du_dy, xy, grad_outputs=torch.ones_like(du_dy))
    #     dv_dx2, dv_dy2 = torch.autograd.grad(dv_dx, xy, grad_outputs=torch.ones_like(dv_dx)), torch.autograd.grad(dv_dy, xy, grad_outputs=torch.ones_like(dv_dy))

    #     # Pack gradients
    #     p_grads = (p, dp_dx, dp_dy)
    #     u_grads = (u, du_dx, du_dy, du_dx2, du_dy2)
    #     v_grads = (v, dv_dx, dv_dy, dv_dx2, dv_dy2)

    #     return psi, p_grads, u_grads, v_grads