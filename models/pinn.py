
import torch
import torch.nn as nn

class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(PINNs, self).__init__()

        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, y, t):
        inputs = torch.cat((x, y, t), dim=-1)
        return self.network(inputs)

def compute_pde_loss(model, x, y, t, u_train, v_train, rho, mu, K, H_train, T_train):
    '''
    Compute the PDE loss for Navier-Stokes and energy conservation.
    '''
    psi_and_p = model(x, y, t)
    psi = psi_and_p[:, 0:1]
    p = psi_and_p[:, 1:2]

    # Compute velocity components
    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
    v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

    # Compute temporal and spatial derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

    # Pressure derivatives
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    # Energy conservation
    T_t = torch.autograd.grad(T_train, t, grad_outputs=torch.ones_like(T_train), retain_graph=True, create_graph=True)[0]
    T_x = torch.autograd.grad(T_train, x, grad_outputs=torch.ones_like(T_train), retain_graph=True, create_graph=True)[0]
    T_y = torch.autograd.grad(T_train, y, grad_outputs=torch.ones_like(T_train), retain_graph=True, create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), retain_graph=True, create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), retain_graph=True, create_graph=True)[0]

    # PDE residuals
    f_u = u_t + (u * u_x + v * u_y) + p_x - (mu / rho) * (u_xx + u_yy)
    f_v = v_t + (u * v_x + v * v_y) + p_y - (mu / rho) * (v_xx + v_yy)
    f_T = T_t + (u * T_x + v * T_y) - K * (T_xx + T_yy)

    # Loss function
    loss = (
        torch.mean((u - u_train) ** 2) +
        torch.mean((v - v_train) ** 2) +
        torch.mean(f_u ** 2) +
        torch.mean(f_v ** 2) +
        torch.mean(f_T ** 2)
    )
    return loss
