import torch
import torch.nn as nn
import torch.nn.functional as F

class PeriodicConv1D(torch.nn.Conv1d):
    """ Implementing 1D convolutional layer with circular padding.

    """
    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding_circ = (self.padding[0] // 2, (self.padding[0] - 1) // 2)
            return F.conv1d(F.pad(input, expanded_padding_circ, mode='circular'), 
                            self.weight, self.bias, self.stride,
                            (0,), self.dilation, self.groups)
        elif self.padding_mode == 'valid':
            expanded_padding_circ = (self.padding[0] // 2, (self.padding[0] - 1) // 2)
            return F.conv1d(F.pad(input, expanded_padding_circ, mode='constant', value=0.), 
                            self.weight, self.bias, self.stride,
                            (0,), self.dilation, self.groups)
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def setup_conv1d(in_channels, out_channels, kernel_size, bias, padding_mode, stride=1):
    """
    Select between regular and circular 1D convolutional layers.
    padding_mode='circular' returns a convolution that wraps padding around the final axis.
    """
    if padding_mode in ['circular', 'valid']:
        return PeriodicConv1D(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size,
                      bias=bias,
                      stride=stride,
                      padding_mode=padding_mode)
    else:
        return torch.nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(kernel_size-1)//2,
                              stride=stride,
                              bias=bias)

class Model_forwarder_rk4default(nn.Module):

    def __init__(self, model):
        super(Model_forwarder_rk4default, self).__init__()            
        self.model = model

    def forward(self, x, t, dt):
        """ Runke-Katta step with 2/6 rule """
        f0 = self.model.forward(x) # ndim=3 for MinimalConvNet96
        f1 = self.model.forward(x + dt/2.*f0)
        f2 = self.model.forward(x + dt/2.*f1)
        f3 = self.model.forward(x + dt * f2)

        x = x + dt/6. * (f0 + 2.* (f1 + f2) + f3)

        return x
    
class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = torch.squeeze(self.avg_pool(x), dim=-1)
        y = torch.unsqueeze(self.fc(y), dim=-1)
        return x * y.expand_as(x)

# fmt: off
def rk4_torch(f, x, t, dt, stages=4, s=0):
    """Runge-Kutta (explicit, non-adaptive) numerical (S)ODE solvers.

    For ODEs, the order of convergence equals the number of `stages`.

    For SDEs with additive noise (`s>0`), the order of convergence
    (both weak and strong) is 1 for `stages` equal to 1 or 4.
    These correspond to the classic Euler-Maruyama scheme and the Runge-Kutta
    scheme for S-ODEs respectively, see `bib.grudzien2020numerical`
    for a DA-specific discussion on integration schemes and their discretization errors.

    Parameters
    ----------
    f : function
        The time derivative of the dynamical system. Must be of the form `f(t, x)`

    x : ndarray or float
        State vector of the forcing term

    t : float
        Starting time of the integration

    dt : float
        Integration time step.

    stages : int, optional
        The number of stages of the RK method.
        When `stages=1`, this becomes the Euler (-Maruyama) scheme.
        Default: 4.

    s : float
        The diffusion coeffient (std. dev) for models with additive noise.
        Default: 0, yielding deterministic integration.

    Returns
    -------
    ndarray
        State vector at the new time, `t+dt`
    """

    # Draw noise
    if s > 0:
        W = s * torch.sqrt(dt) * torch.random.randn(*x.shape)
    else:
        W = 0

    # Approximations to Delta x
    if stages >= 1: k1 = dt * f(x,           t)         + W    # noqa
    if stages >= 2: k2 = dt * f(x+k1/2.0,    t+dt/2.0)  + W    # noqa
    if stages == 3: k3 = dt * f(x+k2*2.0-k1, t+dt)      + W    # noqa
    if stages == 4:
                    k3 = dt * f(x+k2/2.0,    t+dt/2.0)  + W    # noqa
                    k4 = dt * f(x+k3,        t+dt)      + W    # noqa

    # Mix proxies
    if    stages == 1: y = x + k1                              # noqa
    elif  stages == 2: y = x + k2                              # noqa
    elif  stages == 3: y = x + (k1 + 4.0*k2 + k3)/6.0          # noqa
    elif  stages == 4: y = x + (k1 + 2.0*(k2 + k3) + k4)/6.0   # noqa
    else:
        raise NotImplementedError

    return y