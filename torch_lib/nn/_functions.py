import torch
import torch.nn.functional as F
from torch.autograd import Function, Variable


class Mish(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.tanh(F.softplus(x))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class CovPool(Function):
    """Covariance pooling function.
    """
    @staticmethod
    def forward(ctx, x):
        batch, channels, height, width = x.size()
        n = height * width
        xn = x.reshape(batch, channels, n)
        identity_bar = ((1.0 / n) * torch.eye(n, dtype=xn.dtype, device=xn.device)).unsqueeze(dim=0).repeat(batch, 1, 1)
        ones_bar = torch.full((batch, n, n), fill_value=(-1.0 / n / n), dtype=xn.dtype, device=xn.device)
        i_bar = identity_bar + ones_bar
        sigma = xn.bmm(i_bar).bmm(xn.transpose(1, 2))
        ctx.save_for_backward(x, i_bar)
        return sigma

    @staticmethod
    def backward(ctx, grad_sigma):
        x, i_bar = ctx.saved_tensors
        batch, channels, height, width = x.size()
        n = height * width
        xn = x.reshape(batch, channels, n)
        grad_x = grad_sigma + grad_sigma.transpose(1, 2)
        grad_x = grad_x.bmm(xn).bmm(i_bar)
        grad_x = grad_x.reshape(batch, channels, height, width)
        return grad_x


class NewtonSchulzSqrt(Function):
    """Newton-Schulz iterative matrix square root function.
    Parameters:
    ----------
    x : Tensor
        Input tensor (batch * cols * rows).
    n : int
        Number of iterations (n > 1).
    """
    @staticmethod
    def forward(ctx, x, n):
        assert (n > 1)
        batch, cols, rows = x.size()
        assert (cols == rows)
        m = cols
        identity = torch.eye(m, dtype=x.dtype, device=x.device).unsqueeze(dim=0).repeat(batch, 1, 1)
        x_trace = (x * identity).sum(dim=(1, 2), keepdim=True)
        a = x / x_trace
        i3 = 3.0 * identity
        yi = torch.zeros(batch, n - 1, m, m, dtype=x.dtype, device=x.device)
        zi = torch.zeros(batch, n - 1, m, m, dtype=x.dtype, device=x.device)
        b2 = 0.5 * (i3 - a)
        yi[:, 0, :, :] = a.bmm(b2)
        zi[:, 0, :, :] = b2
        for i in range(1, n - 1):
            b2 = 0.5 * (i3 - zi[:, i - 1, :, :].bmm(yi[:, i - 1, :, :]))
            yi[:, i, :, :] = yi[:, i - 1, :, :].bmm(b2)
            zi[:, i, :, :] = b2.bmm(zi[:, i - 1, :, :])
        b2 = 0.5 * (i3 - zi[:, n - 2, :, :].bmm(yi[:, n - 2, :, :]))
        yn = yi[:, n - 2, :, :].bmm(b2)
        x_trace_sqrt = torch.sqrt(x_trace)
        c = yn * x_trace_sqrt
        ctx.save_for_backward(x, x_trace, a, yi, zi, yn, x_trace_sqrt)
        ctx.n = n
        return c

    @staticmethod
    def backward(ctx, grad_c):
        x, x_trace, a, yi, zi, yn, x_trace_sqrt = ctx.saved_tensors
        n = ctx.n
        batch, m, _ = x.size()
        identity0 = torch.eye(m, dtype=x.dtype, device=x.device)
        identity = identity0.unsqueeze(dim=0).repeat(batch, 1, 1)
        i3 = 3.0 * identity

        grad_yn = grad_c * x_trace_sqrt
        b = i3 - yi[:, n - 2, :, :].bmm(zi[:, n - 2, :, :])
        grad_yi = 0.5 * (grad_yn.bmm(b) - zi[:, n - 2, :, :].bmm(yi[:, n - 2, :, :]).bmm(grad_yn))
        grad_zi = -0.5 * yi[:, n - 2, :, :].bmm(grad_yn).bmm(yi[:, n - 2, :, :])
        for i in range(n - 3, -1, -1):
            b = i3 - yi[:, i, :, :].bmm(zi[:, i, :, :])
            ziyi = zi[:, i, :, :].bmm(yi[:, i, :, :])
            grad_yi_m1 = 0.5 * (grad_yi.bmm(b) - zi[:, i, :, :].bmm(grad_zi).bmm(zi[:, i, :, :]) - ziyi.bmm(grad_yi))
            grad_zi_m1 = 0.5 * (b.bmm(grad_zi) - yi[:, i, :, :].bmm(grad_yi).bmm(yi[:, i, :, :]) - grad_zi.bmm(ziyi))
            grad_yi = grad_yi_m1
            grad_zi = grad_zi_m1

        grad_a = 0.5 * (grad_yi.bmm(i3 - a) - grad_zi - a.bmm(grad_yi))

        x_trace_sqr = x_trace * x_trace
        grad_atx_trace = (grad_a.transpose(1, 2).bmm(x) * identity).sum(dim=(1, 2), keepdim=True)
        grad_cty_trace = (grad_c.transpose(1, 2).bmm(yn) * identity).sum(dim=(1, 2), keepdim=True)
        grad_x_extra = (0.5 * grad_cty_trace / x_trace_sqrt - grad_atx_trace / x_trace_sqr).repeat(1, m, m) * identity

        grad_x = grad_a / x_trace + grad_x_extra
        return grad_x, None


class Triuvec(Function):
    """Extract upper triangular part of matrix into vector form.
    """
    @staticmethod
    def forward(ctx, x):
        batch, cols, rows = x.size()
        assert (cols == rows)
        n = cols
        triuvec_inds = torch.ones(n, n).triu().view(n * n).nonzero()
        x_vec = x.reshape(batch, -1)
        y = x_vec[:, triuvec_inds]
        ctx.save_for_backward(x, triuvec_inds)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, triuvec_inds = ctx.saved_tensors
        batch, n, _ = x.size()
        grad_x = torch.zeros_like(x).view(batch, -1)
        grad_x[:, triuvec_inds] = grad_y
        grad_x = grad_x.view(batch, n, n)
        return grad_x


# TODO: pytrochcv shakedrop
class ShakeDrop(Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        if training:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(0, 1)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None
