from torch.autograd import Function


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, X, Lambda):
        ctx.Lambda = Lambda

        return X.view_as(X)

    @staticmethod
    def backward(ctx, grad):
        grad_output = grad.neg() * ctx.Lambda

        return grad_output, None
