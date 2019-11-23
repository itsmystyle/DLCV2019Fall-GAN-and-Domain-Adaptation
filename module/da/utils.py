from torch.autograd import Function


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, X, Lambda):
        ctx.Lambda = Lambda

        return X.view_as(X)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.Lambda

        return output, None
