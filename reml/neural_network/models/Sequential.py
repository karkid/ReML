class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def params_and_grads(self):
        for layer in self.layers:
            for p, g in layer.params_and_grads():
                yield p, g

    def train(self):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()
