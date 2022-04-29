
import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def prop(self, input):
        pass
    def dprop(self, output_grad):
        pass

class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)# turned 1 to 10
    def prop(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias 
    def dprop(self, output_grad, learning_pace):
        weights_grad = np.dot(output_grad, self.input.T)
        input_grad = np.dot(self.weights.T, output_grad)
        self.weights = self.weights - (learning_pace * weights_grad)
        self.bias = self.bias - (learning_pace * output_grad)
        return input_grad

class Activation(Layer):
    def __init__(self, acti, acti_p):
        self.acti = acti
        self.acti_p = acti_p
    def prop(self, input):
        self.input = input
        return self.acti(self.input)
    def dprop(self, output_grad, learning_pace):
        return np.multiply(output_grad, self.acti_p(self.input))

class Hypertan(Activation):
    def __init__(self):
        def hypertan(x):
            return np.tanh(x)
        def hypertan_p(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(hypertan, hypertan_p)
