def predict(network, input):
    output = input
    for layer in network:
        output = layer.prop(output)
    return output

def train(network, loss, dloss, x_train, y_train, epochs = 1000, rate = 0.01):
    for epoch in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
        
            output = predict(network, x)
            #print(output)
            error += loss(y, output)

            grad = dloss(y, output)
            for layer in reversed(network):
                grad = layer.dprop(grad, rate)

        error /= len(x_train)

        print(f"{epoch + 1}/{epochs}, error={error}")
    