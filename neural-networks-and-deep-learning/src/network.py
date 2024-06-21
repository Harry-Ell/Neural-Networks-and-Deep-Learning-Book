import random
import time
import numpy as np

"""
The logical flow of the functions here is as follows: 
1. We initialise a network of some size, with the first layer taken 
to be an input layer. This means it recieves no biases. We define an 
m by n matrix connecting every layer of n to the following layer of 
dimenion m. This allows activations in layer L to be written in terms 
of activations, a, in layer L-1 as a matrix equation, sigmoid(Wa+b).
2. 'feedforward' takes each activations in any given layer and calculates 
the activations in the next layer. This is a simple matirx equation
handled by numpy. For it to do this, it must be given some activation 
funciton. To begin, we select sigmoid. 
3. stochastic_gradient_descent' takes the training data, which is a list of 
tuples, and breaks it up into smaller batches. Each of these batches is 
small, e.g. size 10, and stochastic_gradient_descent will pass this mini batch 
into update_mini_batch. It ends by calling the evaluate function, which tells
us how well our network is doing on any given epoch 
4. update_mini_batch updates the network's weights and biases by applying
gradient descent using backpropagation to a single mini batch.
The 'mini_batch' is a list of tuples '(x, y)', and 'eta'
is the learning rate.
5. 'backpropagation' tells us how to tweak the weights and biases to reduce the 
cost function, using the backpropagation algorithm. 
"""

class Network(object):
    """ We will initialise our network with random weights and 
    values, but later we will look into better ways to do this """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    """"
    Here we are defining a class which has properties of number 
    of layers, the sizes of each of these, a set of biases, which here 
    is for only hidden layers and final layers. The yx1 vector for y 
    in sizes achieves this. We also have a final set of weights, which 
    is a set of x by y matiricies, which join up each layer to the next. 
    E.g., if sizes looks like [784, 128, 64, 10], then sizes[:-1] 
    = [784, 128, 64], and sizes[1:] = [128, 64, 10]. This means that we 
    are defining a set of matiricies which join up all of the layers to 
    one another. As one may expect, to go from a 784 input layer to a 128 
    output, you need a 784x128 matrix, and that is what is returned. By 
    convention, we assume the first layer is an input layer and hence do 
    not have any biases for this layer.
    """
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, 
                                    eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent. The 'training_data' is a list of tuples
        '(x, y)' representing the training inputs and the desired
        outputs. Epochs is the numer of times these batches are shuffled and 
        passed into the function. Eta is the learning rate. If 'test_data' is 
        provided then the network will be evaluated against the test data 
        after each epoch, and partial progress printed out.  This is useful 
        for tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, eta):
        grad_biases = [np.zeros(bias.shape) for bias in self.biases]
        grad_weights = [np.zeros(weights.shape) for weights in self.weights]
        for x, y in mini_batch:
            delta_grad_bias, delta_grad_weights = self.backprop(x, y)
            grad_biases = [gb+dgb for gb, dgb in zip(grad_biases, delta_grad_bias)]
            grad_weights = [nw+dnw for nw, dnw in zip(grad_weights, delta_grad_weights)]
        self.weights = [w-(eta/len(mini_batch))*gw
                        for w, gw in zip(self.weights, grad_weights)]
        self.biases = [b-(eta/len(mini_batch))*gb
                       for b, gb in zip(self.biases, grad_biases)]

    def backprop(self, x, y):
        """Return a tuple '(nabla_b, nabla_w)' representing the
        gradient for the cost function C_x.  'nabla_b' and
        'nabla_w' are layer-by-layer lists of numpy arrays, similar
        to 'self.biases' and 'self.weights'."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward pass. These lines of code will store the activations and 
        # z's of all of the nodes in the network 
        activation = x # This is a 784x1 vector
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # delta is the errors associated with the last layer. Recall, this is 
        # given by the cost functions derivative, multiplied by the derivative 
        # of the activation function
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta # amount to change each of the biases
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # nabla_w is the amount to change the weights. np.dot has vectorised 
        # this operation for us. Each weight is changed by: the error of the 
        # neuron it feeds in to, multiplied by the activation of the one it 
        # leaves from.
         
        # At this point, delta is the error vector of the last layer. It gets 
        # updated recursively throughout this for loop.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
# This is the activation funtion of choice, it could be swapped for other 
# better ones like ReLu, but this is what we work with for now.
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
