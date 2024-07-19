# import mnist_loader
# import network
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network.Network([784, 30, 10])
# net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)
import mnist_loader 
training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 
import network2 
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
        evaluation_data=validation_data,monitor_evaluation_accuracy=True)