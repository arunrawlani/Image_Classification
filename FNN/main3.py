import data
import numpy as np 
import network2 

# get data
training_data, validation_data = data.data_parser()
training_data = list(training_data)
validation_data = list(validation_data)

print("Data ready.")

# ----------- Parameters determined by x-validation -----------
num_of_inst_for_train = len(training_data) # How many data points for training?
num_of_inst_for_valid = len(validation_data) # How many data points for validation?
neuron = 100 # number of neurons
lbda = 5.0 # dlambda value
lr = 0.1 # learning rate
epochs = 30 # number of epochs
mb_size = 10 # mini batch size

net = network2.Network([12288, neuron, 40], cost=network2.CrossEntropyCost)
net.SGD(training_data[:num_of_inst_for_train], epochs, mb_size, lr, lmbda = lbda, evaluation_data = validation_data[:num_of_inst_for_valid], monitor_evaluation_accuracy = True, monitor_evaluation_cost= True,
	monitor_training_accuracy = False, monitor_training_cost=False)