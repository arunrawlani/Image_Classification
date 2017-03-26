import data
import numpy as np 
import network2 

# get data
training_data, validation_data = data.data_parser()
training_data = list(training_data)
validation_data = list(validation_data)

print("Data ready.")

# ----------- Some parameters for x-validation -----------
num_of_inst_for_train = 2000 # How many data points for training?
num_of_inst_for_valid = 200 # How many data points for validation?
p_neurons = [30,40,50] # different number of neurons
p_lambda = [5.0] # different lambda values
p_learn_rate = [0.1, 1.0] # different learning rates
p_epochs = [5,10] # different number of epochs
p_minibatch_size = [10] # mini batch size

# Try out each possible permutation of the above parameters
for neuron in p_neurons:
	for lbda in p_lambda:
		for lr in p_learn_rate:
			for epochs in p_epochs:
				for mb_size in p_minibatch_size:
					# -------- This is a configuration to try out --------
					print("*** neuron = {}, lbda = {}, lr = {}, epochs = {}, mb_size = {} ***".format(neuron, lbda, lr, epochs, mb_size))
					net = network2.Network([12288, neuron, 40], cost=network2.CrossEntropyCost)
					net.SGD(training_data[:num_of_inst_for_train], epochs, mb_size, lr, lmbda = lbda, evaluation_data = validation_data[:num_of_inst_for_valid], monitor_evaluation_accuracy = True, monitor_evaluation_cost= True,
						monitor_training_accuracy = False, monitor_training_cost=False)
					print("") # print new line
