#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy
import scipy.special
import csv
import matplotlib.pyplot as plt
import pandas as pd


class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        pass

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (
            self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (
            self.onodes, self.hnodes))

    def train(self, inputs_list, targets_list):

        inputs = numpy.array(inputs_list, ndmin=2).T  # creates array of
        # inputs_list values with minimum 2 dimensions and returns the array,
        # transposed 
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)  # for 2-D arrays it is
        # equivalent to matrix multiplication, and for 1-D arrays to inner
        # product of vectors
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs))

        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs))
        pass

    # query for the neural network
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

performance_csv = []  # list to store all the results of the learning


def frange(start, end, step):
    tmp = start
    while(tmp < end):
        yield tmp
        tmp += step

for hid_n in range(10, 140, 20):  # hidden_nodes
    for lr in frange(0.1, 1.0, 0.1):  # learning_rate
        # The network will be taught to read 28px x 28px bitmaps which have
        # numbers from 0 to 9, so it is required to set a relevant number 
        # of input nodes, hidden_nodes and  output nodes
        input_nodes = 784
        hidden_nodes = hid_n
        output_nodes = 10
        learning_rate = lr

        # create an instance of the neural network and name it 'n'
        n = neuralNetwork(
            input_nodes, hidden_nodes, output_nodes, learning_rate)

        # upload dataset from given.csv to training_data_file variable
        # populate training_data_list with data
        training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
        training_data_list = training_data_file.readlines()
        training_data_file.close()

        # Training of the neural network
        # Set the epochs variable - number of time training data is to be 
		# used in teaching the network. From 1 to 9

        epochs = 8

        for e in range(epochs):
            # go through all records in training_data_list
            for record in training_data_list:
                # split the record by the',' commas
                all_values = record.split(',')
                # scale and shift the inputs
                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values
                targets = numpy.zeros(output_nodes) + 0.01

                # convert all_values[0] to integer and make them the target
                # label for this record
                targets[int(all_values[0])] = 0.99
                n.train(inputs, targets)
                pass
            pass

        # record test data of the .csv file to the list
        test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()

        # get first test record
        all_values = test_data_list[0].split(',')

        # Display the correct value for the training data
		# (this is the first value in the mnist dataset)
        # print(training_data_list[0])

        # draw the variable
        image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
        fig = plt.imshow(image_array, cmap='Greys', interpolation='None')
        fig.figure.savefig('main_run_number.jpg')
        # plt.show(fig)

        # check the value of the network query
        n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)

        # Test of the neural network
        # we will create a scorecard to check how well the network works,
        # in the beginning we will create an empty list with the name scorecard
        scorecard = []

        # get through all the records of the test data set
        for record in test_data_list:
            # connect records by comma
            all_values = record.split(',')

            # the correct value is the first one
            correct_label = int(all_values[0])
            # print(correct_label, "Correct value")

            # Assign input data from the list to inputs variable
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # query the network
            outputs = n.query(inputs)

            # the index with the highest value corresponds to the label
            label = numpy.argmax(outputs)
            # print(label, 'network response')

            # add to the list of results if the next one is correct,
            # or incorrect
            if (label == correct_label):
                # response is correct -> write 1 to the scorecard
                scorecard.append(1)
            else:
                # response is wrong -> write 0 to the scorecard
                scorecard.append(0)
                pass
            pass

        # print(scorecard)

        # policz wskaźnik wydajności jest to suma wyników prawidłowych
        # i nieprawidłowych  podzielonych przez liczbę poszczególnych prób
        # rozpoznania znaku
		
		# count the performance indicator, it is the sum of the correct 
        # and wrong results divided by the number of individual attempts
        # of character recognition
        scorecard_array = numpy.asarray(scorecard)
        performance = round(
            (scorecard_array.sum() / float(scorecard_array.size)), 2)
        print ("performance = %s" % performance)

        temporary = []  # make up list with results one run
        temporary.append(performance)
        temporary.append(round(learning_rate, 1))
        temporary.append(hidden_nodes)
        temporary.append(epochs)

        performance_csv.append(tuple(temporary))

# print ("performance_csv: %s" % performance_csv)

# create pandas dataframe to record the results
labels = ['Performance', 'Learning_rate', 'Hidden_nodes', 'Epochs']
df = pd.DataFrame.from_records(performance_csv, columns=labels)
print(df)

# save dataframe to cvs file
df.to_csv('performance_csv_main_run.csv', index=False, encoding='utf-8')

# visualize data and save image
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('Records')
ax1.set_ylabel('Performance rate')
ax1.set_title('Learning digits')
df['Performance'].plot(kind='line')
fig.savefig('Performance.png')
plt.show()
