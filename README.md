# Digit-recognition as part of university course

## The goal of the research was to determine the settings for the best network performance when recognizing numbers. 

### The initial settings were as follows:

- Epochs – 1
- Input_nodes - 784
- Hidden_nodes – 10
- Output_nodes – 10
- Learning_rate – 0.1

The first run of the code was with a for loop with a learning rate in the range (0.1, 1) with a step of 0.1. 

`for lr in frange (0.1, 1.0, 0.1):`

The highest performance results (0.87) were achieved with learning rates 0.2 and 0.4.
The lowest result was on the 0.7 learning index.

The next step was to use loops with input nodes and increase the number of epochs to 8. Hidden
nodes have been set in the range of 140 with a step of 20 nodes.

```
for hid_n in range(10, 140, 20): # hidden_nodes 
    for lr in frange(0.1, 1.0, 0.1): # learning_rate
```

The high values of the epochs and hidden node variables have resulted in a significant increase in the time to run all loops. All calculations were completed within 8 hours on Ubuntu 14.04 LTS. This amount of time was needed because a lot of calculations were with variable epochs and an external for loop with learning indicator was activated 7 times. 

After completing the tests, the network learned to read the bitmap with a resolution of 28px to 28px. 
Depending on the number of hidden nodes, the learning speed changes during 3-6 cycles.

The top 3 performance results were achieved with the learning rates 0.1 and 0.2 with the value of hidden nodes 110, 130.
The performance indicator never exceeded 0.97. The next stage of testing may be running a program with an increased number of input nodes.

### CONTACT

Please send you feedback to

  max.savin3@gmail.com
