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
