import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    #print(series)
    #print(window_size)
    for i in range(len(series) - window_size):
        X.append(series[i: i + window_size])
        y.append(series[i + window_size])
        #print(i)
        print(series[i: i + window_size])
        print(series[i + window_size])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size): #so this just outputs a model
    model = Sequential()
    model.add(LSTM(128, input_shape = (None, window_size), return_sequences = True))
    model.add(LSTM(128, return_sequences = False))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    from string import punctuation
    punctuation_keep = ['!', ',', '.', ':', ';', '?']
    keep_text = ''
    for char in text:
        if char not in punctuation or char in punctuation_keep:
            keep_text += char.lower()
        else:
            keep_text += ' '
        
    return keep_text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    idx = window_size
    while idx <= len(text):
        inputs.append(text[idx-window_size:idx])
        outputs.append(text[idx])
        
        #print(idx-window_size,':',idx)
        #print(text[idx-window_size:idx], '-', text[idx])

        idx += 5

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars), return_sequences = True))
    model.add(Dense(num_chars))
    model.add(Dense(num_chars, activation = 'softmax'))

    return model
