from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Define the neural network
def create_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(16, input_dim=input_size, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(output_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer='adam')
    return model
