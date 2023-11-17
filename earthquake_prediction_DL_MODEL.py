import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, concatenate, Dense, Dropout, Activation, Permute, Multiply
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Load the earthquake dataset
data = pd.read_csv('earthquake_data.csv')

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size, :]
test_data = data_scaled[train_size:, :]

# Define the input shape
input_shape = (train_data.shape[1]-1, 1)

# Define the CNN model
input_cnn = Input(shape=input_shape)
conv1 = Conv1D(filters=64, kernel_size=5, activation='relu')(input_cnn)
max_pool1 = MaxPooling1D(pool_size=2)(conv1)
conv2 = Conv1D(filters=128, kernel_size=5, activation='relu')(max_pool1)
max_pool2 = MaxPooling1D(pool_size=2)(conv2)
cnn_output = Dropout(0.2)(max_pool2)

# Define the BiLSTM model
input_lstm = Input(shape=input_shape)
lstm1 = Bidirectional(LSTM(units=64, return_sequences=True))(input_lstm)
lstm2 = Bidirectional(LSTM(units=128))(lstm1)
lstm_output = Dropout(0.2)(lstm2)

# Concatenate the outputs of the CNN and BiLSTM models
merged = concatenate([cnn_output, lstm_output])

# Add the attention mechanism
attention = Dense(1, activation='tanh')(merged)
attention = Permute((2, 1))(attention)
attention = Activation('softmax')(attention)
attention = Permute((2, 1))(attention)
attention = Multiply()([merged, attention])
attention_output = attention

# Add the output layer
output = Dense(1, activation='linear')(attention_output)

# Create the model
model = Model(inputs=[input_cnn, input_lstm], outputs=output)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit([train_data[:, 1:].reshape(train_data.shape[0], train_data.shape[1]-1, 1), train_data[:, 1:].reshape(train_data.shape[0], train_data.shape[1]-1, 1)],
                    train_data[:, 0].reshape(train_data.shape[0], 1),
                    validation_split=0.2,
                    epochs=100,
                    batch_size=64,
                    callbacks=[es],
                    verbose=1)

# Evaluate the model on the testing set
test_loss = model.evaluate([test_data[:, 1:].reshape(test_data.shape[0], test_data.shape[1]-1, 1), test_data[:, 1:].reshape(test_data.shape[0], test_data.shape[1]-1, 1)],
                           test_data[:, 0].reshape(test_data.shape[0], 1),
                           verbose=1)

print(f'Test loss: {test_loss}')