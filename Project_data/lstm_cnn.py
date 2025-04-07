import keras
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Bidirectional, Flatten, TimeDistributed


# Load the new dataset
data = pd.read_excel("new.xlsx")

# Preprocessing
data['time'] = pd.to_datetime(data.time, format="%Y-%m-%d")
data.index = data['time']

# Feature scaling
scaler = MinMaxScaler(feature_range=(0,1))

# Selecting relevant features and labels (alert column is the label)
features = ['降雨强度', '累积雨量', 'mud_level', '高程', '相对高差',
            '土壤类型', '纵坡降', '流域面积', '泥水位变化速率', 'month', 'season',
            '上一个窗口的平均水位','max']
target = 'alert'

# Normalize the features
data[features] = scaler.fit_transform(data[features])

# Prepare the dataset for model input
X = []
Y = []
window_size = 50
for i in range(0, len(data) - window_size - 1, 1):
    temp = []
    for feature in features:
        temp.append(data[feature][i:i+window_size].values)
    X.append(np.array(temp).T)  # Shape (50, num_features)
    Y.append(data[target][i + window_size])

# Train-test split
x_train, x_test, train_label, test_label = train_test_split(X, Y, test_size=0.2, shuffle=True)

train_X = np.array(x_train)
test_X = np.array(x_test)
train_label = np.array(train_label)
test_label = np.array(test_label)

train_X = train_X.reshape(train_X.shape[0], 1, window_size, len(features))
test_X = test_X.reshape(test_X.shape[0], 1, window_size, len(features))

# Model creation
model = keras.Sequential()
model.add(TimeDistributed(Conv1D(128, kernel_size=1, activation='relu', input_shape=(None, window_size, len(features)))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(256, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(512, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(200, return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid since it's a binary classification problem
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

# Model training
model.fit(train_X, train_label, validation_data=(test_X, test_label), epochs=40, batch_size=64, shuffle=False)

# Testing and prediction
predicted = model.predict(test_X)
predicted = np.round(predicted).astype(int)  # Round predictions to 0 or 1

# Visualizing the results
plt.plot(test_label, label='True Alert', color='blue')
plt.plot(predicted, label='Predicted Alert', color='green')
plt.title('Mudslide Risk Prediction')
plt.xlabel('Time')
plt.ylabel('Alert (0: No Risk, 1: Risk)')
plt.legend()
plt.show()
