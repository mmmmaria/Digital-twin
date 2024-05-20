#Levels 3 and 4_Regressor for predicting t, based on 1D CNNs

#Imports
import keras
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statistics import median, mean

"""
#Case Study
The case study consists of a 6-storey metal tower, monitoring the displacements at each of the 6 storeys (d1, d2, d3, d4, d5, d6), 
the force applied to the tower (F), and the time of damage since the bolts loosening began (t).
"""

#Data
# load dataset
dataframe = pd.read_csv("data.csv", sep=',')
dataset = dataframe.values
dataset = shuffle(dataset)

# load DATEST de Test
dataTest = pd.read_excel("data_TEST.xlsx") 
dataTest= shuffle(dataTest)
dataTest_values = dataTest.values
X_test0 = dataTest_values[:,[F,d1,d2,d3,d4,d5,d6]] 
y_test0 = dataTest_values[:,t] 

# split into input (X) and output (Y) variables
X = dataset[:,[1,2,3,4,5,6,7]] # F,d1,d2,d3,d4,d5,d6
y = dataset[:,0] #t 

rows,columns=X.shape
N=rows
split1=int(0.80*N)
split2=int(0.90*N)

# Split the dataset into training and testing sets
X_train, X_valid, X_test = X[:split1], X[split1:split2], X[split2:]
y_train, y_valid, y_test = y[:split1], y[split1:split2], y[split2:]

scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
X_test0 = scaler.transform(X_test0)

X_train= X_train.reshape(-1,7,1)
X_valid= X_valid.reshape(-1,7,1)
X_test= X_test.reshape(-1,7,1)
X_test0= X_test0.reshape(-1,7,1)

# Model architecture
# define model architecture : 1DCNN-classification
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

Inp = Input(shape=(7,1))
x1 = Sequential()

x1 = Conv1D(filters=32, kernel_size=3, input_shape=(7,1))(Inp) #32
#x1 = MaxPooling1D(pool_size=2)(x1)

x1 = Conv1D(filters=32, kernel_size=3, activation='relu')(x1) #32
#x1 = MaxPooling1D(pool_size=2)(x1)

x1 = Conv1D(filters=32, kernel_size=2, activation='relu')(x1)
#x1 = MaxPooling1D(pool_size=2)(x1)

#x1 = Conv1D(filters=128, kernel_size=3, activation='relu')(x1)
#x1 = MaxPooling1D(pool_size=2)(x1)

#x1 = Conv1D(filters=256, kernel_size=3, activation='relu')(x1)
#x1 = MaxPooling1D(pool_size=2)(x1)

x1 = Flatten()(x1)

cnn1 = Model(Inp,x1)

from keras.layers import concatenate
x = cnn1.output

x = Dense(64, activation='relu')((x)) #63
x = Dense(32, activation='relu')(x) #54
x = Dense(1, activation=None)(x) 

model = Model(inputs = Inp, outputs=x)

# summary of the model
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])

# Train
epochs=6000
batch_size=128
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_size, verbose=2,shuffle = False)

# Evaluate model on test data
X_train=X_train.reshape(-1,7,1)
X_valid= X_valid.reshape(-1,7,1)
X_test= X_test.reshape(-1,7,1)
X_test0= X_test0.reshape(-1,7,1)

y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
y_test_pred = model.predict(X_test)
y_test_pred0 = model.predict(X_test0)

print('R2 score (train):', r2_score(y_train, y_train_pred))
print('R2 score (valid):', r2_score(y_valid, y_valid_pred))
print('R2 score (test):', r2_score(y_test, y_test_pred))
print('R2 score (test0):', r2_score(y_test0, y_test_pred0))

print('MAE (train):', mean_absolute_error(y_train, y_train_pred))
print('MAE (test):', mean_absolute_error(y_valid, y_valid_pred))
print('MAE (test):', mean_absolute_error(y_test, y_test_pred))
print('MAE (test0):', mean_absolute_error(y_test0, y_test_pred0))

print('MSE (train):', mean_squared_error(y_train, y_train_pred))
print('MSE (test):', mean_squared_error(y_valid, y_valid_pred))
print('MSE (test):', mean_squared_error(y_test, y_test_pred))
print('MSE (test0):', mean_squared_error(y_test0, y_test_pred0))

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss: Mean Squared Error(MSE)')
plt.ylabel('Loss = MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Plot training and validation loss
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Mean Absolute Error (MAE)')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot the regression results
plt.scatter(y_train, y_train_pred, label='train', color='blue')
plt.scatter(y_test, y_test_pred, label='test', color='orange')
plt.title('Model Regression')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.xlim(0,110)
plt.ylim(0,150)
plt.legend()
plt.axhline(y = 100, color = 'black', linestyle = '-', linewidth=0.25)
plt.axvline(x = 100, color = 'black', linestyle = '-', linewidth=0.25)
plt.plot([0,0],[100,100])
point1 = [0, 0]
point2 = [100, 100]
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
plt.plot(x_values, y_values)
plt.show()

plt.figure(figsize=(20,4))
plt.plot(y_test[:], marker="o", label="real")
plt.plot(y_test_pred[:], marker="+", label="pred")
plt.title('Y_test vs. Y_pred')
plt.ylabel('y')
plt.ylim(0,110)
plt.legend( loc='upper right')
plt.show()

# Plot R2 score, MAE, and MSE for both training and testing data

train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].bar(['Train', 'Test'], [train_r2, test_r2])
axs[0].set_title('R2 score')
axs[1].bar(['Train', 'Test'], [train_mae, test_mae])
axs[1].set_title('MAE')
axs[2].bar(['Train', 'Test'], [train_mse, test_mse])
axs[2].set_title('MSE')
plt.show()

# Plot the regression results
plt.scatter(y_test0, y_test_pred0, label='test', color='green')
plt.title('Model Regression')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.xlim(0,110)
plt.ylim(0,170)
plt.legend()
plt.axhline(y = 100, color = 'black', linestyle = '-', linewidth=0.25)
plt.axvline(x = 100, color = 'black', linestyle = '-', linewidth=0.25)
plt.plot([0,0],[100,100])
point1 = [0, 0]
point2 = [100, 100]
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
plt.plot(x_values, y_values)
plt.show()

X_test0= np.squeeze(X_test0, axis=2)
y_test0 = np.expand_dims(y_test0, axis=1)
array0 = np.concatenate((X_test0, y_test_pred0), axis=1)
dataframe0=pd.DataFrame(array0)

#Boxplot
condicion_0 = dataframe0[t] <=15
dataframe0_0 = dataframe0[condicion_0]
dataframe0_66 = dataframe0.loc[(dataframe0[t] > 15) & (dataframe0[7] < 80)]
dataframe0_100 = dataframe0.loc[(dataframe0[t] > 80) & (dataframe0[7] < 115)]

datos0 = dataframe0_0[t]
datos66= dataframe0_66[t]
datos100= dataframe0_100[t]

mediana0 = median(dataframe0_0[t])
mediana66 = median(dataframe0_66[t])
mediana100 = median(dataframe0_100[t])

datos = [datos0, datos66, datos100]
nombres = ['t=0', 't=66', 't=100']

#boxplots
fig, ax = plt.subplots()
ax.boxplot(datos, labels=nombres)

# horizontal lines
for i in range(0, 110, 10):
    ax.axhline(y=i, color='gray', linestyle='--', linewidth=0.5)

# Titles
plt.title('Time of the damage (t)')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.text(1.55, mediana0, f"Mediana: {mediana0}", ha='center', va='baseline')
plt.text(1.8, mediana66, f"Mediana: {mediana66}", ha='right', va='baseline')
plt.text(2.82, mediana100, f"Mediana: {mediana100}", ha='right', va='baseline')
plt.yticks(np.arange(0, 101, 10))
plt.show()


