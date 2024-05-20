# Level 2_Multiclass Classifier for Damage Location based on 1D-CNNs

#imports
import os
import random 
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv1D, MaxPooling1D, Dropout, Lambda, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import to_categorical,plot_model
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import *
from sklearn.preprocessing import MinMaxScaler

#Callbacks
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('val_acc')>0.99) and (logs.get('acc')>0.99) and (logs.get('val_loss')<0.07) and (logs.get('loss')<0.07):
      print("\nReached perfect accuracy so cancelling training!")
      self.model.stop_training = True
epoch_schedule = myCallback()

"""
#Case Study
The case study consists of a 6-storey metal tower, monitoring the displacements at each of the 6 storeys (d1, d2, d3, d4, d5, d6), 
the force applied to the tower (F), and the time of damage since the bolts loosening began (t).
"""

#Data
### Dataset
dataframe=pd.read_csv('data.csv')
dataset=dataframe.to_numpy()
dataset = shuffle(dataset)

X = dataset[:,d1,d2,d3,d4,d5,d6].astype(float)
Y = dataset[:,label1,label2,label3,label4,label5,label6].astype(int) #Each label corresponds to each floor and is '0' if healthy and '1' if damaged

# load DATEST 
dataTest = pd.read_excel("data_TEST.xlsx") 
dataTest= shuffle(dataTest)
dataTest_values = dataTest.values
X_test0 = dataTest_values[:,[d1,d2,d3,d4,d5,d6]] 
y_test0 = dataTest_values[:,[label1,label2,label3,label4,label5,label6]] 

scaler = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler.fit_transform(X)
X=X_scaled
X_test0 = scaler.transform(X_test0)
rows,columns=X.shape
rows0,columns0=X_test0.shape
X=X.reshape(rows,columns,1)
X_test0=X_test0.reshape(rows0,columns0,1)

# Splitting testing and test set
N=rows
split1=int(0.8*N)
split2=int(0.9*N)

# Split the dataset into training and testing sets
X_train, X_valid, X_test = X[:split1], X[split1:split2], X[split2:]
y_train, y_valid, y_test = Y[:split1], Y[split1:split2], Y[split2:]

# Model Architecture
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

Inp = Input(shape=(6,1))
x1 = Sequential()

x1 = Conv1D(filters=32, kernel_size=3, input_shape=(6,1))(Inp) #32
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

x = Dense(128, activation='relu')((x)) #63
x = Dense(64, activation='relu')(x) #54
x = Dropout(0.25)(x)
x = Dense(6, activation='softmax')(x)

model = Model(inputs = Inp, outputs=x)

# summary of the model
model.summary()

# Functions for recall, precision and f1 score
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Choose Hyperparameters and compile the model
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['acc', precision_m, recall_m])

#Training
bsize = 128 
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size = bsize, verbose=2, shuffle = False)

#Plot Accuracy and Loss
#Summarize history for loss
plt.figure(figsize=(20,5))
plt.plot(history.history['loss'],'-o')
plt.plot(history.history['val_loss'],'-s')
plt.title('Loss Curve',fontsize=20)
plt.ylabel('Loss',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Number of epochs',fontsize=18)
plt.legend(['train', 'valid'], loc='upper right',fontsize=18)
plt.show()

#Summarize history for accuracy
plt.figure(figsize=(20,5))
plt.plot(history.history['acc'],'-o')
plt.plot(history.history['val_acc'],'-s')
plt.title('Accuracy Curve',fontsize=20)
plt.ylabel('Accuracy',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Number of epochs',fontsize=18)
plt.legend(['train', 'valid'], loc='lower right',fontsize=18)
plt.show()

# Predictions
from sklearn.utils.multiclass import unique_labels
import itertools
import matplotlib.pyplot as plt
from keras.utils import to_categorical
fig = plt.gcf()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i,j],
      horizontalalignment = 'center',
      color = "white" if cm[i,j] > thresh else "black")

    fig.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')

y_predicted = model.predict(X_valid, verbose = 2)
y_actual = y_valid
y_predict = model.predict(X_valid, verbose = 2)
y_predict = np.argmax(y_predict, axis=1)
y_act = np.argmax(y_valid, axis=1)  # converting one hot representation back to numerical data
cm = confusion_matrix(y_act, y_predict)

# Plot Non-Normalized confusion matrix
cm_labels = ['d1','d2','d3','d4','d5','d6']
plot_confusion_matrix(cm, classes=cm_labels, title='Confusion matrix')

# Plot Normalized confusion matrix
plot_confusion_matrix(cm, classes=cm_labels, normalize = True, title='Confusion matrix')

# Classification Report
print('Classification Report')
print(classification_report(y_act, y_predict, target_names = cm_labels))

# AUC-ROC curve
# Compute ROC curve and ROC area for each class
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

n_classes = 6
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_actual[:, i], np.round(y_predicted[:,i]))
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for all classes
plt.figure(figsize=(8,8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Receiver operating characteristic example',fontsize=18)
plt.legend(cm_labels,loc="lower right",fontsize=18)

# Testing with dataTest
y_predicted0 = model.predict(X_test0, verbose = 2)
y_actual0 = y_test0

y_predict0 = model.predict(X_test0, verbose = 2)
y_predict0 = np.argmax(y_predict0, axis=1)

y_act0 = np.argmax(y_test0, axis=1)  # converting one hot representation back to numerical data

cm = confusion_matrix(y_act0, y_predict0)

# Plot Non-Normalized confusion matrix
cm_labels = ['d1','d2','d3','d4','d5','d6']
plot_confusion_matrix(cm, classes=cm_labels, title='Confusion matrix')

# Plot Normalized confusion matrix
plot_confusion_matrix(cm, classes=cm_labels, normalize = True, title='Confusion matrix')

# Classification Report
print('Classification Report:')
print(classification_report(y_act0, y_predict0, target_names = cm_labels))

y_actual0=np.squeeze(y_actual0)
# Compute ROC curve and ROC area for each class
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

n_classes = 6
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_actual0[:, i], np.round(y_predicted0[:,i]))
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for all classes
plt.figure(figsize=(8,8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], 'k--', label='reference')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Receiver operating characteristic example',fontsize=18)
plt.legend(cm_labels,loc="lower right",fontsize=18)
