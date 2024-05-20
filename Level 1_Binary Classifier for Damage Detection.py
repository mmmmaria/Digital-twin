# Level 1_Binary Classifier for Damage Detection, based on 1D CNNs

#Imports
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
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-6 * 10**(epoch / 5))

"""
#Case Study
The case study consists of a 6-storey metal tower, monitoring the displacements at each of the 6 storeys (d1, d2, d3, d4, d5, d6), 
the force applied to the tower (F), and the time of damage since the bolts loosening began (t). 'label' represents a healthy state if 0 or a damaged state if 1.
"""

#Data
dataset = pd.read_csv("data.csv") 
dataset=shuffle(dataset)

dataset['F'] = round(dataset['F'],6)
dataset['d1'] = round(dataset['d1'] , 3)
dataset['d2'] = round(dataset['d2'] , 3)
dataset['d3'] = round(dataset['d3'] , 3)
dataset['d4'] = round(dataset['d4'] , 3)
dataset['d5'] = round(dataset['d5'] , 3)
dataset['d6'] = round(dataset['d6'] , 3)

dataset=dataset.values
X = dataset[:,[d1,d2,d3,d4,d5,d6]].astype(float)
Y = dataset[:,label]

dataset=shuffle(dataset)

#Data for the Test
dataTest = pd.read_excel("datos_TEST.xlsx") 
dataTest= shuffle(dataTest)
dataTest_values = dataTest.values
X_test0 = dataTest_values[:,[d1,d2,d3,d4,d5,d6]] 
y_test0 = dataTest_values[:,label] 


scaler = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler.fit_transform(X)
X=X_scaled
X_test0 = scaler.transform(X_test0)

rows,columns=X.shape
rows0,columns0=X_test0.shape

X=X.reshape(rows,columns,1)
X_test0=X_test0.reshape(rows0,columns0,1)

#Splitting testing and test set
N=rows
split1=int(0.8*N)
split2=int(0.99*N)

# Split the dataset into training and testing sets
X_train, X_test, X_valid = X[:split1], X[split1:split2], X[split2:]
y_train, y_test, y_valid = Y[:split1], Y[split1:split2], Y[split2:]

#Model Architecture
# define model architecture : 1DCNN-classification
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
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
#tf.keras.Input(shape=None,
#Inp = Input(shape=(X.shape[1],1))
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

x = Dense(64, activation='relu')((x)) #63
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs = Inp, outputs=x)
# summary of the model
model.summary()

#Metrics
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

# Choose Hyperparameters and compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['acc', precision_m, recall_m])

#Training
# train the model
bsize = 128 #64, 50
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),  epochs=300, batch_size = bsize, verbose=2, shuffle = True, callbacks = [epoch_schedule])
print("Accuracy in Train:", 'acc')
print("Precision in Train:", 'precision_m')
print("Accuracy in Validation:", 'val_acc')
print("Precision in Validation:", 'val_precision_m')

# Plot Accuracy and Loss
print("Accuracy in Train:", round(history.history['acc'][-1], 3))
print("Precision in Train:", round(history.history['precision_m'][-1],3))
print("Recall in Train:", round(history.history['recall_m'][-1],3))
print("Accuracy in Validation:", round(history.history['val_acc'][-1],3))
print("Precision in Validation:", round(history.history['val_precision_m'][-1],3))
print("Recall in Validation:", round(history.history['val_recall_m'][-1],3))

#Summarize history for loss
plt.figure(figsize=(20,5))
plt.plot(history.history['loss'],'-o')
plt.plot(history.history['val_loss'],'-s')
plt.title('Loss Curve',fontsize=20)
plt.ylabel('Loss',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Number of epochs',fontsize=18)
plt.legend(['train', 'valid'], loc='lower right',fontsize=18)
plt.show()

#---Summarize history for accuracy---
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

y_predicted = model.predict(X_test, verbose = 2)
y_actual = y_test
y_predict = np.round(y_predicted)
y_act = y_actual
cm = confusion_matrix(y_act, y_predict)

cm = confusion_matrix(y_actual, y_predict)
cm_labels = ['Health','Damaged']  #'0' is 'Health', '1' is 'Damage'
plot_confusion_matrix(cm, classes=cm_labels, title='Confusion matrix')

# Plot Normalized confusion matrix
plot_confusion_matrix(cm, classes=cm_labels, normalize = True, title='Confusion matrix')

# Classification Report
print('Classification Report:')
print(classification_report(y_act, y_predict, target_names = cm_labels))

# AUC-ROC curve
y_actual = np.expand_dims(y_actual, axis=1)
n_classes = 1
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    print(y_actual[:, i])

# Compute ROC curve and ROC area for each class
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
n_classes = 1
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_actual[:, i], np.round(y_predicted[:,i]))
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for all classes
plt.figure(figsize=(6,6))
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

y_predicted0 = model.predict(X_test0, verbose = 2)
y_actual0 = y_test0
y_predict0 = np.round(y_predicted0)
y_act0 = y_actual0
cm = confusion_matrix(y_act0, y_predict0)

cm = confusion_matrix(y_actual0, y_predict0)
cm_labels = ['Health','Damaged'] #'0' is 'Health', '1' is 'Damage'
plot_confusion_matrix(cm, classes=cm_labels, title='Confusion matrix')

# Plot Normalized confusion matrix
plt.rcParams.update({'font.size': 12})
plot_confusion_matrix(cm, classes=cm_labels, normalize = True, title='Confusion matrix')
# Classification Report
print('Classification Report:')
print(classification_report(y_act0, y_predict0, target_names = cm_labels))

y_actual0 = np.expand_dims(y_actual0, axis=1)
# Compute ROC curve and ROC area for each class
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

n_classes = 1
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_actual0[:, i], np.round(y_predicted0[:,i]))
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for all classes
plt.figure(figsize=(6,6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('ROC curve',fontsize=18)
    cm_labels = ['Damage','Baseline']
    plt.legend(cm_labels,loc="lower right",fontsize=18)

# Plot of a ROC curve for all classes
plt.figure(figsize=(4,4))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    #plt.plot([0, 0.2], [0.8, 1.0], 'k--')
    plt.xlim([0.0, 0.2])
    plt.ylim([0.8, 1.0])
     
    plt.xticks(np.arange(0, 0.21, step=0.04),fontsize=12)
    plt.yticks(np.arange(0.8, 1.01, step=0.02), fontsize=12)
        
    plt.grid(True)
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    plt.title('ROC curve',fontsize=18)
    cm_labels = ['Damage','Baseline']
    plt.legend(cm_labels,loc="lower right",fontsize=12)
