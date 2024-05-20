# Common imports
import numpy as np
from scipy.stats import wasserstein_distance

import os
from backend import import_excel, export_excel
# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
# style.use('bmh')
from mpl_toolkits.mplot3d import Axes3D
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import pandas as pd
import tensorflow as tf
import keras
import random
import sys
sys.path.append("..")
import network
import WGAN_Model
from WGAN_Model import fid_score

import time
timestr=time.strftime("%Y%m%d-%H%M")

import warnings
warnings.filterwarnings("ignore")

############################################################################
############################################################################

j = 4   #CAMBIAR J CADA VEZ QUE EJECUTE EL SCRIPT

############################################################################
############################################################################
# Setting the datasets

scenario = "heter"
n_features = 2

df1 = pd.read_excel("datos_sano.xlsx")
dataset1 = df1.values
n_instance1=len(dataset1)
x1 = dataset1[:n_instance1,2] # Fuerza (N) = [0]; Ruido de la Fuerza (N) = [2]
y1 = dataset1[:n_instance1,3] #D6 (mm)
X_train, y_train = x1,y1

df2 = pd.read_excel("datos_sano_original.xlsx")
dataset2 = df2.values
n_instance2=len(dataset2)
x2 = dataset2[:n_instance2,2] # Ruido de la Fuerza (N); Ruido de la Fuerza (N) = [2]
y2 = dataset2[:n_instance2,3] #D6 (mm)
X_test, y_test = x2,y2

if not os.path.exists('Dataset'):
    os.system('mkdir Dataset')
if not os.path.exists('GANS'):
    os.system('mkdir GANS')
if not os.path.exists('GANS/Models'):                
    os.makedirs ('GANS/Models')
if not os.path.exists('GANS/Losses'):                
    os.makedirs ('GANS/Losses')    
if not os.path.exists('GANS/Random_test'):   
    os.makedirs ('GANS/Random_test') 
    
export_excel(X_train, 'Dataset/X_train')
export_excel(y_train, 'Dataset/y_train')
export_excel(X_test, 'Dataset/X_test')
export_excel(y_test, 'Dataset/y_test')

X_train = import_excel('Dataset/X_train')
y_train = import_excel('Dataset/y_train')
X_test = import_excel('Dataset/X_test')
y_test = import_excel('Dataset/y_test')

print('made dataset')

############################################################################
############################################################################
# Preprocessing

n_var = 2
latent_spaces = [3,10,50,100]
latent_space = 10 #int(latent_spaces[int(vars[1,j])])
batchs = [10,100,1000]
BATCH_SIZE = 100 #int(batchs[int(vars[2,j])])
scales = ['-1-1','0-1']
scaled = '-1-1'#scales[int(vars[3,j])]
epochs = 500   #[1000,10000,10000]
# epoch = int(epochs[int(vars[4,j])])
bias = [True,False]
use_bias = True #(bias[int(vars[5,j])])

############################################################################
############################################################################
#Training
wgan = WGAN_Model.WGAN(n_features,latent_space,BATCH_SIZE,n_var,use_bias)
train_dataset, scaler, X_train_scaled = wgan.preproc1(X_train, y_train, scaled)
hist = wgan.train(train_dataset, epochs, scaler, scaled, X_train, y_train)
wgan.generator.save('GANS/Models/GAN_'+str(j))

# plot loss
print('Loss: ')
fig, ax = plt.subplots(1,1, figsize=[10,5])
ax.plot(hist)
plt.title("Loss")
ax.legend(['loss_gen', 'loss_critic'])
ax.set_yscale('log') #OJO
ax.grid()
plt.tight_layout()
plt.savefig('GANS/Losses/GANS_loss'+ timestr +'.png')
generator = keras.models.load_model('GANS/Models/GAN_'+str(j))
plt.close()

latent_values = tf.random.normal([1000, latent_space], mean=0.0, stddev=0.1)
predicted_values=wgan.generator.predict(tf.convert_to_tensor(latent_values))
predicted_values=predicted_values.reshape(1000, n_features)

if scaled == '-1-1':
    predicted_values = scaler.inverse_transform(predicted_values)
elif scaled =='0-1':
    predicted_values = scaler.inverse_transform(predicted_values)

#Metrics: Wasserstein distance and FID
y_train=y_train[1:,0]
X_train=X_train[1:,0]

distance = wasserstein_distance(y_train, predicted_values[:,1])
print("Waserstein distance of the Predicted Training", distance)

xx1= predicted_values[:,1]
y_train=y_train.reshape(-1,1)
xx1=xx1.reshape(-1,1)
fid=fid_score(y_train,xx1)
print("Frechet Inception Distance (FID) of the Training", fid)

#Plot
predicted_values = predicted_values[1:,:]
plt.clf()
plt.plot(X_train,y_train,'o', label="Train values")
plt.plot(predicted_values[:,0],predicted_values[:,1],'o', label="Generated values")
plt.title("Training Predictions after " + str(epochs) + " epochs")
plt.legend(loc='lower right')
texto1= "Wasserstein distance of the Predicted Training: " + str(distance)
plt.text(-35, 37, texto1, fontsize=8)
texto2="Frechet Inception Distance (FID) of the Training: " + str(fid)
plt.text(-35, 35, texto2, fontsize=8)  #0.25 F, -35 N
plt.tight_layout()
plt.savefig("predictions_train" + '_' + str(j) + '_' + timestr +".png")


############################################################################
############################################################################
#Prediction

test_dataset, scaler, X_test_scaled = wgan.preproc2(X_test, y_test, scaled)

X_generated = wgan.predict(X_test_scaled, scaler)

#Metrics: Wasserstein distance and FID
y_test=y_test[1:,0]
X_test=X_test[1:,0]
    
distance = wasserstein_distance(y_test,X_generated[:,1])
print("Waserstein distance of the Predicted Test", distance)

xx2= X_generated[:,1]
y_test=y_test.reshape(-1,1)
xx2=xx2.reshape(-1,1)
fid=fid_score(y_test,xx2)
print("Frechet Inception Distance (FID)", fid)

plt.clf()
# plt.title("Prediction at x = -1, 0, 0.5, 1.5")
#plt.scatter(X_train, y_train, label="Training data")
plt.scatter(X_test, y_test, label="Test data")
#plt.scatter(predictthis[:,0], predictthis[:,1], label="Sample data", c="pink")
plt.scatter(X_generated[:,0], X_generated[:,1], label="Prediction")
plt.title("Test Predictions")
plt.legend(loc='lower right')
plt.tight_layout()
texto1= "Wasserstein distance of the Predicted Test: " + str(distance)
plt.text(-35, 37, texto1, fontsize=8)  #0.25 F, -35 N
texto2="Frechet Inception Distance (FID) of the Test: " + str(fid)
plt.text(-35, 35, texto2, fontsize=8)
# plt.xlabel("x")
# plt.ylabel("y")
plt.savefig("predictions_test" + '_' + str(j) + '_' + timestr +".png")

plt.clf()
plt.plot(X_train,y_train,'o', label="Train values", color="green", alpha= 0.2)
plt.scatter(X_test, y_test, label="Test data")
plt.scatter(X_generated[:,0], X_generated[:,1], label="Prediction")
plt.title("Training data, Test Data and Test Predictions")
plt.legend(loc='lower right')
plt.tight_layout()
# plt.xlabel("x")
# plt.ylabel("y")
plt.savefig("training and test" + '_' + str(j) + '_' + timestr +".png")


export_excel(X_generated, 'Dataset/X_generated')

