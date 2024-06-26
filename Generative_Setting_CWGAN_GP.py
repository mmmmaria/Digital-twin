"""
## Generative Setting (CWGAN-GP) ##
"""

#Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
from tensorflow.keras.layers import Dense, concatenate, Input
from sklearn.utils import shuffle

from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim

import scipy
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import MinMaxScaler

# define model architecture : 1DCNN-classification
from keras.layers import Input,concatenate
from keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input
from keras.models import Model

"""
#Case Study
The case study consists of a 6-storey metal tower, monitoring the displacements at each of the 6 storeys (d1, d2, d3, d4, d5, d6), 
the force applied to the tower (F), and the time of damage since the bolts loosening began (t).
"""

#Data
## Variables ##
j = 1                  #trial number
epoch_num = 8          # total epoch to run
BATCH_SIZE = 100       # batch size of train set
noise_dim = 25         # dimension of noise vector for generator
condition_dim = 6      # dimension of condition vector for generator
gen_dim = 8            # dimension of generator's output vector
D_cycle = 5            # train disctriminator "D_cycle" times in one epoch, number of critic iterations per epoch
steps_show = 1         # update figure per "steps_show" epoches

#"inp" = "input" = "condition" = (label1, label2, label3, label4, label5, label6)
#"out" = "output" = "target" = (d1,d2,d3,d4,d5,d6,F,t)

dataframe=pd.read_csv('data.csv')
dataframe = shuffle(dataframe)
inp = dataframe[['label1','label2','label3','label4','label5','label6']]
inp=np.array(inp, dtype=np.float32)
out = dataframe[['d1','d2','d3','d4','d5','d6','F', 't']]
out=np.array(out, dtype=np.float32)
data_real = tf.concat([inp, out], axis=1)
scaler = MinMaxScaler(feature_range=(0,1))
out_scaled = scaler.fit_transform(out)

#Training
data = tf.concat([inp, out_scaled], axis=1)
train_dataset = tf.data.Dataset.from_tensor_slices(data[:,:]).batch(BATCH_SIZE)
inp_condition = Input(shape=[condition_dim, 1], name='condition_G')
inp_noise = Input(shape=[noise_dim, 1], name='noise')
X = concatenate([inp_condition, inp_noise], axis=1)

## Generator and Discriminator ##
def Generator():
    inp_condition = Input(shape=[condition_dim,1 ], name='condition_G')
    inp_noise = Input(shape=[noise_dim,1 ], name='noise')
    X = concatenate([inp_condition, inp_noise], axis=1)
    
    X = Conv1D(filters = 32, kernel_size = 3)(X)
    #X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=2)(X)
    
    X = Conv1D(filters = 32, kernel_size = 3)(X)
    #X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=2)(X)
         
    X = Conv1D(filters = 32, kernel_size = 2)(X)
    #X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=2)(X)
    
    X = Flatten()(X)
    
    X = Dense(64, activation='relu')(X)
    #X = BatchNormalization()(X)
    X = Dense(32, activation='relu')(X)
    #X = BatchNormalization()(X)
    
    last = Dense(gen_dim, activation="tanh")(X)
    return tf.keras.Model(inputs=[inp_condition, inp_noise], outputs=last, name='Generator')
    
def Discriminator():
    inp_condition = Input(shape=[condition_dim, 1], name='condition_D')
    inp_target = tf.keras.layers.Input(shape=[gen_dim,1], name='target')
    X = concatenate([inp_condition, inp_target], axis=1)
        
    X = Conv1D(filters = 32, kernel_size = 3)(X)
    #X = BatchNormalization()(X)
    X = Activation('LeakyReLU')(X)
    X = Dropout(0.3)(X)
    
    X = Conv1D(filters = 32, kernel_size = 3)(X)
    #X = BatchNormalization()(X)
    X = Activation('LeakyReLU')(X)
    #X = Dropout(0.3)(X)
           
    X = Conv1D(filters = 32, kernel_size = 2)(X)
    #X = BatchNormalization()(X)
    X = Activation('LeakyReLU')(X)
    #X = Dropout(0.3)(X)
    
    X = Flatten()(X)
        
    last = Dense(1)(X)
    return tf.keras.Model(inputs=[inp_condition, inp_target], outputs=last, name='Discriminator')   

generator = Generator()
discriminator = Discriminator()
generator.summary()

## Generator loss and Discriminator loss ##
lambda_reg = 10  #0.5,  Gradient penalty coefficient (λ)
def discriminator_loss(D_real, D_fake, penalty):
    D_loss = tf.reduce_mean(D_fake - D_real + lambda_reg * penalty)
    return D_loss

def generator_loss(D_fake):
    G_loss = -tf.reduce_mean(D_fake)
    return G_loss

## Optimizers ##
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9) #1e-3, beta_1=0.5, beta_2=0.9
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

## Gradient penalty to the Discriminator loss ##
def penalty_calculation(X_real, G_fake, condition):
    # Create the gradient penalty operations.
    epsilon = tf.random.uniform(shape=tf.shape(X_real), minval=0., maxval=1.)  #minval=0., maxval=1.
    interpolation = epsilon * X_real + (1 - epsilon) * G_fake
    with tf.GradientTape() as pena_tape:
        pena_tape.watch(interpolation)
        penalty = (tf.norm(
            pena_tape.gradient(discriminator([condition, interpolation]), interpolation),
            axis=1) - 1) ** 2.0
    return penalty

## Train Generator and Discriminator independently  ##
@tf.function
def train_G(data_batch):
    noise = tf.random.normal([data_batch.shape[0], noise_dim], mean=0.0, stddev=0.1, #stddev=1.0
                             dtype=tf.dtypes.float32)
    condition = data_batch[:, :condition_dim]                                   
    with tf.GradientTape() as gen_tape:
        G_fake = generator([condition, noise], training=True)
        D_fake = discriminator([condition, G_fake], training=True)
        G_loss = generator_loss(D_fake)
    gradients_of_generator = gen_tape.gradient(G_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return G_loss

@tf.function
def train_D(data_batch):
    noise = tf.random.normal([data_batch.shape[0], noise_dim], mean=0.0, stddev=0.1, #stddev=1.0
                             dtype=tf.dtypes.float32)
    condition = data_batch[:, :condition_dim]                                   
    target = data_batch[:, condition_dim :condition_dim + gen_dim]                                       
    with tf.GradientTape() as disc_tape:
        G_fake = generator([condition, noise], training=True)
        D_real = discriminator([condition, target], training=True)
        D_fake = discriminator([condition, G_fake], training=True)
        penalty = penalty_calculation(target, G_fake, condition)
        D_loss = discriminator_loss(D_real, D_fake, penalty)
    gradients_of_discriminator = disc_tape.gradient(D_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return D_loss


def train(dataset, epochs, D_cycle=D_cycle, steps_show=steps_show):          #D_cycle=1
    start = time.time()
    
    loss_G_train = []
    loss_D_train = []
    for epoch in range(epochs):
        for data_batch in dataset:
            G_loss = train_G(data_batch)
            for _ in range(D_cycle):
                D_loss = train_D(data_batch)

        loss_G_train.append(G_loss.numpy())
        loss_D_train.append(D_loss.numpy())

        num_test = num_first
        condition = data[:num_test, :condition_dim]                         
        noise = tf.random.normal([num_test, noise_dim], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32)  #stddev=1.0
        generated_out = generator([condition, noise], training=True)

        generated_out_final = scaler.inverse_transform(generated_out)


        tiempo = time.time() - start

        minutos = int(tiempo // 60)
        segundos = int(tiempo % 60)

        #print(f"Minutos: {minutos}")
        #print(f"Segundos: {segundos}")
        print('Time for epoch {}/{} is {} sec, {} minutos y {} segundos'.format(epoch, epochs, time.time() - start, minutos, segundos ))

    return loss_G_train, loss_D_train, generated_out_final

loss_G_train, loss_D_train, generated_out_final = train(train_dataset, epochs=epoch_num, D_cycle=D_cycle, steps_show=steps_show)
generator.save('Models/CWGAN_'+str(j))
print('Model saved in Models/CWGAN_')

condition = data[:, :condition_dim]
data_generated = np.concatenate([condition.numpy(), generated_out_final], axis=1)
dataframe_generated=pd.DataFrame(data_generated)

#Training visualization
#Visualization
condition = data[:, :condition_dim]
fontsize = 8
list_limx = [[0, .055], [0, .055], [0, .055], [0, .055], [0, .055], [0, .055], [0, .055]]
#list_limy = [[0, 100], [0, 450], [-0.1, 1.1], [-0.1, 1.1]]
figure, ax = plt.subplots(1, 7, figsize=(15, 3))
figure.suptitle("Conditional GAN (CWGAN-GP)")
sns.set(color_codes=True, style='white', palette='colorblind')
          
### d1  -  label1  ###
i = 0
ax[i].clear()
ax[i].set_xlim(list_limx[i])
ax[i].set_xlabel('d1 (mm)')
ax[i].set_ylabel('label1')
plot_data_real= data_real.numpy()[:num_test, :]                
ax[i].plot(plot_data_real[:num_test, 7], plot_data_real[:num_test, 0], '.b', alpha=.5, label="Real")
ax[i].yaxis.set_major_locator(plt.MultipleLocator(base=1)) 
ax[i].yaxis.set_major_formatter('{:.0f}'.format)  
plot_data_generated = dataframe_generated.values
ax[i].plot(plot_data_generated[:num_test, 7], plot_data_generated[:num_test, 0]+0.05, '.r', alpha=.5, label="Generated")
ax[i].legend(loc='center right', fontsize=fontsize)
plt.subplots_adjust(wspace=0.5)  
                                
### d2  -  label2  ###
i = 1
ax[i].clear()
ax[i].set_xlim(list_limx[i])
ax[i].set_xlabel('d2 (mm)')
ax[i].set_ylabel('label2')
plot_data_real= data_real.numpy()[:num_test, :]                
ax[i].plot(plot_data_real[:num_test, 8], plot_data_real[:num_test, 1], '.b', alpha=.5, label="Real")
ax[i].yaxis.set_major_locator(plt.MultipleLocator(base=1)) 
ax[i].yaxis.set_major_formatter('{:.0f}'.format)  
plot_data_generated = dataframe_generated.values
ax[i].plot(plot_data_generated[:num_test, 8], plot_data_generated[:num_test, 1]+0.05, '.r', alpha=.5, label="Generated")
ax[i].legend(loc='center right', fontsize=fontsize)
plt.subplots_adjust(wspace=0.5)     

### d3  -  label3  ###
i = 2
ax[i].clear()
ax[i].set_xlim(list_limx[i])
ax[i].set_xlabel('d3 (mm)')
ax[i].set_ylabel('label3')
plot_data_real= data_real.numpy()[:num_test, :]                
ax[i].plot(plot_data_real[:num_test, 9], plot_data_real[:num_test, 2], '.b', alpha=.5, label="Real")
ax[i].yaxis.set_major_locator(plt.MultipleLocator(base=1)) 
ax[i].yaxis.set_major_formatter('{:.0f}'.format)  
plot_data_generated = dataframe_generated.values
ax[i].plot(plot_data_generated[:num_test, 9], plot_data_generated[:num_test, 2]+0.05, '.r', alpha=.5, label="Generated")
ax[i].legend(loc='center right', fontsize=fontsize)
plt.subplots_adjust(wspace=0.5)     

### d4  -  label4  ###
i = 3
ax[i].clear()
ax[i].set_xlim(list_limx[i])
ax[i].set_xlabel('d4 (mm)')
ax[i].set_ylabel('label4')
plot_data_real= data_real.numpy()[:num_test, :]                
ax[i].plot(plot_data_real[:num_test, 10], plot_data_real[:num_test, 3], '.b', alpha=.5, label="Real")
ax[i].yaxis.set_major_locator(plt.MultipleLocator(base=1)) 
ax[i].yaxis.set_major_formatter('{:.0f}'.format)  
plot_data_generated = dataframe_generated.values
ax[i].plot(plot_data_generated[:num_test, 10], plot_data_generated[:num_test, 3]+0.05, '.r', alpha=.5, label="Generated")
ax[i].legend(loc='center right', fontsize=fontsize)
plt.subplots_adjust(wspace=0.5)  

### d5  -  label5  ###
i = 4
ax[i].clear()
ax[i].set_xlim(list_limx[i])
ax[i].set_xlabel('d5 (mm)')
ax[i].set_ylabel('label5')
plot_data_real= data_real.numpy()[:num_test, :]                
ax[i].plot(plot_data_real[:num_test, 11], plot_data_real[:num_test, 4], '.b', alpha=.5, label="Real")
ax[i].yaxis.set_major_locator(plt.MultipleLocator(base=1)) 
ax[i].yaxis.set_major_formatter('{:.0f}'.format)  
plot_data_generated = dataframe_generated.values
ax[i].plot(plot_data_generated[:num_test, 11], plot_data_generated[:num_test, 4]+0.05, '.r', alpha=.5, label="Generated")
ax[i].legend(loc='center right', fontsize=fontsize)
plt.subplots_adjust(wspace=0.5)  

### d6  -  label6  ###
i = 5
ax[i].clear()
ax[i].set_xlim(list_limx[i])
ax[i].set_xlabel('d6 (mm)')
ax[i].set_ylabel('label6')
plot_data_real= data_real.numpy()[:num_test, :]                 
ax[i].plot(plot_data_real[:num_test, 12], plot_data_real[:num_test, 5], '.b', alpha=.5, label="Real")
ax[i].yaxis.set_major_locator(plt.MultipleLocator(base=1)) 
ax[i].yaxis.set_major_formatter('{:.0f}'.format)  
plot_data_generated = dataframe_generated.values
ax[i].plot(plot_data_generated[:num_test, 12], plot_data_generated[:num_test, 5]+0.05, '.r', alpha=.5, label="Generated")
ax[i].legend(loc='center right', fontsize=fontsize)
plt.subplots_adjust(wspace=0.5)    
            
### d6 - label ###
i = 6
ax[i].clear()
ax[i].set_xlim(list_limx[i])
#ax[i].set_ylim(list_limy[i])
ax[i].set_xlabel('d6 (mm)')
ax[i].set_ylabel('label')                                    
plot_data_real  = data_real.numpy()[:num_test, :]
ax[i].plot(plot_data_real [:num_test, 12], plot_data_real [:num_test, 6], '.b', alpha=.5, label="Real")
ax[i].yaxis.set_major_locator(plt.MultipleLocator(base=1)) 
ax[i].yaxis.set_major_formatter('{:.0f}'.format)  
plot_data_generated = dataframe_generated.values
ax[i].plot(plot_data_generated[:num_test, 12], plot_data_generated[:num_test, 6]+0.05, '.r', alpha=.5, label="Generated")
ax[i].legend(loc='center right', fontsize=fontsize)
figure.show()

names = ['label1', 'label2', 'label3', 'label4', 'label5', 'label6','d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'F','t']
dataframe_generated.columns = names
columns= ['label1', 'label2', 'label3', 'label4', 'label5', 'label6']
dataframe_generated[columns] = dataframe_generated[columns].astype(int)
dataframe_generated
np.savetxt("X_generated_training " + str(j) + ".csv",dataframe_generated, delimiter=",")

# Training Metrics
#Metrics: SSIM, (see reference Guan S. et al, Evaluation of GAN performance ...)
# Calculate the SSIM between the two distributions, real and generated
ssim_value = ssim(out, generated_out_final, multichannel=True)
print('SSIM:', ssim_value)

#Metrics: Wasserstein distance (see reference Gulrajani I. et al, Improved training of Wasserstein gans ...)
wasserstein_dist = wasserstein_distance(out.ravel(), generated_out_final.ravel())
print("Waserstein distance of the generated dataset:", wasserstein_dist)

#Metrics: FID: Frechet Incepcion Distance (see reference Heusel M. et al, Gans trained by a two time-scale update rule converge ...)
real_mean = np.mean(out, axis=0)
real_covariance = np.cov(out, rowvar=False)
fake_mean = np.mean(generated_out_final, axis=0)
fake_covariance = np.cov(generated_out_final, rowvar=False)
mean_difference = real_mean - fake_mean
mean_difference_squared = np.dot(mean_difference, mean_difference)
prod_covariance = real_covariance * fake_covariance
covariance_sqrt, _ = scipy.linalg.sqrtm(prod_covariance, disp=False)
if not np.isfinite(covariance_sqrt).all():
    offset = np.eye(sum_covariance.shape[0]) * 1e-6
    covariance_sqrt = scipy.linalg.sqrtm((prod_covariance + offset), disp=False)  
fid1 = mean_difference_squared + np.trace(real_covariance + fake_covariance - 2 * covariance_sqrt)
print("Frechet Inception Distance (FID):", fid1)

# Visualization of the comparison between real data and training generated data
#Visualization
F_train = dataframe['F']
d6_train = dataframe['d6']
t_train = dataframe['t']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title("Projection over the 3 axes od the TRAINING DATASET")
#ax.scatter(x_train,y_train,z_train, 'o')
ax.set_xlabel('X axis = F (N)', fontsize='small')
ax.set_ylabel('Y axis = d6 (m)', fontsize='small')
ax.set_zlabel('Z axis = t (years)', fontsize='small')

cx = np.ones_like(F_train) * ax.get_xlim3d()[0]
cy = np.ones_like(d6_train) * ax.get_ylim3d()[0]
cz = np.ones_like(t_train) * ax.get_zlim3d()[0]

ax.scatter(F_train,  d6_train,  cz,               marker='.', lw=0,  color = "blue" , label= "F (N)")
ax.scatter(F_train,  cy,       t_train,        marker='.', lw=0, alpha=0.05 , color ="orange", label = "d6 (m)")
ax.scatter(cx,       d6_train,  t_train,         marker='.', lw=0,  color ="green", label ="t (years)")

ax.set_xlim3d(ax.get_xlim3d())
ax.set_ylim3d(ax.get_ylim3d())
ax.set_zlim3d(ax.get_zlim3d())
ax.set_xlabel('X axis = F (N)')
ax.set_ylabel('Y axis = d6 (m)')
ax.set_zlabel('Z axis = t (years)')
ax.legend(loc='upper left')
plt.show()

F_predicted = dataframe_generated['F']
d6_predicted = dataframe_generated['d6']
t_predicted = dataframe_generated['t']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title("Projection over the 3 axes of the PREDICTED DATASET")
#ax.scatter(x_train,y_train,z_train, 'o')
ax.set_xlabel('X axis = F (N)', fontsize='small')
ax.set_ylabel('Y axis = d6 (m)', fontsize='small')
ax.set_zlabel('Z axis = t (years)', fontsize='small')

cx = np.ones_like(F_predicted) * ax.get_xlim3d()[0]
cy = np.ones_like(d6_predicted) * ax.get_ylim3d()[0]
cz = np.ones_like(t_predicted) * ax.get_zlim3d()[0]

ax.scatter(F_predicted,  d6_predicted,  cz,               marker='.', lw=0,  color = "blue" , label= "F (N)")
ax.scatter(F_predicted,  cy,       t_predicted,        marker='.', lw=0, alpha=0.05 , color ="orange", label = "d6 (m)")
ax.scatter(cx,       d6_predicted,  t_predicted,         marker='.', lw=0,  color ="green", label ="t (years)")

ax.set_xlim3d(ax.get_xlim3d())
ax.set_ylim3d(ax.get_ylim3d())
ax.set_zlim3d(ax.get_zlim3d())
ax.set_xlabel('X axis = F (N)')
ax.set_ylabel('Y axis = d6 (m)')
ax.set_zlabel('Z axis = t (years)')
ax.legend(loc='upper left')
plt.show()

# Testing
dataTest= pd.read_excel('dataTest.xlsx', header=None)
out_test = dataTest.drop(dataTest.columns[:7], axis=1)
condition_test = dataTest.drop(dataTest.columns[6:], axis=1)
condition_test = condition_test.to_numpy()
noise = tf.random.normal([num_test, noise_dim], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32)
test_generated_out = generator([condition_test, noise], training=False)
test_generated_out_final = scaler.inverse_transform(test_generated_out)
plot_data_generated = np.concatenate([condition_test, test_generated_out_final], axis=1)
df_data_generated=pd.DataFrame(plot_data_generated)

# Testing Metrics
#Metrics: Wasserstein distance
wasserstein_dist = wasserstein_distance(out_test.ravel(), test_generated_out_final.ravel())
print("Waserstein distance of the generated dataset:", wasserstein_dist)

#Metrics: FID
real_mean = np.mean(out_test, axis=0)
real_covariance = np.cov(out_test, rowvar=False)
fake_mean = np.mean(test_generated_out_final, axis=0)
fake_covariance = np.cov(test_generated_out_final, rowvar=False)
mean_difference = real_mean - fake_mean
mean_difference_squared = np.dot(mean_difference, mean_difference)
prod_covariance = real_covariance * fake_covariance
covariance_sqrt, _ = scipy.linalg.sqrtm(prod_covariance, disp=False)
if not np.isfinite(covariance_sqrt).all():
    offset = np.eye(sum_covariance.shape[0]) * 1e-6
    covariance_sqrt = scipy.linalg.sqrtm((prod_covariance + offset), disp=False)
fid1 = mean_difference_squared + np.trace(real_covariance + fake_covariance - 2 * covariance_sqrt)
print("Frechet Inception Distance (FID):", fid1)

#Metrics: SSIM
ssim_value = ssim(out_test, test_generated_out_final, multichannel=True)
print('SSIM:', ssim_value)

#Visualization of the comparison between real data and testing generated data
#Visualization
figure, ax = plt.subplots(1, 4, figsize=(15, 3))
figure.suptitle("Conditional GAN (WCGAN-GP)")
sns.set(color_codes=True, style='white', palette='colorblind')
          
### d6 - t  ###
i = 0
ax[i].clear()
ax[i].set_xlabel('d6 (mm)')
ax[i].set_ylabel('t (years)')
plot_data_real = dataTest               
ax[i].plot(plot_data_real[ 12], plot_data_real[ 14], '.b', alpha=1, label="Test")
ax[i].plot(plot_data_generated[:num_test, 12], plot_data_generated[:num_test, 14], '.r', alpha=1, label="Generated")
ax[i].legend(loc='lower right')  
            
###  d6 - F  ###
i = 1
ax[i].clear()
ax[i].set_xlabel('d6 (mm)')
ax[i].set_ylabel('F (N)')
ax[i].plot(plot_data_real [ 12], plot_data_real [ 13], '.b', alpha=1, label="Test")
ax[i].plot(plot_data_generated[:num_test, 12], plot_data_generated[:num_test, 13], '.r', alpha=1, label="Generated")
ax[i].legend(loc='lower right')
            
### d6 - label ###
i = 2
ax[i].clear()
ax[i].set_xlabel('d6 (mm)')
ax[i].set_ylabel('label')
ax[i].plot(plot_data_real [ 12], plot_data_real [ 6], '.b', alpha=1, label="Test")
ax[i].yaxis.set_major_locator(plt.MultipleLocator(base=1)) 
ax[i].yaxis.set_major_formatter('{:.0f}'.format)  
ax[i].plot(plot_data_generated[:num_test, 12], plot_data_generated[:num_test, 6]+0.05, '.r', alpha=1, label="Generated")
ax[i].legend(loc='center right')

### d6  -  label6  ###
i = 3
ax[i].clear()
ax[i].set_xlabel('d6 (mm)')
ax[i].set_ylabel('label6')
ax[i].plot(plot_data_real[ 12], plot_data_real[ 5], '.b', alpha=1, label="Real")
ax[i].yaxis.set_major_locator(plt.MultipleLocator(base=1)) 
ax[i].yaxis.set_major_formatter('{:.0f}'.format)  
ax[i].plot(plot_data_generated[:num_test, 12], plot_data_generated[:num_test, 5]+0.05, '.r', alpha=1, label="Generated")
ax[i].legend(loc='center right')
plt.subplots_adjust(wspace=0.5)    
figure.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title("Projection over the 3 axes")
#ax.scatter(x_train,y_train,z_train, 'o')
ax.set_xlabel('X axis = F (N)', fontsize='small')
ax.set_ylabel('Y axis = d6 (m)', fontsize='small')
ax.set_zlabel('Z axis = t (years)', fontsize='small')

cx = np.ones_like(F_predicted) * ax.get_xlim3d()[0]
cy = np.ones_like(d6_predicted) * ax.get_ylim3d()[0]
cz = np.ones_like(t_predicted) * ax.get_zlim3d()[0]

ax.scatter(F_predicted,  d6_predicted,  cz,               marker='.', lw=0,  color = "blue" , label= "F (N)")
ax.scatter(F_predicted,  cy,       t_predicted,        marker='.', lw=0, alpha=0.5 , color ="orange", label = "d6 (m)")
ax.scatter(cx,       d6_predicted,  t_predicted,         marker='.', lw=0,  color ="green", label ="t (years)")

ax.set_xlim3d(ax.get_xlim3d())
ax.set_ylim3d(ax.get_ylim3d())
ax.set_zlim3d(ax.get_zlim3d())
ax.set_xlabel('X axis = F (N)')
ax.set_ylabel('Y axis = d6 (m)')
ax.set_zlabel('Z axis = t (years)')
ax.legend(loc='upper left')
plt.show()