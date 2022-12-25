# -*- coding: utf-8 -*-
"""
@author: Ibrahim Kovan
https://ibrahimkovan.medium.com/

dataset: http://www.dronedataset.icg.tugraz.at/
dataset link: https://www.kaggle.com/awsaf49/semantic-drone-dataset
License: CC0: Public Domain

"""
#%% Libraries
"""1"""
from architecture import multiclass_unet_architecture, jacard, jacard_loss
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random 
from skimage.io import imshow
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

#%% Import train and mask dataset
"""2"""
train_path = r"PycharmProjects/DataUnet/semantic_drone_dataset/training_set/images/*.jpg"
def importing_data(path):
    sample = []
    for filename in glob.glob(path):
        img = Image.open(filename,'r')
        img = img.resize((256,256))
        img = np.array(img)
        sample.append(img)  
    return sample

data_train   = importing_data(train_path)
data_train = np.asarray(data_train)


mask_path = r"PycharmProjects/DataUnet/semantic_drone_dataset/training_set/gt/semantic/label_images/*.png"
def importing_data(path):
    sample = []
    for filename in glob.glob(path):
        img = Image.open(filename,'r')
        img = img.resize((256,256))
        img = np.array(img)
        sample.append(img)  
    return sample

data_mask   = importing_data(mask_path)
data_mask  = np.asarray(data_mask)

#%% Random visualization
x = random.randint(0, len(data_train))
plt.figure(figsize=(24,18))
plt.subplot(1,2,1)
imshow(data_train[x])
plt.subplot(1,2,2)
imshow(data_mask[x])
plt.show()

#%% Normalization
"""3"""
scaler = MinMaxScaler()
nsamples, nx, ny, nz = data_train.shape
d2_data_train = data_train.reshape((nsamples,nx*ny*nz))
train_images = scaler.fit_transform(d2_data_train)
train_images = train_images.reshape(400,256,256,3)

#%% Labels of the masks
"""4"""
labels = pd.read_csv(r"PycharmProjects/DataUnet/semantic_drone_dataset/training_set/gt/semantic/class_dict.csv")
labels = labels.drop(['name'],axis = 1)
labels = np.array(labels)

def image_labels(label):
    image_labels = np.zeros(label.shape, dtype=np.uint8)
    for i in range(24):
        image_labels [np.all(label == labels[i,:],axis=-1)] = i
    image_labels = image_labels[:,:,0]
    return image_labels


label_final = []
for i in range(data_mask.shape[0]):
    label = image_labels(data_mask[i])
    label_final.append(label)    

label_final = np.array(label_final)   
#%% train_test
"""5"""
n_classes = len(np.unique(label_final))
labels_cat = to_categorical(label_final, num_classes=n_classes)
x_train, x_test, y_train, y_test = train_test_split(train_images, labels_cat, test_size = 0.20, random_state = 42)

#%% U-Net
"""6"""
img_height = x_train.shape[1]
img_width  = x_train.shape[2]
img_channels = x_train.shape[3]


metrics=['accuracy', jacard]

def get_model():
    return multiclass_unet_architecture(n_classes=n_classes, height=img_height, 
                           width=img_width, channels=img_channels)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()

history = model.fit(x_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=100, 
                    validation_data=(x_test, y_test), 
                    shuffle=False)
#%%
"""7"""
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['jacard']
val_acc = history.history['val_jacard']

plt.plot(epochs, acc, 'y', label='Training Jaccard')
plt.plot(epochs, val_acc, 'r', label='Validation Jaccard')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Jaccard')
plt.legend()
plt.show()
#%%
"""8"""
y_pred=model.predict(x_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(y_test, axis=3)

test_jacard = jacard(y_test,y_pred)
print(test_jacard)
#%%
"""9"""
fig, ax = plt.subplots(5, 3, figsize = (12,18)) 
for i in range(0,5):
    test_img_number = random.randint(0, len(x_test))
    test_img = x_test[test_img_number]
    ground_truth=y_test_argmax[test_img_number] 
    test_img_input=np.expand_dims(test_img, 0) 
    prediction = (model.predict(test_img_input)) 
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]    
    
    ax[i,0].imshow(test_img)
    ax[i,0].set_title("RGB Image",fontsize=16)
    ax[i,1].imshow(ground_truth)
    ax[i,1].set_title("Ground Truth",fontsize=16)
    ax[i,2].imshow(predicted_img)
    ax[i,2].set_title("Prediction",fontsize=16)
    i+=i
    
plt.show()
#%% pre-trained model
"""10"""
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
x_train_new = preprocess_input(x_train)
x_test_new = preprocess_input(x_test)

# define model
model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')

metrics=['accuracy', jacard]
# compile keras model with defined optimozer, loss and metrics
#model_resnet_backbone.compile(optimizer='adam', loss=focal_loss, metrics=metrics)
model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model_resnet_backbone.summary())

history_tf=model_resnet_backbone.fit(x_train_new, 
          y_train,
          batch_size=16, 
          epochs=100,
          verbose=1,
          validation_data=(x_test_new, y_test))

#%%
"""11"""
history = history_tf
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['jacard']
val_acc = history.history['val_jacard']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation Jaccard')
plt.xlabel('Epochs')
plt.ylabel('Jaccard')
plt.legend()
plt.show()

#%%
"""12"""
y_pred_tf=model_resnet_backbone.predict(x_test)
y_pred_argmax_tf=np.argmax(y_pred_tf, axis=3)
y_test_argmax_tf=np.argmax(y_test, axis=3)

test_jacard = jacard(y_test,y_pred_tf)
print(test_jacard)

#%%
"""13"""
fig, ax = plt.subplots(5, 3, figsize = (12,18)) 
for i in range(0,5):

    test_img_number = random.randint(0, len(x_test))
    test_img_tf = x_test_new[test_img_number]
    ground_truth_tf=y_test_argmax_tf[test_img_number]
    test_img_input_tf=np.expand_dims(test_img_tf, 0)
    prediction_tf = (model_resnet_backbone.predict(test_img_input_tf))
    predicted_img_transfer_learning=np.argmax(prediction_tf, axis=3)[0,:,:]   
    
    ax[i,0].imshow(test_img_tf)
    ax[i,0].set_title("RGB Image",fontsize=16)
    ax[i,1].imshow(ground_truth_tf)
    ax[i,1].set_title("Ground Truth",fontsize=16)
    ax[i,2].imshow(predicted_img_transfer_learning)
    ax[i,2].set_title("Prediction(Transfer Learning)",fontsize=16)
    i+=i
    
plt.show()

"""14"""
fig, ax = plt.subplots(5, 4, figsize = (16,20)) 
for i in range(0,5):
    test_img_number = random.randint(0, len(x_test))
    
    test_img = x_test[test_img_number]
    ground_truth=y_test_argmax[test_img_number] 
    test_img_input=np.expand_dims(test_img, 0) 
    prediction = (model.predict(test_img_input)) 
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]    
    
    test_img_tf = x_test_new[test_img_number]
    ground_truth_tf=y_test_argmax_tf[test_img_number]
    test_img_input_tf=np.expand_dims(test_img_tf, 0)
    prediction_tf = (model_resnet_backbone.predict(test_img_input_tf))
    predicted_img_transfer_learning=np.argmax(prediction_tf, axis=3)[0,:,:]  
    
    ax[i,0].imshow(test_img_tf)
    ax[i,0].set_title("RGB Image",fontsize=16)
    ax[i,1].imshow(ground_truth_tf)
    ax[i,1].set_title("Ground Truth",fontsize=16)
    ax[i,2].imshow(predicted_img)
    ax[i,2].set_title("Prediction",fontsize=16)
    ax[i,3].imshow(predicted_img_transfer_learning)
    ax[i,3].set_title("Prediction Transfer Learning",fontsize=16)
    
    i+=i   
plt.show()