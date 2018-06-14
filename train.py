'''
data/
    train/
        elephant/
            ele001.jpg
            ele002.jpg
            ...
        others/
            other001.jpg
            other002.jpg
            ...
    validation/
        elephant/
            ele801.jpg
            ele802.jpg
            ...
        others/
            other801.jpg
            other802.jpg
            ...
'''
#from numpy.random import seed
#seed(42)
#Import the necessary package
from tensorflow import set_random_seed
seed_val=set_random_seed(42)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import model_from_json
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
import matplotlib.pyplot as plt
import math
import cv2

#seed=42
img_width, img_height = 150, 150  # dimensions of our images.

train_data_dir = 'data/train'  #path to the training images
validation_data_dir = 'data/validation'  #path to the testing images
nb_train_samples = 8470         #no of training samples
nb_validation_samples = 2180    #no of validating samples
epochs = 20                     #no of epoch
batch_size = 16                 #batch size

#backend selection whether it is tensorflow or Theano

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#Layer creation using Keras framework    
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2 )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

#Defining training parameters
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    seed=seed_val,	
    class_mode='binary')
#Defining Validation Parameters
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    seed=seed_val,
    class_mode='binary')
#using model.fit_generator to run the training
history=model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
#history = model.fit(epochs, batch_size)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('none_20_16_cnn_32_fc_16.h5')
print("Saved model to disk")

plot_model(model, to_file='model.png') #create the image for the model which is defined in code.
#Line Plot
#using matplotlib for plotiing the training and validation accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
#Line Plot
#using matplotlib for plotiing the training and validation  error
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Dotted Plot
#using matplotlib for plotiing the training and validation accuracy 
x=history.history['acc']
y=history.history['val_acc']
plt.plot(x,'bs',y,'g^')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
#Dotted Plot
#using matplotlib for plotiing the training and validation accuracy and error
a=history.history['loss']
b=history.history['val_loss']
plt.plot(a,'bs',b,'g^')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

model.summary()  #to get the summary of the training
model.get_weights() # to get the weight details of each layer
