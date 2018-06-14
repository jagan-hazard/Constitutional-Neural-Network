#import necessary package
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
import tensorflow as tf


img_width, img_height = 150, 150 # dimensions of our images.
input_shape = (img_width, img_height,3)  #for tensorflow backend in whih we trainned our model

#defining theexact same model which is used for training
test_model = Sequential()

test_model.add(Conv2D(32, (3, 3), input_shape=input_shape))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))

test_model.add(Conv2D(64, (3, 3)))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))

test_model.add(Conv2D(128, (3, 3)))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))

test_model.add(Conv2D(256, (3, 3)))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))

test_model.add(Flatten())
test_model.add(Dense(256))
test_model.add(Activation('relu'))
test_model.add(Dropout(0.5))
test_model.add(Dense(1))
test_model.add(Activation('sigmoid'))

# load json and create model
json_file = open('model.json', 'r')
load_model_json = json_file.read()
json_file.close()
load_model = model_from_json(load_model_json)

# load weights into new model
test_model =load_model.load_weights("/home/jagan/Desktop/RESULTS_FOR_PAPER/LINEAR/iter_20_16_cnn32_64_128_fc256.h5")
print("Loaded model from disk")

#test_model = load_model('/home/student/Desktop/II_review_with_aspect_ratio/first_try.h5')
basedir = "/home/jagan/Desktop/RESULTS_FOR_PAPER/LINEAR/data/test/" #test image path
basedir1="/home/jagan/Desktop/RESULTS_FOR_PAPER/LINEAR/data/test/"  #test image path
def predict(basedir, model):
	for i in range(1,71):                             #Here i used 70 images for training hence used for loop.
													  #if you want to use single image for testing delete the for loop syntax along retain rest of the code
		path = basedir + str(i) + '.jpg'
		img = load_img(path,False,target_size=(img_width,img_height))
		x = img_to_array(img)
		x = np.expand_dims(x, axis=0)
		preds = load_model.predict_classes(x)
		probs = load_model.predict_proba(x)
		print(probs)
		print("Processed Image:",str(i))
		path1= basedir1 + str(i) + '.jpg'		
		image=cv2.imread(path1)
		if probs<1:		
			cv2.putText(image,"Its an elephant!!!", (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
		elif probs>=1:
			cv2.putText(image,"Its not an elephant!!!", (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
		else: 
			cv2.putText(image,"error on prediction!!!", (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)		
		cv2.imshow('image',image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

predict(basedir, test_model)

print('done')
