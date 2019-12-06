from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn import metrics
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def model_predict(model,layer):
	
	# load model
	base_model = load_model(model)
	#opt = Adam(lr = 0.001)
	#model.compile(loss='binary_crossentropy', optimizer=adada, metrics=['accuracy'])
	base_model.summary()
	#choose the layer 
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output) 
	# load test feature
	test_feature = np.load('../test_feature.npy')
	# predict output
	output = []
	for i in range(len(test_feature)):
   		output.append(model.predict(np.reshape(test_feature[i], (1,40,500,1))))

	output = np.array(output)
	np.save('model_ouput', output)
	
	
## sed prediction from sed_sad_onset model

model_predict('sed_sad_onset_model.hdf5','sl_out')

