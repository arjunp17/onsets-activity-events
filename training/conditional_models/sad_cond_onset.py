from keras.models import Model
from keras.layers import Input, merge, Multiply, Concatenate, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation, Permute, Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling1D, Conv1D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import keras
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from itertools import repeat


# input dimensions  
rows, cols , channels = 40,500,1  

#model params
epochs = 200
batch_size =  32
nb_classes = 10
lr = 0.001

def sad_cond_onset(rows,cols,channels,nb_classes,lr):

######################### SHARED ARCHITECTURE ##############################################	
	#conv1
	conv2d_1 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same')(input = Input(shape=(rows,cols,channels)))
	conv2d_1 = BatchNormalization(axis=-1)(conv2d_1)
	conv2d_1 = Dropout(0.30)(conv2d_1)

	#maxpool1
	MP_1 = MaxPooling2D(pool_size=(5, 1), strides=(5, 1))(conv2d_1)

	#conv2
	conv2d_2 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same')(MP_1)
	conv2d_2 = BatchNormalization(axis=-1)(conv2d_2)
	conv2d_2 = Dropout(0.30)(conv2d_2)

	#maxpool2
	MP_2 = MaxPooling2D(pool_size=(4, 1), strides=(4, 1))(conv2d_2)

##################### SAD BRANCH #############################################################
	#conv3
	conv2d_3_pa = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same')(MP_2)
	conv2d_3_pa = BatchNormalization(axis=-1)(conv2d_3_pa)
	conv2d_3_pa = Dropout(0.30)(conv2d_3_pa)

	#maxpool3
	MP_3_pa = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(conv2d_3_pa)

	#reshape
	reshape1_pa = Reshape((64,-1))(MP_3_pa)
	reshape1_pa = Permute((2,1))(reshape1_pa)

	#GRU
	gru_pa = GRU(64, activation='tanh', return_sequences=True)(reshape1_pa)
	
	#intermediate sad prediction
	pa_out_int = TimeDistributed(Dense(nb_classes, activation='sigmoid'))(gru_pa)
	pa_out_int = TimeDistributed(Dense(1, activation='sigmoid'), name = 'pa_out_int')(pa_out_int)

##################### ONSET BRANCH #############################################################
	#conv3
	conv2d_3_on = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same')(MP_2)
	conv2d_3_on = BatchNormalization(axis=-1)(conv2d_3_on)
	conv2d_3_on = Dropout(0.30)(conv2d_3_on)

	#maxpool3
	MP_3_on = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(conv2d_3_on)

	#reshape
	reshape1_on = Permute((3,2,1))(MP_3_on)
	MP_4_on = AveragePooling2D(pool_size=(64, 1), strides=(1, 1))(reshape1_on)
	reshape2_on = Reshape((500,-1))(MP_4_on)

	#GRU
	gru_on = Bidirectional(GRU(64, activation='tanh', return_sequences=True))(reshape2_on)
	gru_on = Bidirectional(GRU(64, activation='tanh', return_sequences=True))(gru_on)

	#onset prediction
	dense_on = TimeDistributed(Dense(nb_classes, activation='sigmoid'))(gru_on)
	on_out = TimeDistributed(Dense(1, activation='sigmoid'), name = 'on_out')(dense_on)
################################################################################################

	#aggregation
	concat = concatenate([pa_out_int, on_out], axis=2)
	gru_agg = Bidirectional(GRU(64, activation='tanh', return_sequences=True))(concat)
	pa_out = TimeDistributed(Dense(nb_classes, activation='sigmoid'))(gru_agg)
	# final sad prediction
	pa_out = TimeDistributed(Dense(1, activation='sigmoid'), name = 'pa_out')(pa_out)
	model = Model(input, outputs = [pa_out,on_out], name='sad_onset')
	# compile model
	opt = Adam(lr = lr)
	losses = {'pa_out': 'binary_crossentropy','on_out': 'binary_crossentropy'}
	lossWeights = {'pa_out': 0.5, 'on_out': 0.5}
	model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model


sad_cond_onset = sad_cond_onset(rows,cols,channels,nb_classes,lr)

# feature and label
X_train = np.load('.../train_feature.npy')
X_val = np.load('../val_feature.npy')

Y_train_SAD = np.load('../train_label_SAD.npy')
Y_val_SAD = np.load('../val_label_SAD.npy')

Y_train_onset = np.load('../train_label_ONSET.npy')
Y_val_onset = np.load('../val_label_ONSET.npy')


## training
def model_train(model,X_train,Y_train_SAD,Y_train_onset,X_val,Y_val_SAD,Y_val_onset,epochs,batch_size,model_name,output_folder):
	filepath=os.path.join(output_folder,model_name + '-{epoch:02d}-{val_pa_out_loss:.2f}.hdf5')
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	hist = model.fit(X_train, {'pa_out': Y_train_SAD, 'on_out': Y_train_onset}, validation_data=(X_val, {'pa_out': Y_val_SAD, 'on_out': Y_val_onset}), callbacks=callbacks_list, epochs=epochs, shuffle=False,batch_size=batch_size,verbose=2)
	return hist
	
	
train_history = model_train(sad_cond_onset,X_train,Y_train_SAD,Y_train_onset,X_val,Y_val_SAD,Y_val_onset,epochs,batch_size,sad_onset,output_folder)


