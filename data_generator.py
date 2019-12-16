import numpy as np
import datetime
import os
import cv2
import random as rn
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization
from keras.layers import Activation, Dropout, ZeroPadding3D
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D
from keras.layers.recurrent import LSTM

class DataGenerator:
    def __init__(self, width=120, height=120, frames=30, channel=3, crop = True, normalize = False, affine = False, flip = False, edge = False  ):
        self.width = width   # X dimension of the image
        self.height = height # Y dimesnion of the image
        self.frames = frames # length/depth of the video frames
        self.channel = channel # number of channels in images 3 for color(RGB) and 1 for Gray  
        self.affine = affine # augment data with affine transform of the image
        self.flip = flip
        self.normalize =  normalize
        self.edge = edge # edge detection
        self.crop = crop

    # Helper function to generate a random affine transform on the image
    def __get_random_affine(self): # private method
        dx, dy = np.random.randint(-1.7, 1.8, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return M

    # Helper function to initialize all the batch image data and labels
    def __init_batch_data(self, batch_size): # private method
        batch_data = np.zeros((batch_size, self.frames, self.width, self.height, self.channel)) 
        batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
        return batch_data, batch_labels

    def __load_batch_images(self, source_path, folder_list, batch_num, batch_size, t): # private method
    
        batch_data,batch_labels = self.__init_batch_data(batch_size)
        # We will also build a agumented batch data
        if self.affine:
            batch_data_aug,batch_labels_aug = self.__init_batch_data(batch_size)
        if self.flip:
            batch_data_flip,batch_labels_flip = self.__init_batch_data(batch_size)

        #create a list of image numbers you want to use for a particular video
        img_idx = [x for x in range(0, self.frames)] 

        for folder in range(batch_size): # iterate over the batch_size
            # read all the images in the folder
            imgs = sorted(os.listdir(source_path+'/'+ t[folder + (batch_num*batch_size)].split(';')[0])) 
            # Generate a random affine to be used in image transformation for buidling agumented data set
            M = self.__get_random_affine()
            
            #  Iterate over the frames/images of a folder to read them in
            for idx, item in enumerate(img_idx): 
                image = cv2.imread(source_path+'/'+ t[folder + (batch_num*batch_size)].strip().split(';')[0]+'/'+imgs[item], cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                #crop the images and resize them. Note that the images are of 2 different shape 
                #and the conv3D will throw error if the inputs in a batch have different shapes  
                image = self.__crop(image) # : image
                # If normalize is set normalize the image else use the raw image.
                resized = self.__normalize(self.__resize(image)) # : self.__resize(image)
                # If the input is edge detected image then use the sobelx, sobely and laplacian as 3 channel of the edge detected image
                resized = self.__edge(resized) # : resized

                batch_data[folder,idx] = resized
                if self.affine:
                    batch_data_aug[folder,idx] = self.__affine(resized)   
                if self.flip:
                    batch_data_flip[folder,idx] = self.__flip(resized)   

            batch_labels[folder, int(t[folder + (batch_num*batch_size)].strip().split(';')[2])] = 1
            
            if self.affine:
                batch_labels_aug[folder, int(t[folder + (batch_num*batch_size)].strip().split(';')[2])] = 1
            
            if self.flip:
                if int(t[folder + (batch*batch_size)].strip().split(';')[2])==0:
                    batch_labels_flip[folder, 1] = 1
                elif int(t[folder + (batch*batch_size)].strip().split(';')[2])==1:
                    batch_labels_flip[folder, 0] = 1
                else:
                    batch_labels_flip[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
        
        if self.affine:
            batch_data = np.append(batch_data, batch_data_aug, axis = 0) 
            batch_labels = np.append(batch_labels, batch_labels_aug, axis = 0) 
        if self.flip:
            batch_data = np.append(batch_data, batch_data_flip, axis = 0) 
            batch_labels = np.append(batch_labels, batch_labels_flip, axis = 0) 

        return batch_data, batch_labels
    
    def generator(self, source_path, folder_list, batch_size): # public method
        print( 'Source path = ', source_path, '; batch size =', batch_size)
        while True:
            t = np.random.permutation(folder_list)
            num_batches = len(folder_list)//batch_size # calculate the number of batches
            for batch in range(num_batches): # we iterate over the number of batches
                # you yield the batch_data and the batch_labels, remember what does yield do
                yield self.__load_batch_images(source_path, folder_list, batch, batch_size, t) 
            
            # Code for the remaining data points which are left after full batches
            if (len(folder_list) != batch_size*num_batches):
                batch_size = len(folder_list) - (batch_size*num_batches)
                yield self.__load_batch_images(source_path, folder_list, num_batches, batch_size, t)

    # Helper function to perfom affice transform on the image
    def __affine(self, image, M):
        return cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))

    # Helper function to flip the image
    def __flip(self, image):
        return np.flip(image,1)
    
    # Helper function to normalise the data
    def __normalize(self, image):
        return image/127.5-1
    
    # Helper function to resize the image
    def __resize(self, image):
        return cv2.resize(image, (self.width,self.height), interpolation = cv2.INTER_AREA)
    
    # Helper function to crop the image
    def __crop(self, image):
        if image.shape[0] != image.shape[1]:
            return image[0:120, 20:140]
        else:
            return image

    # Helper function for edge detection
    def __edge(self, image):
        return imag

class ModelGenerator(object):
    @classmethod
    def c3d1(cls, input_shape, nb_classes):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            8, (3,3,3), activation='relu', input_shape=input_shape
        ))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(16, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(32, (3,3,3), activation='relu'))
        model.add(Conv3D(32, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (2,2,2), activation='relu'))
        model.add(Conv3D(64, (2,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        return model
    
    @classmethod
    def c3d2(cls, input_shape, nb_classes):
        """
        Build a 3D convolutional network, aka C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """        
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(16, 3, 3, 3, activation='relu',
                         padding='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Conv3D(32, 3, 3, 3, activation='relu',
                         padding='same', name='conv2',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         padding='same', name='conv3a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         padding='same', name='conv3b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Conv3D(128, 3, 3, 3, activation='relu',
                         padding='same', name='conv4a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(128, 3, 3, 3, activation='relu',
                         padding='same', name='conv4b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))

        # 5th layer group
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         padding='same', name='conv5a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         padding='same', name='conv5b',
                         subsample=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               padding='valid', name='pool5'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(512, activation='relu', name='fc6'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu', name='fc7'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        return model
    
    @classmethod
    def c3d3(cls, input_shape, nb_classes):
        model = Sequential()
        model.add(Conv3D(16, kernel_size=(3, 3, 3), input_shape=input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv3D(16, padding="same", kernel_size=(3, 3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
        model.add(Activation('relu'))
        model.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
        model.add(Activation('relu'))
        model.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        return model
 
    @classmethod
    def c3d4(cls, input_shape, nb_classes):
        # Define model
        model = Sequential()

        model.add(Conv3D(8, kernel_size=(3,3,3), input_shape=input_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(2,2,2)))

        model.add(Conv3D(16, kernel_size=(3,3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(2,2,2)))

        model.add(Conv3D(32, kernel_size=(1,3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(2,2,2)))

        model.add(Conv3D(64, kernel_size=(1,3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(MaxPooling3D(pool_size=(2,2,2)))

        #Flatten Layers
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        #softmax layer
        model.add(Dense(5, activation='softmax'))
        return model
    
    @classmethod
    def c3d5(cls, input_shape, nb_classes):
        # Define model
        model = Sequential()

        model.add(Conv3D(8, kernel_size=(3,3,3), input_shape=input_shape, padding='same'))
        #model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(MaxPooling3D(pool_size=(2,2,2)))

        model.add(Conv3D(16, kernel_size=(3,3,3), padding='same'))
        #model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(MaxPooling3D(pool_size=(2,2,2)))

        model.add(Conv3D(32, kernel_size=(1,3,3), padding='same'))
        #model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(MaxPooling3D(pool_size=(2,2,2)))

        model.add(Conv3D(64, kernel_size=(1,3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(MaxPooling3D(pool_size=(2,2,2)))

        #Flatten Layers
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        #softmax layer
        model.add(Dense(5, activation='softmax'))
        return model   
    
    @classmethod
    def lstm(cls, input_shape, nb_classes):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=input_shape,
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        return model

    @classmethod
    def lrcn(cls, input_shape, nb_classes):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py
        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556
        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
            activation='relu', padding='same'), input_shape=input_shape))
        model.add(TimeDistributed(Conv2D(32, (3,3),
            kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        
        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        return model

    @classmethod
    def mlp(cls, input_shape, nb_classes):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

def train(batch_size, num_epochs, model, train_generator, val_generator, optimiser=None):
    curr_dt_time = datetime.datetime.now()
    num_train_sequences = len(train_doc)
    print('# training sequences =', num_train_sequences)
    num_val_sequences = len(val_doc)
    print('# validation sequences =', num_val_sequences)
    print('# batch size =', batch_size)    
    print('# epochs =', num_epochs)
    #optimizer = Adam(lr=rate) 
    #write your optimizer
    if optimiser == None:
        optimiser = Adam() 
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print (model.summary())
    model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    if not os.path.exists(model_name):
        os.mkdir(model_name)   
    filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'
    checkpoint = ModelCheckpoint(filepath, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=False, 
                                 save_weights_only=False, 
                                 mode='auto', 
                                 period=1)
    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)
    callbacks_list = [checkpoint, LR]
    if (num_train_sequences%batch_size) == 0:
        steps_per_epoch = int(num_train_sequences/batch_size)
    else:
        steps_per_epoch = (num_train_sequences//batch_size) + 1
    if (num_val_sequences%batch_size) == 0:
        validation_steps = int(num_val_sequences/batch_size)
    else:
        validation_steps = (num_val_sequences//batch_size) + 1
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                callbacks=callbacks_list, validation_data=val_generator, 
                validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
    K.clear_session()