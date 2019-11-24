import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import my_app.processing1 as proc


from PIL import Image
from PIL import ImageFilter
from math import ceil
from datetime import datetime

# Modeling
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def join_datasets(x_arr,y_arr):
    '''
    Joins MNIST and Kensanata arrays and labels into one train set.
    
    Input:  
        [x_arr] array of Kensanata pixel arrays
        [y_arr] array of Kensanata image labels
    Output: 
        [data]  dataframe of MNIST and Kensanata data
                with columns:
                    [X] pixel arrays
                    [y] labels
    '''
    (X_tr,y_tr),(X_ts,y_ts) = mnist.load_data()
    X_joined = np.concatenate((x_arr,X_tr, X_ts), axis=0)
    y_joined = np.concatenate((y_arr,y_tr, y_ts), axis=0)
    data = pd.DataFrame(y_joined)
    data.columns = ['y']
    data['X'] = X_joined

    data.to_csv('final_train_data.csv',index=False)
    return data



def prepare_kensanata():
    '''
    Converts Kensanata image files to pixel arrays with MNIST properties
        - Black background, white object
        - Centered and padded
        - Thick object
        - Size: 28 x 28
    
    Input:  
        none
    Output: 
        [x_arr] array of 28x28 pixel arrays
        [y_arr] array of image labels
    '''

    ### read paths for Kensanata images (40,799)
    images = os.popen("ls Kensanata/").read().split('\n')[:-1]
    for i,img in enumerate(images):
        images[i]='Kensanata/'+img

    file_arr = []
    x_arr = []
    y_arr = []
    subset = images
    for i,file in enumerate(subset):
        if i%500 == 0:
            print(i,'of',len(subset))
        answer=file[-5:-4]
        binary_arr,label_arr, segments,orig = proc.label_segments(file,'',photo=False,marker=False)
        found = binary_arr==0
        try:
            x,y = np.where(found)
            xmin,xmax,ymin,ymax = np.min(x),np.max(x),np.min(y),np.max(y)
            xlen,ylen = found[xmin:xmax,ymin:ymax].shape
            diff = np.abs(ylen-xlen)
            change = ceil(diff/2)
            if diff!=0:
                if ylen>xlen:
                    xmin-=change
                    xmax+=change
                else:
                    ymin-=change
                    ymax+=change

                xlen,ylen = xmax-xmin,ymax-ymin
                diff=np.abs(ylen-xlen)
                if xlen>ylen: ymax+=diff
                elif ylen>xlen: xmax+=diff
            digit = 1-binary_arr[xmin:xmax,ymin:ymax]
            digit = np.pad(digit,int(len(digit)*.2),mode= 'constant', constant_values=(0,0))                  
            im = Image.fromarray(np.array(digit)*255.0).convert("RGB")
            im.save('000.png')
            del im
            del digit
            img = cv2.resize(cv2.imread('000.png',cv2.IMREAD_GRAYSCALE),(28,28),interpolation=cv2.INTER_CUBIC)
            img = cv2.blur(img, (2, 2))
            os.remove('000.png')
            file_arr.append(file)
            x_arr.append(img)
            y_arr.append(answer)
        except:
            pass
    y_arr = [int(y) for y in y_arr]
    df = pd.DataFrame(x_arr,y_arr, columns=['X','y'])
    df.to_csv('kensanata.csv')
    return np.array(x_arr), np.array(y_arr)


def create_model(data,train_size=.80,batch_size=32,epochs=20):
    '''
    Given a dataframe with X and y train data, creates and saves a CNN model.
    
    Input:  
        [data]      dataframe of MNIST and Kensanata data
                        with columns:
                            [X] pixel arrays
                            [y] labels       
        [train_size]    (optional) default value = 0.80
        [batch_size]    (optional) default value = 32
        [epochs]        (optional) default value = 20

    Output: 
        [conv_model]    trained CNN model

    '''

    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'],train_size=train_size)

    ## Model parameters
    conv_model = models.Sequential()
    conv_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.BatchNormalization())
    conv_model.add(layers.Dropout(0.4))
    conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.BatchNormalization())
    conv_model.add(layers.Dropout(0.4))
    conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(layers.Flatten())
    conv_model.add(layers.Dense(256, activation = "relu"))
    conv_model.add(layers.Dropout(0.4))
    conv_model.add(layers.Dense(64, activation='relu'))
    conv_model.add(layers.Dropout(0.4))
    conv_model.add(layers.Dense(len(set(list(data.y))), activation='softmax'))
    conv_model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    ## Reshape input data
    x_train = X_train.astype('int32')
    x_test = X_test.astype('int32')
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    x_train = np.array([x.reshape(28,28,1) for x in x_train])
    x_test = np.array([x.reshape(28,28,1) for x in x_test])    

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    # dummy_y_train = np_utils.to_categorical(y_train)#.astype('int32')
    dummy_y_train = np_utils.to_categorical(pd.DataFrame(y_train)[0].factorize()[0])

    encoder = LabelEncoder()
    encoder.fit(y_test)
    encoded_y_test = encoder.transform(y_test)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_test = np_utils.to_categorical(pd.DataFrame(y_test)[0].factorize()[0])
    # dummy_y_test = np_utils.to_categorical(y_test)#.astype('int32')
    conv_model.fit(x_train, dummy_y_train, batch_size=batch_size, epochs=epochs)

    saved_model_path = "tf_model__{}.h5".format(datetime.now().strftime("%Y%m%d%s")) # _%H%M%S
    # Save entire model to a HDF5 file
    conv_model.save(saved_model_path)

    return conv_model


def from_np_array(array_string):
    array_string = array_string.replace('\n','').replace('   ',',').replace('[  ','[').replace('  ',',').replace(' ',',')
    return np.array(ast.literal_eval(array_string)).reshape((28,28))