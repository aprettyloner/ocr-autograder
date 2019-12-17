import cv2
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle

def make_pixel_array(df):
    im = Image.open(df['path'],'r').convert('L')
    result = np.asarray(im)  
    im.close()
    return result 


def read_process_images(img_list,y,invert=False):
    # 28 x 28 to match MNIST data shape
    nrows = 28
    ncolumns = 28
    channels = 1
    img_list,y = shuffle(img_list,y)
    X=[]
    for path in img_list:     
        X.append(cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(nrows,ncolumns),interpolation=cv2.INTER_CUBIC))
    if invert==True:
        return 255-np.array(X),np.array(y)
    return np.array(X),np.array(y)


def load_HASY():
    # Pregenerated csv for ease
    file = 'HASYv2/hasy-data-labels.csv'
    df = pd.read_csv(file)

    # Limit to digits, ignore symbols/letters
    digits = df[(df['symbol_id']<=79) & (df['symbol_id']>=70)]
    digits['path'] = [ 'HASYv2/' + path for path in digits['path']]
    digits['pixels']=digits.apply(make_pixel_array,axis=1)

    X,y = read_process_images(digits['path'],digits['latex'],invert=True)
    return X, y

def load_Kensanata(dataframe=False):
    # Pre-labeled images for ease
    folder = 'Kensanata/'
    imgs = os.popen("ls "+folder).read().split('\n')[:-1]
    for i,img in enumerate(imgs):
        imgs[i]=folder+img
    
    # Create data labels array
    labels = []
    for filename in imgs:
        for digit in filename.split('/')[1][-5:-4]:
            labels.append(int(digit))
    
    # Extra demographic data
    country = []
    age = []
    gender = []
    for FILE in imgs:
        country.append(FILE.split('_')[1][:2])
        try:
            age.append(int(FILE.split('_')[1][2])*10)
        except:
            age.append(np.nan)
        gender.append(FILE.split('_')[1][3])
    
    df = pd.DataFrame(imgs)
    df.columns = ['path']
    df['label'] = labels
    df['country'] = country
    df['gender'] = gender
    df['age'] = age

    # Create image arrays from png
    df['array']=df.apply(make_pixel_array,axis=1)

    X = df['array']
    y = df['label']

    if dataframe:
        return df
    else:
        return X,y

