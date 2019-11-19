import os
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template

from tensorflow.keras import models
from predict import predict_tf
#from predict_withsymbols import predict_tf
from predict import *
#model='static/mnist_hasyv2_master_20epochs_batch64__ALLDATA_201911081573211546.h5'
#model='static/nosymbols.h5'
model='static/mnist_hasyv2_master_20epochs_batch64__ALLDATA_201911081573211546.h5'
tf_model = models.load_model(model)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

import boto3

# Let's use Amazon S3
s3 = boto3.resource('s3')

#tf_model = models.load_model('static/mnist_hasyv2_master_20epochs_batch64_201911081573209782.h5')  #tf_model.h5
#tf_model = models.load_model('static/tf_model.h5')
#tf_model = models.load_model('static/mnist_hasyv2_master_20epochs_batch64__ALLDATA_201911081573211546.h5')

def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
        # return render_template('upload.html')
        return render_template('index.html')

@app.route('/predict')
def predict():
        # return render_template('crop.html')
        filename = '~/Downloads/imagename.png'
        predictions = predict_tf(tf_model,filename)
        return str(predictions)

@app.route('/mobile')
def crop():
        # return render_template('crop.html')
        return render_template('testcrop.html')   ##credit to Pen by Moncho Varela

# @app.route('/landingpage/<imgurl>',methods=['GET', 'POST', 'PUT'])
# def landingpage(imgurl):
@app.route('/landingpage',methods=['GET', 'POST', 'PUT'])
def landingpage():
        

        # handle both GET (in url) and POST (in payload) requests
        if request.method=="GET":
                url=request.args['imgurl']
                ans=request.args['answer']
        else:
                url=request.form['imgurl']
                ans=request.form['answer']
        imgfile,typee  = urllib.request.urlretrieve(str(url) )
        arr=cv2.imread(imgfile) 
        im = Image.fromarray(arr)
        tempfile = 'cropped_image.jpg'
        im.save(tempfile)
        predictions = predict_tf(tf_model,tempfile)
        s = [str(i) for i in predictions] 
        response = ''.join(s)
	
        outcome = 'Incorrect ['+response+']'
        if response.replace(' ','') == ans.replace(' ',''):
                outcome = 'Correct! ['+response+']'
        returnstring = "<h4>Answer Key: \t"+ans+'<br>Student Response: \t'+response+"<br><br><a href='/mobile'><img src='https://image.flaticon.com/icons/svg/13/13964.svg' alt='Go Back' width=50px/></h4></a>"
        if ans=='Enter Answer' or ans=='':
                outcome = 'Detected ['+response+']'
                ans = 'NONE PROVIDED'
        returnstring = "<h4>Answer Key: \t"+ans+'<br>Student Response: \t'+response+"<br><br><a href='javascript:history.back()'><img src='https://image.flaticon.com/icons/svg/13/13964.svg'  alt='Go Back' width=50px/></h4></a>"
        return '<img src=\"'+url+'\" width=\"700\" ><br><h1>'+outcome+'</h1>' +returnstring




if __name__ == "__main__":
    app.run()






