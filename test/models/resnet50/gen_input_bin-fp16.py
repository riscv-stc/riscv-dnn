import numpy as np
import os
import random
import subprocess
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
# from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
# from keras.utils import get_file
# from classification_models.models.resnet import ResNet18, preprocess_input
# from classification_models.keras import Classifiers
# ResNet18, preprocess_input = Classifiers.get('resnet18')
from keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet import ResNet101 as resnet
def main():
    # fp16
    K.set_floatx('float16')

    # load model
    model = resnet(weights='imagenet', input_shape=(224, 224, 3))

    image_names = []
    file_names=sorted(os.listdir('dataset'))
    for i in file_names:
        if "ILS" in i:
            image_names.append(i)
    image_count = len(image_names)
 
    f = open('bindata/imagenet_pic_data.bin', 'wb+')
 
    r = open('tf_result-fp16.txt', 'w+')
    for i in range(image_count):
        img = imread('dataset/'+image_names[i])
        print('dataset/'+image_names[i], end=": ")
        img = resize(img, (224, 224, 3)) * 255    # cast back to 0-255 range
        img = preprocess_input(img)
        # fp16
        img = img.astype(np.float16)
        img = np.expand_dims(img, 0)

        # predict the result by using Keras directly
        img_pred = model.predict(img)
        # decode the keras prediction
        pred_dec_k = decode_predictions(img_pred)
        print(pred_dec_k[0][0][0] + ',' + pred_dec_k[0][0][1] + ',' + str(pred_dec_k[0][0][2]))
        r.write(pred_dec_k[0][0][0] + ',' + pred_dec_k[0][0][1] + ',' + str(pred_dec_k[0][0][2]) + '\n')

        # write bin data
        img = img.astype(np.float16).flatten()
        #the dtype change is just for right format of data store
        img.dtype='uint16'
        #for value in img:
        #    f.write(hex(value) + ',')
        #f.write('\n')
        for value in img:
            f.write(value)
    f.close()
    r.close()

if __name__ == '__main__':
    main()
