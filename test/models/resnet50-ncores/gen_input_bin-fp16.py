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
from imagenet_preprocessing import _aspect_preserving_resize, _central_crop, _mean_image_subtraction, _RESIZE_MIN, _CHANNEL_MEANS
import tensorflow as tf


def preprocessor_all(num):
    skip=[4476,8630,8889,9010,12084,14274,17329,18337,20858,20908,23222,29216,31906,37629,37881]
    os.system("mkdir -p done")
    for i in range(num):
        if (os.path.exists('done/ILSVRC2012_val_000%05d.npy' %(i+1))) :
            continue
        if (i+1) in skip:
            continue
        print("Processing %d"%(i+1))
        #load jpg
        jpg=open('dataset/ILSVRC2012_val_000%05d.JPEG' %(i+1), 'rb')
        image= tf.image.decode_jpeg(jpg.read(), channels = 3)
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, 224, 224)

        x = _mean_image_subtraction(image, _CHANNEL_MEANS, 3)
        x = tf.expand_dims(x, 0)
        x = np.array(x)
        x = x.astype(np.float16)
        np.save('done/ILSVRC2012_val_000%05d.npy' %(i+1), x)

    with open("bindata/imagenet_pic_data.bin", 'wb') as f:
        for i in range(64):
            with open('done/ILSVRC2012_val_000%05d.npy' %(i+1), "rb") as fimg:
                data = fimg.readlines()
                for j in range(len(data)-1):
                    f.write(data[j+1])

    os.system("rm -rf done")

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
 
    preprocessor_all(image_count)
 
    # r = open('tf_result-fp16.txt', 'w+')
    # for i in range(image_count):
    #     img = imread('dataset/'+image_names[i])
    #     print('dataset/'+image_names[i], end=": ")
    #     img = resize(img, (224, 224, 3)) * 255    # cast back to 0-255 range
    #     img = preprocess_input(img)
    #     # fp16
    #     img = img.astype(np.float16)
    #     img = np.expand_dims(img, 0)

    #     # predict the result by using Keras directly
    #     img_pred = model.predict(img)
    #     # decode the keras prediction
    #     pred_dec_k = decode_predictions(img_pred)
    #     print(pred_dec_k[0][0][0] + ',' + pred_dec_k[0][0][1] + ',' + str(pred_dec_k[0][0][2]))
    #     r.write(pred_dec_k[0][0][0] + ',' + pred_dec_k[0][0][1] + ',' + str(pred_dec_k[0][0][2]) + '\n')
    # r.close()



if __name__ == '__main__':
    os.system("rm -rf bindata && mkdir -p bindata")
    main()
