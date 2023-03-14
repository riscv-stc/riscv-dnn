from cgi import print_form
from filecmp import DEFAULT_IGNORES
import numpy as np
import os
import sys
sys.path.append("../../../utils") 
from check import from_txt, check_to_txt, get_sig_addr
from work import do_test

# Tensorflow imports
import tensorflow.compat.v1 as tf
# Tensorflow utility functions
from imagenet_preprocessing import (
    _aspect_preserving_resize,
    _central_crop,
    _mean_image_subtraction,
    _RESIZE_MIN,
    _CHANNEL_MEANS,
)

sys.path.append("../..") 
from check import *

title = "8 CORES for Resnet50 Net"
opt_levels = {"16": "-O2 -D__RVM__"}

pwd = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(pwd, "resnet-50_v2.pb")
img_url = os.path.join(pwd, "dataset")
golden_dir = os.path.join(pwd, "golden")


# preprocess one picture
def preprocess(img_path):
    jpg = open(img_path, "rb")
    image = tf.image.decode_jpeg(jpg.read(), channels=3)
    image = _aspect_preserving_resize(image, _RESIZE_MIN)
    image = _central_crop(image, 224, 224)

    x = _mean_image_subtraction(image, _CHANNEL_MEANS, 3)
    x = tf.expand_dims(x, 0)
    return x


# prepare input data
def prepare_input(num):
    x_all = []
    for i in range(num):
        # load jpg
        img_path = os.path.join(img_url, "ILSVRC2012_val_000000%02d.JPEG" % (i + 1))
        x = preprocess(img_path)
        x_all.append(x)

    tx = tf.concat([x for x in x_all], 0)
    with tf.Session() as session:
        x = tx.eval()
    return x


def get_golden(num, layer, BATCH):
    if 'dense' in layer:
        layername = layer.split('/')[2]
    elif 'softmax' in layer:
        layername = layer
    else:
        layername = layer.split('/')[1]
    layername = layername + '.bin'
    layer0 = layer + ":0"
    ######################################################################
    # Import model
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        new_input_tensor = tf.placeholder(shape=(16, 224, 224, 3), dtype="float16", name="input_tensor")
        tf.import_graph_def(graph_def, name="", input_map={"input_tensor": new_input_tensor})
        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name(layer0)
            predictions = sess.run(softmax_tensor, {"input_tensor:0": x})
            predictions.tofile(os.path.join(golden_dir, layername))


if __name__ == "__main__" :
    # params:
    #   BATCH
    ## check
    tf.disable_eager_execution()
    # set number pictures to predict

    x = prepare_input(16)
    # stage 1
    get_golden(2,  "resnet_model/conv2d/Conv2D", 7*7*3)
    # get_golden(3,  "resnet_model/max_pooling2d/MaxPool", 3*3)
    # get_golden(4,  "resnet_model/batch_normalization/FusedBatchNormV2", 7*7*3)
    get_golden(5,  "resnet_model/Relu", 7*7*3)
    # get_golden(6,  "resnet_model/conv2d_1/Conv2D", 200)
    # get_golden(7,  "resnet_model/conv2d_2/Conv2D", 200)
    # get_golden(8,  "resnet_model/batch_normalization_1/FusedBatchNormV2", 200)
    get_golden(9,  "resnet_model/Relu_1", 200)
    # get_golden(10, "resnet_model/conv2d_3/Conv2D", 200)
    # get_golden(11, "resnet_model/batch_normalization_2/FusedBatchNormV2", 200)
    get_golden(12, "resnet_model/Relu_2", 200)
    # get_golden(13, "resnet_model/conv2d_4/Conv2D", 200)
    get_golden(14, "resnet_model/add", 200)
    # get_golden(15, "resnet_model/batch_normalization_3/FusedBatchNormV2", 200)
    get_golden(16, "resnet_model/Relu_3", 200)
    # get_golden(17, "resnet_model/conv2d_5/Conv2D", 200)
    # get_golden(18, "resnet_model/batch_normalization_4/FusedBatchNormV2", 200)
    get_golden(19, "resnet_model/Relu_4", 200)
    # get_golden(20, "resnet_model/conv2d_6/Conv2D", 200)
    # get_golden(21, "resnet_model/batch_normalization_5/FusedBatchNormV2", 200)
    get_golden(22, "resnet_model/Relu_5", 200)
    # get_golden(23, "resnet_model/conv2d_7/Conv2D", 200)
    
    get_golden(24, "resnet_model/add_1", 200)
    # get_golden(25, "resnet_model/batch_normalization_6/FusedBatchNormV2", 200)
    get_golden(26, "resnet_model/Relu_6", 200)
    # get_golden(27, "resnet_model/conv2d_8/Conv2D", 200)
    # get_golden(28, "resnet_model/batch_normalization_7/FusedBatchNormV2", 200)
    get_golden(29, "resnet_model/Relu_7", 200)
    # get_golden(30, "resnet_model/conv2d_9/Conv2D", 200)
    # get_golden(31, "resnet_model/batch_normalization_8/FusedBatchNormV2", 200)
    get_golden(32, "resnet_model/Relu_8", 200)
    # get_golden(33, "resnet_model/conv2d_10/Conv2D", 200)
    # get_golden(34, "resnet_model/add_2", 200)
    # get_golden(35, "resnet_model/batch_normalization_9/FusedBatchNormV2", 200)
    get_golden(36, "resnet_model/Relu_9", 200)
    
    # # stage 2
    # get_golden(37, "resnet_model/conv2d_11/Conv2D", 200)
    # # get_golden(38, "resnet_model/conv2d_12/Conv2D", 200)
    # # get_golden(39, "resnet_model/batch_normalization_10/FusedBatchNormV2", 200)
    # get_golden(40, "resnet_model/Relu_10", 200)
    # # get_golden(41, "resnet_model/conv2d_13/Conv2D", 200)
    # # get_golden(42, "resnet_model/batch_normalization_11/FusedBatchNormV2", 200)
    # get_golden(43, "resnet_model/Relu_11", 200)
    # # get_golden(44, "resnet_model/conv2d_14/Conv2D", 200)
    
    # get_golden(45, "resnet_model/add_3", 200)
    # # get_golden(46, "resnet_model/batch_normalization_12/FusedBatchNormV2", 200)
    # get_golden(47, "resnet_model/Relu_12", 200)
    # # get_golden(48, "resnet_model/conv2d_15/Conv2D", 200)
    # # get_golden(49, "resnet_model/batch_normalization_13/FusedBatchNormV2", 200)
    # get_golden(50, "resnet_model/Relu_13", 200)
    # # get_golden(51, "resnet_model/conv2d_16/Conv2D", 200)
    # # get_golden(52, "resnet_model/batch_normalization_14/FusedBatchNormV2", 200)
    # get_golden(53, "resnet_model/Relu_14", 200)
    # # get_golden(54, "resnet_model/conv2d_17/Conv2D", 200)

    # get_golden(55, "resnet_model/add_4", 200)
    # # get_golden(56, "resnet_model/batch_normalization_15/FusedBatchNormV2", 200)
    # get_golden(57, "resnet_model/Relu_15", 200)
    # # get_golden(58, "resnet_model/conv2d_18/Conv2D", 200)
    # # get_golden(59, "resnet_model/batch_normalization_16/FusedBatchNormV2", 200)
    # get_golden(60, "resnet_model/Relu_16", 200)
    # # get_golden(61, "resnet_model/conv2d_19/Conv2D", 200)
    # # get_golden(62, "resnet_model/batch_normalization_17/FusedBatchNormV2", 200)
    # get_golden(63, "resnet_model/Relu_17", 200)
    # # get_golden(64, "resnet_model/conv2d_20/Conv2D", 200)

    # get_golden(65, "resnet_model/add_5", 200)
    # # get_golden(66, "resnet_model/batch_normalization_18/FusedBatchNormV2", 200)
    # get_golden(67, "resnet_model/Relu_18", 200)
    # # get_golden(68, "resnet_model/conv2d_21/Conv2D", 200)
    # # get_golden(69, "resnet_model/batch_normalization_19/FusedBatchNormV2", 200)
    # get_golden(70, "resnet_model/Relu_19", 200)
    # # get_golden(71, "resnet_model/conv2d_22/Conv2D", 200)
    # # get_golden(72, "resnet_model/batch_normalization_20/FusedBatchNormV2", 200)
    # get_golden(73, "resnet_model/Relu_20", 200)
    # # get_golden(74, "resnet_model/conv2d_23/Conv2D", 200)

    # get_golden(75, "resnet_model/add_6", 200)
    # # get_golden(76, "resnet_model/batch_normalization_21/FusedBatchNormV2", 200)
    # get_golden(77, "resnet_model/Relu_21", 200)

    # stage 3
    # get_golden(78, "resnet_model/conv2d_24/Conv2D", 200)
    # # get_golden(79, "resnet_model/conv2d_25/Conv2D", 200)
    # # get_golden(80, "resnet_model/batch_normalization_22/FusedBatchNormV2", 200)
    # get_golden(81, "resnet_model/Relu_22", 200)
    # # get_golden(82, "resnet_model/conv2d_26/Conv2D", 200)
    # # get_golden(83, "resnet_model/batch_normalization_23/FusedBatchNormV2", 200)
    # get_golden(84, "resnet_model/Relu_23", 200)
    # # get_golden(85, "resnet_model/conv2d_27/Conv2D", 200)

    # # get_golden(86, "resnet_model/add_7", 200)
    # # get_golden(87, "resnet_model/batch_normalization_24/FusedBatchNormV2", 200)
    # get_golden(88, "resnet_model/Relu_24", 200)
    # # get_golden(89, "resnet_model/conv2d_28/Conv2D", 200)
    # # get_golden(90, "resnet_model/batch_normalization_25/FusedBatchNormV2", 200)
    # get_golden(91, "resnet_model/Relu_25", 200)
    # # get_golden(91, "resnet_model/conv2d_29/Conv2D", 200)
    # # get_golden(93, "resnet_model/batch_normalization_26/FusedBatchNormV2", 200)
    # get_golden(94, "resnet_model/Relu_26", 200)
    # # get_golden(95, "resnet_model/conv2d_30/Conv2D", 200)

    # # get_golden(96, "resnet_model/add_8", 200)
    # # get_golden(97, "resnet_model/batch_normalization_27/FusedBatchNormV2", 200)
    # get_golden(98, "resnet_model/Relu_27", 200)
    # # get_golden(99, "resnet_model/conv2d_31/Conv2D", 200)
    # # get_golden(100, "resnet_model/batch_normalization_28/FusedBatchNormV2", 200)
    # get_golden(101, "resnet_model/Relu_28", 200)
    # # get_golden(102, "resnet_model/conv2d_32/Conv2D", 200)
    # # get_golden(103, "resnet_model/batch_normalization_29/FusedBatchNormV2", 200)
    # get_golden(104, "resnet_model/Relu_29", 200)
    # # get_golden(105, "resnet_model/conv2d_33/Conv2D", 200)

    # # get_golden(106, "resnet_model/add_9", 200)
    # # get_golden(107, "resnet_model/batch_normalization_30/FusedBatchNormV2", 200)
    # get_golden(108, "resnet_model/Relu_30", 200)
    # # get_golden(109, "resnet_model/conv2d_34/Conv2D", 200)
    # # get_golden(110, "resnet_model/batch_normalization_31/FusedBatchNormV2", 200)
    # get_golden(111, "resnet_model/Relu_31", 200)
    # # get_golden(112, "resnet_model/conv2d_35/Conv2D", 200)
    # # get_golden(113, "resnet_model/batch_normalization_32/FusedBatchNormV2", 200)
    # get_golden(114, "resnet_model/Relu_32", 200)
    # # get_golden(115, "resnet_model/conv2d_36/Conv2D", 200)

    # # get_golden(116, "resnet_model/add_10", 200)
    # # get_golden(117, "resnet_model/batch_normalization_33/FusedBatchNormV2", 200)
    # get_golden(118, "resnet_model/Relu_33", 200)
    # # get_golden(119, "resnet_model/conv2d_37/Conv2D", 200)
    # # get_golden(120, "resnet_model/batch_normalization_35/FusedBatchNormV2", 200)
    # get_golden(121, "resnet_model/Relu_34", 200)
    # # get_golden(122, "resnet_model/conv2d_38/Conv2D", 200)
    # # get_golden(123, "resnet_model/batch_normalization_35/FusedBatchNormV2", 200)
    # get_golden(124, "resnet_model/Relu_35", 200)
    # # get_golden(125, "resnet_model/conv2d_39/Conv2D", 200)

    # get_golden(126, "resnet_model/add_11", 200)
    # get_golden(127, "resnet_model/batch_normalization_36/FusedBatchNormV2", 200)
    # get_golden(128, "resnet_model/Relu_36", 200)
    # # get_golden(129, "resnet_model/conv2d_40/Conv2D", 200)
    # # get_golden(130, "resnet_model/batch_normalization_37/FusedBatchNormV2", 200)
    # get_golden(131, "resnet_model/Relu_37", 200)
    # # get_golden(132, "resnet_model/conv2d_41/Conv2D", 200)
    # # get_golden(133, "resnet_model/batch_normalization_38/FusedBatchNormV2", 200)
    # get_golden(134, "resnet_model/Relu_38", 200)
    # # get_golden(135, "resnet_model/conv2d_42/Conv2D", 200)

    # # get_golden(136, "resnet_model/add_12", 200)
    # # get_golden(137, "resnet_model/batch_normalization_39/FusedBatchNormV2", 200)
    # get_golden(138, "resnet_model/Relu_39", 200)
    
    # # stage 4
    # get_golden(139, "resnet_model/Relu_42", 200)
    # get_golden(140, "resnet_model/Relu_45", 200)
    # get_golden(141, "resnet_model/Relu_48", 200)

    
    # get_golden(142, "resnet_model/Mean", 200)
    # get_golden(143, "resnet_model/dense/MatMul", 200)
    # get_golden(144, "resnet_model/dense/BiasAdd", 200)
    get_golden(145, "softmax_tensor_fp16", 200)

    
