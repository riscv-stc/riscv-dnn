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

title = "Single CORES for Resnet50 Net"
opt_levels = {"loop4": "-O2 -D__RVM__ -DNLOOPS=4", "loop8": "-O2 -D__RVM__ -DNLOOPS=8"}


pwd = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(pwd, "resnet-50_v2.pb")
img_url = os.path.join(pwd, "dataset")

simulator = 'spike'
begin_addr = 0
if len(sys.argv) > 1:
    simulator = sys.argv[1]

if simulator == 'spike':
    opt_levels = {"1": "-O2 -D__RVM__ -DNLOOPS=1 "}

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


def get_golden_check(num, layer, BATCH):
    begin_addr = get_sig_addr(f"build/{num}/test.map", "begin_signature")
    print("begin_signature: ", hex(begin_addr))

    ## check
    tf.disable_eager_execution()
    # set number pictures to predict

    x = prepare_input(BATCH)

    if 'dense' in layer:
        layername = layer.split('/')[2]
    elif 'softmax' in layer:
        layername = layer
    else:
        layername = layer.split('/')[1]

    layer0 = layer + ":0"
    ######################################################################
    # Import model
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        new_input_tensor = tf.placeholder(shape=(BATCH, 224, 224, 3), dtype="float16", name="input_tensor")
        tf.import_graph_def(graph_def, name="", input_map={"input_tensor": new_input_tensor})
        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name(layer0)
            predictions = sess.run(softmax_tensor, {"input_tensor:0": x})
            sig_addr = get_sig_addr(f"build/{num}/test.map", layername+"_data")
            start_offset = sig_addr - begin_addr
            print(layername, "addr: ", hex(sig_addr), "offset: ", start_offset)
            result = from_txt(f'build/{num}/{simulator}.sig', predictions, start_offset)
            os.makedirs('check', exist_ok=True)
            check_result = check_to_txt( predictions, result, f'check/{layername}.data', 'np.allclose( result, golden, rtol=1e-2, atol=1e-2, equal_nan=True)' )
            print(f"> {num}, check result: {check_result}")
            print(str(num)+"-"+layer, predictions.shape)
            
            
def test(num, params, defs, ncores=8):
    BATCH, *extras = params
    extras = None

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")
    os.system(f"make clean && make DEFS='{defs} -DBATCH={BATCH}' run SIM={simulator} NUM={num}  > build/{num}/test.log 2>&1")

    # get_golden_check(num, "resnet_model/Relu", BATCH)
    # # # get_golden_check(num,  "resnet_model/conv2d_1/Conv2D", BATCH)
    # # # get_golden_check(num,  "resnet_model/Relu_1", BATCH)
    # # # get_golden_check(num,  "resnet_model/Relu_4", BATCH)
    # get_golden_check(num, "resnet_model/Relu_9", BATCH)
    # get_golden_check(num, "resnet_model/Relu_21", BATCH)
    # get_golden_check(num, "resnet_model/Relu_39", BATCH)
    # # get_golden_check(num, "resnet_model/Relu_40", BATCH)
    # # get_golden_check(num, "resnet_model/Relu_41", BATCH)
    # # get_golden_check(num, "resnet_model/Relu_44", BATCH)
    # # get_golden_check(num, "resnet_model/Relu_48", BATCH)
    # get_golden_check(num, "resnet_model/Mean", BATCH)
    # get_golden_check(num, "resnet_model/dense/MatMul", BATCH)
    # get_golden_check(num, "softmax_tensor_fp16", BATCH)

if __name__ == "__main__" :
    ## run at first time
    # os.system("rm *.o")
    # os.system("python3 gen_weight_bin.py")
    # os.system("python3 gen_input_bin-fp16.py")
    # os.system("python3 gen_header.py")
    # os.system("python3 gen_padding_input.py")
    
    # params:
    #   BATCH
    params = (
        (1,),
    )
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')
        # stage 1
        # get_golden_check(2,  "resnet_model/conv2d/Conv2D", 7*7*3)
        # get_golden_check(3,  "resnet_model/max_pooling2d/MaxPool", 3*3)
        # get_golden_check(4,  "resnet_model/batch_normalization/FusedBatchNormV2", 7*7*3)
        # get_golden_check(5,  "resnet_model/Relu", 7*7*3)
        # # get_golden_check(6,  "resnet_model/conv2d_1/Conv2D", 200)
        # # get_golden_check(7,  "resnet_model/conv2d_2/Conv2D", 200)
        # # get_golden_check(8,  "resnet_model/batch_normalization_1/FusedBatchNormV2", 200)
        # get_golden_check(9,  "resnet_model/Relu_1", 200)
        # # get_golden_check(10, "resnet_model/conv2d_3/Conv2D", 200)
        # # get_golden_check(11, "resnet_model/batch_normalization_2/FusedBatchNormV2", 200)
        # get_golden_check(12, "resnet_model/Relu_2", 200)
        # # get_golden_check(13, "resnet_model/conv2d_4/Conv2D", 200)
    
        # get_golden_check(14, "resnet_model/add", 200)
        # # get_golden_check(15, "resnet_model/batch_normalization_3/FusedBatchNormV2", 200)
        # get_golden_check(16, "resnet_model/Relu_3", 200)
        # # get_golden_check(17, "resnet_model/conv2d_5/Conv2D", 200)
        # # get_golden_check(18, "resnet_model/batch_normalization_4/FusedBatchNormV2", 200)
        # get_golden_check(19, "resnet_model/Relu_4", 200)
        # # get_golden_check(20, "resnet_model/conv2d_6/Conv2D", 200)
        # # get_golden_check(21, "resnet_model/batch_normalization_5/FusedBatchNormV2", 200)
        # get_golden_check(22, "resnet_model/Relu_5", 200)
        # # get_golden_check(23, "resnet_model/conv2d_7/Conv2D", 200)
        
        # get_golden_check(24, "resnet_model/add_1", 200)
        # # get_golden_check(25, "resnet_model/batch_normalization_6/FusedBatchNormV2", 200)
        # get_golden_check(26, "resnet_model/Relu_6", 200)
        # # get_golden_check(27, "resnet_model/conv2d_8/Conv2D", 200)
        # # get_golden_check(28, "resnet_model/batch_normalization_7/FusedBatchNormV2", 200)
        # get_golden_check(29, "resnet_model/Relu_7", 200)
        # # get_golden_check(30, "resnet_model/conv2d_9/Conv2D", 200)
        # # get_golden_check(31, "resnet_model/batch_normalization_8/FusedBatchNormV2", 200)
        # get_golden_check(32, "resnet_model/Relu_8", 200)
        # # get_golden_check(33, "resnet_model/conv2d_10/Conv2D", 200)
        # # get_golden_check(34, "resnet_model/add_2", 200)
        # # get_golden_check(35, "resnet_model/batch_normalization_9/FusedBatchNormV2", 200)
        # get_golden_check(36, "resnet_model/Relu_9", 200)
        
        # # stage 2
        # get_golden_check(37, "resnet_model/conv2d_11/Conv2D", 200)
        # # get_golden_check(38, "resnet_model/conv2d_12/Conv2D", 200)
        # # get_golden_check(39, "resnet_model/batch_normalization_10/FusedBatchNormV2", 200)
        # get_golden_check(40, "resnet_model/Relu_10", 200)
        # # get_golden_check(41, "resnet_model/conv2d_13/Conv2D", 200)
        # # get_golden_check(42, "resnet_model/batch_normalization_11/FusedBatchNormV2", 200)
        # get_golden_check(43, "resnet_model/Relu_11", 200)
        # # get_golden_check(44, "resnet_model/conv2d_14/Conv2D", 200)
        
        # get_golden_check(45, "resnet_model/add_3", 200)
        # # get_golden_check(46, "resnet_model/batch_normalization_12/FusedBatchNormV2", 200)
        # get_golden_check(47, "resnet_model/Relu_12", 200)
        # # get_golden_check(48, "resnet_model/conv2d_15/Conv2D", 200)
        # # get_golden_check(49, "resnet_model/batch_normalization_13/FusedBatchNormV2", 200)
        # get_golden_check(50, "resnet_model/Relu_13", 200)
        # # get_golden_check(51, "resnet_model/conv2d_16/Conv2D", 200)
        # # get_golden_check(52, "resnet_model/batch_normalization_14/FusedBatchNormV2", 200)
        # get_golden_check(53, "resnet_model/Relu_14", 200)
        # # get_golden_check(54, "resnet_model/conv2d_17/Conv2D", 200)

        # get_golden_check(55, "resnet_model/add_4", 200)
        # # get_golden_check(56, "resnet_model/batch_normalization_15/FusedBatchNormV2", 200)
        # get_golden_check(57, "resnet_model/Relu_15", 200)
        # # get_golden_check(58, "resnet_model/conv2d_18/Conv2D", 200)
        # # get_golden_check(59, "resnet_model/batch_normalization_16/FusedBatchNormV2", 200)
        # get_golden_check(60, "resnet_model/Relu_16", 200)
        # # get_golden_check(61, "resnet_model/conv2d_19/Conv2D", 200)
        # # get_golden_check(62, "resnet_model/batch_normalization_17/FusedBatchNormV2", 200)
        # get_golden_check(63, "resnet_model/Relu_17", 200)
        # # get_golden_check(64, "resnet_model/conv2d_20/Conv2D", 200)

        # get_golden_check(65, "resnet_model/add_5", 200)
        # # get_golden_check(66, "resnet_model/batch_normalization_18/FusedBatchNormV2", 200)
        # get_golden_check(67, "resnet_model/Relu_18", 200)
        # # get_golden_check(68, "resnet_model/conv2d_21/Conv2D", 200)
        # # get_golden_check(69, "resnet_model/batch_normalization_19/FusedBatchNormV2", 200)
        # get_golden_check(70, "resnet_model/Relu_19", 200)
        # # get_golden_check(71, "resnet_model/conv2d_22/Conv2D", 200)
        # # get_golden_check(72, "resnet_model/batch_normalization_20/FusedBatchNormV2", 200)
        # get_golden_check(73, "resnet_model/Relu_20", 200)
        # # get_golden_check(74, "resnet_model/conv2d_23/Conv2D", 200)

        # get_golden_check(75, "resnet_model/add_6", 200)
        # # get_golden_check(76, "resnet_model/batch_normalization_21/FusedBatchNormV2", 200)
        # get_golden_check(77, "resnet_model/Relu_21", 200)

        # stage 3
        # get_golden_check(78, "resnet_model/conv2d_24/Conv2D", 200)
        # # get_golden_check(79, "resnet_model/conv2d_25/Conv2D", 200)
        # # get_golden_check(80, "resnet_model/batch_normalization_22/FusedBatchNormV2", 200)
        # get_golden_check(81, "resnet_model/Relu_22", 200)
        # # get_golden_check(82, "resnet_model/conv2d_26/Conv2D", 200)
        # # get_golden_check(83, "resnet_model/batch_normalization_23/FusedBatchNormV2", 200)
        # get_golden_check(84, "resnet_model/Relu_23", 200)
        # # get_golden_check(85, "resnet_model/conv2d_27/Conv2D", 200)

        # # get_golden_check(86, "resnet_model/add_7", 200)
        # # get_golden_check(87, "resnet_model/batch_normalization_24/FusedBatchNormV2", 200)
        # get_golden_check(88, "resnet_model/Relu_24", 200)
        # # get_golden_check(89, "resnet_model/conv2d_28/Conv2D", 200)
        # # get_golden_check(90, "resnet_model/batch_normalization_25/FusedBatchNormV2", 200)
        # get_golden_check(91, "resnet_model/Relu_25", 200)
        # # get_golden_check(91, "resnet_model/conv2d_29/Conv2D", 200)
        # # get_golden_check(93, "resnet_model/batch_normalization_26/FusedBatchNormV2", 200)
        # get_golden_check(94, "resnet_model/Relu_26", 200)
        # # get_golden_check(95, "resnet_model/conv2d_30/Conv2D", 200)

        # # get_golden_check(96, "resnet_model/add_8", 200)
        # # get_golden_check(97, "resnet_model/batch_normalization_27/FusedBatchNormV2", 200)
        # get_golden_check(98, "resnet_model/Relu_27", 200)
        # # get_golden_check(99, "resnet_model/conv2d_31/Conv2D", 200)
        # # get_golden_check(100, "resnet_model/batch_normalization_28/FusedBatchNormV2", 200)
        # get_golden_check(101, "resnet_model/Relu_28", 200)
        # # get_golden_check(102, "resnet_model/conv2d_32/Conv2D", 200)
        # # get_golden_check(103, "resnet_model/batch_normalization_29/FusedBatchNormV2", 200)
        # get_golden_check(104, "resnet_model/Relu_29", 200)
        # # get_golden_check(105, "resnet_model/conv2d_33/Conv2D", 200)

        # # get_golden_check(106, "resnet_model/add_9", 200)
        # # get_golden_check(107, "resnet_model/batch_normalization_30/FusedBatchNormV2", 200)
        # get_golden_check(108, "resnet_model/Relu_30", 200)
        # # get_golden_check(109, "resnet_model/conv2d_34/Conv2D", 200)
        # # get_golden_check(110, "resnet_model/batch_normalization_31/FusedBatchNormV2", 200)
        # get_golden_check(111, "resnet_model/Relu_31", 200)
        # # get_golden_check(112, "resnet_model/conv2d_35/Conv2D", 200)
        # # get_golden_check(113, "resnet_model/batch_normalization_32/FusedBatchNormV2", 200)
        # get_golden_check(114, "resnet_model/Relu_32", 200)
        # # get_golden_check(115, "resnet_model/conv2d_36/Conv2D", 200)

        # # get_golden_check(116, "resnet_model/add_10", 200)
        # # get_golden_check(117, "resnet_model/batch_normalization_33/FusedBatchNormV2", 200)
        # get_golden_check(118, "resnet_model/Relu_33", 200)
        # # get_golden_check(119, "resnet_model/conv2d_37/Conv2D", 200)
        # # get_golden_check(120, "resnet_model/batch_normalization_35/FusedBatchNormV2", 200)
        # get_golden_check(121, "resnet_model/Relu_34", 200)
        # # get_golden_check(122, "resnet_model/conv2d_38/Conv2D", 200)
        # # get_golden_check(123, "resnet_model/batch_normalization_35/FusedBatchNormV2", 200)
        # get_golden_check(124, "resnet_model/Relu_35", 200)
        # # get_golden_check(125, "resnet_model/conv2d_39/Conv2D", 200)

        # get_golden_check(126, "resnet_model/add_11", 200)
        # get_golden_check(127, "resnet_model/batch_normalization_36/FusedBatchNormV2", 200)
        # get_golden_check(128, "resnet_model/Relu_36", 200)
        # # get_golden_check(129, "resnet_model/conv2d_40/Conv2D", 200)
        # # get_golden_check(130, "resnet_model/batch_normalization_37/FusedBatchNormV2", 200)
        # get_golden_check(131, "resnet_model/Relu_37", 200)
        # # get_golden_check(132, "resnet_model/conv2d_41/Conv2D", 200)
        # # get_golden_check(133, "resnet_model/batch_normalization_38/FusedBatchNormV2", 200)
        # get_golden_check(134, "resnet_model/Relu_38", 200)
        # # get_golden_check(135, "resnet_model/conv2d_42/Conv2D", 200)

        # # get_golden_check(136, "resnet_model/add_12", 200)
        # # get_golden_check(137, "resnet_model/batch_normalization_39/FusedBatchNormV2", 200)
        # get_golden_check(138, "resnet_model/Relu_39", 200)
        
        # # stage 4
        # get_golden_check(139, "resnet_model/Relu_42", 200)
        # get_golden_check(140, "resnet_model/Relu_45", 200)
        # get_golden_check(141, "resnet_model/Relu_48", 200)

        
        # get_golden_check(142, "resnet_model/Mean", 200)
        # get_golden_check(143, "resnet_model/dense/MatMul", 200)
        # get_golden_check(144, "resnet_model/dense/BiasAdd", 200)
        # get_golden_check(145, "softmax_tensor_fp16", 200)

