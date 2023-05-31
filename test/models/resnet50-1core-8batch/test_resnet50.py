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
opt_levels = {"loop1": "-O2 -D__RVM__ -DNLOOPS=1", "loop2": "-O2 -D__RVM__ -DNLOOPS=2", "loop4": "-O2 -D__RVM__ -DNLOOPS=4"}
# opt_levels = {"loop1": "-O2 -D__RVM__ -DNLOOPS=1"}

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
            
            
def test(num, params, defs):
    BATCH, *extras = params
    extras = None

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")
    os.system(f"make clean && make DEFS='{defs} -DBATCH={BATCH}' run SIM={simulator} NUM={num}  > build/{num}/test.log 2>&1")

    # get_golden_check(num, "resnet_model/Relu", BATCH)
    # get_golden_check(num,  "resnet_model/conv2d_1/Conv2D", BATCH)
    # get_golden_check(num,  "resnet_model/Relu_1", BATCH)
    # get_golden_check(num,  "resnet_model/Relu_2", BATCH)
    # get_golden_check(num,  "resnet_model/Relu_3", BATCH)
    # get_golden_check(num, "resnet_model/Relu_9", BATCH)
    # get_golden_check(num, "resnet_model/Relu_18", BATCH)
    # get_golden_check(num, "resnet_model/Relu_19", BATCH)
    # get_golden_check(num, "resnet_model/Relu_27", BATCH)
    # get_golden_check(num, "resnet_model/Relu_30", BATCH)
    # get_golden_check(num, "resnet_model/Relu_39", BATCH)
    # get_golden_check(num, "resnet_model/Relu_42", BATCH)
    # get_golden_check(num, "resnet_model/Relu_45", BATCH)
    # # get_golden_check(num, "resnet_model/Relu_44", BATCH)
    # get_golden_check(num, "resnet_model/Relu_48", BATCH)
    # get_golden_check(num, "resnet_model/Mean", BATCH)
    # get_golden_check(num, "resnet_model/dense/MatMul", BATCH)
    # get_golden_check(num, "resnet_model/dense/BiasAdd", BATCH)
    # get_golden_check(num, "softmax_tensor_fp16", BATCH)

if __name__ == "__main__" :
    ## run at first time
    os.system("rm *.o")
    # os.system("python3 gen_input_bin-fp16.py")
    # os.system("python3 gen_single_weight_bin.py")
    # os.system("python3 gen_padding_input.py")
    
    # params:
    #   BATCH
    params = (
        (8,),
    )
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')
