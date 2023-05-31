import tensorflow as tf
import numpy as np
import re

outdir = "bindata/"

with tf.compat.v1.gfile.FastGFile("resnet-50_v2.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.compat.v1.import_graph_def(graph_def, name="")
    nodes = [n for n in tf.compat.v1.get_default_graph().as_graph_def().node]
    
    with tf.compat.v1.Session() as sess:
        tensor_name_list = []
        tensor_list = []
        for node in nodes:
            if "read" in node.name:
                tensor_name_list.append(node.name)
                tensor = sess.graph.get_tensor_by_name(node.name + ":0")
                tensor_list.append(tensor)
        # print(tensor_list)

        preds = sess.run(tensor_list)
        for i, tensor_name in enumerate(tensor_name_list):
            param_name = tensor_name.replace("/", "_")
            param_name = param_name[13:-5]
            file_name = param_name + '.bin'
            if "kernel" in file_name:
                preds[i].astype(np.float16).tofile(outdir + file_name)
            else:
                preds[i].tofile(outdir + file_name)
            # print(i, file_name)
