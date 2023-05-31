import tensorflow as tf
import numpy as np
import re

outdir = "bindata/"

fweight = open(outdir+"weight.bin", "wb")
woffset = [0,]
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
                tmp = preds[i].astype(np.float16)
                if preds[i].shape[-1] > 64 and preds[i].ndim == 4:
                    tmp = np.pad(tmp, ((0, 0), (0, 0), (0, 0), (0, 64)), 'constant')
                fweight.write(tmp.tobytes())
                woffset.append(woffset[-1]+tmp.size)
            else:
                preds[i].tofile(outdir + file_name)
fweight.close()

falpha = open(outdir+"alpha.bin", "wb")
aoffset = [0,]
boffset = [64,]
for i in range(49):
    fhead = outdir + "batch_normalization_"
    gamend = "gamma.bin"
    betaend = "beta.bin"
    meanend = "moving_mean.bin"
    varend = "moving_variance.bin"
    epsd = 0.000010009999641624745
    if i == 0:
        gamd = np.fromfile(fhead+gamend, dtype=np.float32).astype(np.float64)
        betad = np.fromfile(fhead+betaend, dtype=np.float32).astype(np.float64)
        meand = np.fromfile(fhead+meanend, dtype=np.float32).astype(np.float64)
        vard = np.fromfile(fhead+varend, dtype=np.float32).astype(np.float64)
        outalpha = outdir + "batch_normalization_new_alpha.bin"
        outbeta = outdir + "batch_normalization_new_beta.bin"
    else:
        gamd = np.fromfile(fhead + str(i) + "_" + gamend, dtype=np.float32).astype(np.float64)
        betad = np.fromfile(fhead + str(i) + "_" + betaend, dtype=np.float32).astype(np.float64)
        meand = np.fromfile(fhead + str(i) + "_" + meanend, dtype=np.float32).astype(np.float64)
        vard = np.fromfile(fhead + str(i) + "_" + varend, dtype=np.float32).astype(np.float64)
        outalpha = outdir + "batch_normalization_" + str(i) + "_new_alpha.bin"
        outbeta = outdir + "batch_normalization_" + str(i) + "_new_beta.bin"
    
    alphad = np.multiply(gamd, np.reciprocal(np.sqrt(vard + epsd)))
    betanew = np.subtract(betad, np.multiply(meand, alphad))
    falpha.write(alphad.astype('float16').tobytes())
    falpha.write(betanew.astype('float16').tobytes())
    if i == 0:
        aoffset[-1] = 0
        boffset[-1] = aoffset[-1] + alphad.size
    else:
        aoffset.append(aoffset[-1]+2*(boffset[-1]-aoffset[-1]))
        boffset.append(aoffset[-1]+alphad.size)


dense_bias_data = np.fromfile(outdir+"dense_bias.bin", dtype=np.float32)
falpha.write(dense_bias_data.astype('float16').tobytes())

falpha.close()


inhead = '''
#ifndef __RESNET50V2_WEIGHTDATA_NEW_
#define __RESNET50V2_WEIGHTDATA_NEW_

#include <stdint.h>
#include "../../../include/incbin.h"

'''
with open('resnet50_parameters_new.h', 'w') as fhead:
    fhead.write(inhead)
    fhead.write("int32_t woffset[] = " + str(woffset) + ";\n")
    fhead.write("int32_t aoffset[] = " + str(aoffset) + ";\n")
    fhead.write("int32_t boffset[] = " + str(boffset) + ";\n")
    fhead.write("extern uint8_t weight_data[];\n")
    fhead.write('''INCBIN(weight_data, "weight.bin",  ".scdata.params");\n''')
    fhead.write("extern uint8_t alpha_data[];\n")
    fhead.write('''INCBIN(alpha_data, "alpha.bin",  ".scdata.params");\n''')
    fhead.write("\n#endif\n")
    