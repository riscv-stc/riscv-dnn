#!/usr/bin/python3
import os
from random import gammavariate
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from work import do_test


title = "Diffent Optimization levels for add operator"
opt_levels = {"loop=4":"-O2 -DNLOOPS=4", "loop=8":"-O2 -DNLOOPS=8"}

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)

if simulator == "spike":
    opt_levels = {"ln": "-O2"}

def layernorm(num, hin, win):
    vs1 = np.random.random((hin, win)).astype('float16') * 200 - 100
    gamma = np.random.random(win).astype('float16') * 2 - 1
    beta = np.random.random(win).astype('float16') * 2 - 1

    layer = tf.keras.layers.LayerNormalization(axis = 1, epsilon=1e-5)
    layer.build([1, win])
    layer.beta = beta.astype("float32")
    layer.gamma = gamma.astype("float32")
    vd =layer(vs1.astype("float32"))

    vd = vd.numpy().astype("float16")


    vs1.tofile(f'build/{num}/src.bin')
    beta.tofile(f'build/{num}/beta.bin')
    gamma.tofile(f'build/{num}/gamma.bin')
    vd.tofile(f'build/{num}/golden.bin')

    return vd



def test(num, params, defs):
    h, w = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = layernorm(num, h, w)

    os.system(f"make DEFS='-DH={h} -DW={w} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=1e-3, atol=1e-2, equal_nan=True)' )
    print(f"> {h}x{w}, check result: {check_result}")
    

if __name__ == "__main__":
    ############# h, w, c, base
    params = (
            (  4, 8),
            (  8, 8),
            ( 32, 8),
            (128, 128),
            (128, 256),
            (256, 128),
            (512, 512),
            )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')
