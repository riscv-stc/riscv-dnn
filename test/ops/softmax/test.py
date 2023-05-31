#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from work import do_test


title = "Diffent Optimization levels for softmax operator"
opt_levels = {"loop=1":"-O2", "loop=2":"-O2 -DNLOOPS=2"}

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def softmax(num, hin, win):
    # np.random.seed( 100 )
    vs1 = np.random.random((hin, win)).astype('float32') * 10 - 5 # random  number between -5 and 5
    vd = tf.nn.softmax(vs1.flatten())
    vd = vd.numpy().astype('float32')
    
    vs1.tofile(f'build/{num}/src.bin')
    vd.tofile(f'build/{num}/golden.bin')

    return vd


def test(num, params, defs):
    h, w = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = softmax(num, h, w)
    os.system(f"make DEFS='-DH={h} -DW={w} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")
    #os.system(f"make DEFS='-DH={h} -DW={w} {defs}' dump SIM={simulator} NUM={num}")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=2e-3, atol=0, equal_nan=True)' )
    print(f"> {h}x{w}, check result: {check_result}")
    

if __name__ == "__main__":
    #############  h w
    params = (
            ( 1, 8 ),
            ( 4, 8 ),
            ( 8, 8 ),
            ( 32, 8 ),
            )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')
