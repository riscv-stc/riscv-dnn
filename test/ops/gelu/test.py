#!/usr/bin/python3
import os
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
    opt_levels = {"gelu": "-O2"}

def gelu(num, hin, win):
    vs1 = np.random.random((hin, win)).astype('float16') * 2 - 1
    vd = tf.nn.gelu(vs1)
    vd = vd.numpy().astype('float16')

    vs1.tofile(f'build/{num}/src.bin')
    vd.tofile(f'build/{num}/golden.bin')

    return vd



def test(num, params, defs):
    h, w = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = gelu(num, h, w)

    os.system(f"make DEFS='-DH={h} -DW={w} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=1e-3, atol=0, equal_nan=True)' )
    print(f"> {h}x{w}, check result: {check_result}")
    

if __name__ == "__main__":
    ############# h, w, c, base
    params = (
            (  1, 8),
            (  4, 8),
            (  8, 8),
            ( 32, 8),
            (128, 8),
            (512, 8),
            )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')
