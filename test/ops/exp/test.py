#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

from parallelize import parallelize

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from work import do_test

title = "Diffent Optimization levels for exp operator"
opt_levels = {"O2":"-O2", "O2-unroll-loops":"-O2 -funroll-loops"}

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)

def exp(num, hin, win, cin):
    vs1 = np.random.random((hin, win, cin)).astype('float16') * 20 - 10
    vd = np.exp(vs1)
    vd = vd.astype('float16')
    
    vs1.tofile(f'build/{num}/src.bin')
    vd.tofile(f'build/{num}/golden.bin')

    return vd


def test(num, params, defs):
    h, w, cin = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = exp(num, h, w, cin)

    os.system(f"make DEFS='-DH={h} -DW={w} -DCIN={cin} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=2e-3, atol=0, equal_nan=True)' )
    print(f"> {h}x{w}x{cin}, check result: {check_result}")
    

if __name__ == "__main__":
    #############  h w cin
    params = (
            (2, 2, 1),
            (5, 5, 5),
            (5, 5, 65),
            (5, 5, 130),
            (7, 7, 3),
            (1, 1, 64),
            )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')

