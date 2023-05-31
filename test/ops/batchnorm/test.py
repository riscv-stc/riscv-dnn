#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from work import do_test


title = "Diffent Optimization levels for add operator"
opt_levels = {"loop=1":"-O2", "loop=2":"-O2 -DNLOOPS=2"}

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def batchnorm(num, hin, win, c):
    vs1 = np.random.random((hin, win, c)).astype('float32') * 2 -1
    mean = np.random.random(c).astype('float32') * 2 - 1
    var = np.random.random(c).astype('float32')
    gam = np.random.random(c).astype('float32') * 2 - 1 
    beta = np.random.random(c).astype('float32') * 2 - 1
    eps = np.random.random(1).astype('float32')
    eps_d = eps[0]
    vd = np.multiply(gam, vs1-mean) / np.sqrt(var + eps_d) + beta
    vs1.astype('float16').tofile(f"build/{num}/src.bin")
    mean.astype('float16').tofile(f"build/{num}/mean.bin")
    var.astype('float16').tofile(f"build/{num}/var.bin")
    gam.astype('float16').tofile(f"build/{num}/gam.bin")
    beta.astype('float16').tofile(f"build/{num}/beta.bin") 
    eps.astype('float16').tofile(f"build/{num}/eps.bin")
    vd.astype('float16').tofile(f'build/{num}/golden.bin')

    return vd.astype('float16')

def batchnorm2(num, hin, win, c):
    vs1 = np.random.random((hin, win, c)).astype('float16') * 2 -1
    gam = np.random.random(c).astype('float16') * 2 - 1 
    beta = np.random.random(c).astype('float16') * 2 - 1
    vs1 = vs1.astype('float16')
    gam = gam.astype('float16')
    beta = beta.astype('float16')

    vd = np.multiply(gam, vs1)  + beta

    vs1.tofile(f"build/{num}/src.bin")
    gam.tofile(f"build/{num}/gam.bin")
    beta.tofile(f"build/{num}/beta.bin")
    vd.tofile(f'build/{num}/golden.bin')

    return vd

def test(num, params, defs):
    h, w, c = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = batchnorm2(num, h, w, c)

    os.system(f"make DEFS='-DH={h} -DW={w} -DC={c} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', f'np.allclose( result, golden, rtol={1e-3*h*w*c}, atol={1e-8*h*w*c}, equal_nan=True)' )
    print(f"> {h}x{w}x{c}, check result: {check_result}")
    

if __name__ == "__main__":
    #############  h w c
    params = (
            (1, 1, 8),
            (1, 4, 8),
            (1, 8, 8),
            (1, 32, 8),
            )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')

