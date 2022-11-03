#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf

from parallelize import parallelize

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from work import do_test

title = "Diffent Optimization levels for conv operator"

# opt_levels = {"rvv_fp16acc":"-O2 -DFP16_ACC16", "rvv":"-O2"}
opt_levels = {"rvv":"-O2", "rvm":"-O2 -D__RVM__" }

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def conv(num, hin, win, cin, cout, kh, kw, sh=1, sw=1, dh=1, dw=1, pt=0, pb=0, pl=0, pr=0):
    shape_input = [1, hin, win, cin]
    shape_weight = [kh, kw, cin, cout]
    vs1 = np.random.random(shape_input).astype('float16') * 2 - 1
    vs2 = np.random.random(shape_weight).astype('float16') * 2 - 1
    tf_pad = [[0, 0], [pt, pb], [pl, pr], [0, 0]]
    vd = tf.nn.conv2d(vs1, vs2, [1, sh, sw, 1], tf_pad, data_format='NHWC', dilations=[1, dh, dw, 1])
    vd = vd.numpy()

    vs1.tofile(f"build/{num}/src.bin")
    vs2.tofile(f"build/{num}/weight.bin")
    vd.tofile(f'build/{num}/golden.bin')

    return vd


def test(num, params, defs):
    h, w, cin, cout, kh, kw, *extras = params

    extras1 = None
    if extras:
        stride_h, stride_w, *extras1 = extras
    else:
        stride_h, stride_w = 1, 1
        dh, dw = 1, 1
        pt, pb, pl, pr = 0, 0, 0, 0

    extras2 = None
    if extras1:
        dh, dw, *extras2 = extras1
    else:
        dh, dw = 1, 1
        pt, pb, pl, pr = 0, 0, 0, 0

    if extras2:
        pt, pb, pl, pr = extras2
    else:
        pt, pb, pl, pr = 0, 0, 0, 0

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = conv(num, h, w, cin, cout, kh, kw, stride_h, stride_w, dh, dw, pt, pb, pl, pr)
    defines = (
        f'-DHIN={h} -DWIN={w} -DCIN={cin} -DCOUT={cout} -DKH={kh} -DKW={kw} '
              f'-DSTRIDE_H={stride_h} -DSTRIDE_W={stride_w} '
              f'-DDILATION_H={dh} -DDILATION_W={dw} '
              f'-DPAD_TOP={pt} -DPAD_BOTTOM={pb} -DPAD_LEFT={pl} -DPAD_RIGHT={pr}'
    )

    os.system(f"make DEFS='{defines} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', f'np.allclose( result, golden, rtol={1e-5*kh*kw*cin}, atol={1e-8*kh*kw*cin}, equal_nan=True)' )
    print(f"> {num}, check result: {check_result}")
    

if __name__ == "__main__":
    # params:
    #
    #   hin, win, cin, cout, kh, kw, 
    #                              sh=1, sw=1
    #                                      dh=1, dw=1
    #                                               pt=0, pb=0, pl=0, pr=0
    params = (
        (8, 8, 8, 8, 3, 3,   1, 1,  1, 1,   0, 0, 0, 0),
        (8, 8, 8, 8, 3, 3,   1, 1,  1, 1,   1, 1, 1, 1),
        (8, 8, 8, 8, 3, 3,   2, 2,  1, 1,   0, 0, 0, 0),
        (8, 8, 8, 8, 3, 3,   1, 1,  2, 2,   0, 0, 0, 0),
        (16, 16, 16, 16, 3, 3,   1, 1,  1, 1,   0, 0, 0, 0),
        (16, 16, 16, 16, 3, 3,   1, 1,  1, 1,   1, 1, 1, 1),
        (16, 16, 16, 16, 3, 3,   2, 2,  1, 1,   0, 0, 0, 0),
        (16, 16, 16, 16, 3, 3,   1, 1,  2, 2,   0, 0, 0, 0),
    )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')
