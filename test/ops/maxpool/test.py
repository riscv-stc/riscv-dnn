#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from work import do_test


title = "Diffent Optimization levels for conv operator"
opt_levels = {"loop=1":"-O2", "loop=2":"-O2 -DNLOOPS=2"}

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def maxpool(num, hin, win, cin, kh, kw, sh=1, sw=1, pt=0, pb=0, pl=0, pr=0):
    shape_input = [1, hin, win, cin]
    vs1 = np.random.random(shape_input).astype('float16') * 2 - 1
    if (pt + pb + pl + pr)== 0:
        padding = "VALID"
    else: 
        padding = "SAME"
    vd = tf.nn.max_pool(vs1, [1, kh, kw, 1], [1, sh, sw, 1], padding)
    vd = vd.numpy().astype('float16')
    vs1.tofile(f"build/{num}/src.bin")
    vd.tofile(f'build/{num}/golden.bin')

    return vd


def test(num, params, defs):
    h, w, cin, kh, kw, *extras = params

    extras1 = None
    if extras:
        stride_h, stride_w, *extras1 = extras
    else:
        stride_h, stride_w = 1, 1
        pt, pb, pl, pr = 0, 0, 0, 0

    extras2 = None
    if extras1:
        pt, pb, pl, pr = extras1
    else:
        pt, pb, pl, pr = 0, 0, 0, 0

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = maxpool(num, h, w, cin, kh, kw, stride_h, stride_w, pt, pb, pl, pr)
    defines = (
        f'-DHIN={h} -DWIN={w} -DCIN={cin} -DKH={kh} -DKW={kw} '
              f'-DSTRIDE_H={stride_h} -DSTRIDE_W={stride_w} '
              f'-DPAD_TOP={pt} -DPAD_BOTTOM={pb} -DPAD_LEFT={pl} -DPAD_RIGHT={pr}'
    )

    os.system(f"make DEFS='{defines} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=1e-3, atol=0, equal_nan=True)' )
    print(f"> {num}, check result: {check_result}")
    

if __name__ == "__main__":
    # params:
    #
    #   hin, win, cin, kh, kw, 
    #                        sh=1, sw=1
    #                                   pt=0, pb=0, pl=0, pr=0
    params = (
        (16,  16,  8, 3, 3, 1, 1),
        (16,  16,  8, 3, 3, 2, 2),
        (32,  32,  8, 3, 3, 1, 1),
        (32,  32,  8, 3, 3, 2, 2),
        (128, 128, 8, 3, 3, 1, 1),
        (128, 128, 8, 3, 3, 2, 2),
        (16,  16,  64, 3, 3, 1, 1),
        (16,  16,  64, 3, 3, 2, 2),
        (32,  32,  64, 3, 3, 1, 1),
        (32,  32,  64, 3, 3, 2, 2),
    )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')
