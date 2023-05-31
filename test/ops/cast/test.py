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


def cast_f32_to_f16(num, hin, win, cin, cout):
    vs1 = np.random.random((hin, win, cin, cout)).astype('float32') * 2 - 1
    vd = vs1.copy().astype('float16')

    vs1.tofile(f'build/{num}/src.bin')
    vd.tofile(f'build/{num}/golden.bin')

    return vd


def test(num, params, defs):
    h, w, cin, cout = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = cast_f32_to_f16(num, h, w, cin, cout)

    os.system(f"make DEFS='-DH={h} -DW={w} -DCIN={cin} -DCOUT={cout} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=1e-3, atol=0, equal_nan=True)' )
    print(f"> {h}x{w}x{cin}x{cout}, check result: {check_result}")
    

if __name__ == "__main__":
    ############# h, w, cin, cout
    params = (
            (1, 1, 1, 8),
            (1, 1, 4, 8),
            (1, 1, 8, 8),
            (1, 1, 32, 8),
            )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')

