#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from work import do_test


title = "Diffent Optimization levels for add operator"
opt_levels = {"loop=100":"-O2 -DNLOOPS=100"}

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def add(num, hin, win, cin, cout):
    vs1 = np.random.random((hin, win, cin, cout)).astype('float16') * 2 - 1
    vs2 = np.random.random((hin, win, cin, cout)).astype('float16') * 2 - 1

    vd = np.add(vs1, vs2)
    vd.astype('float16')
    
    vs1.tofile(f'build/{num}/src1.bin')
    vs2.tofile(f'build/{num}/src2.bin')
    vd.tofile(f'build/{num}/golden.bin')

    return vd



def test(num, params, defs):
    h, w, cin, cout = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = add(num, h, w, cin, cout)

    os.system(f"make DEFS='-DH={h} -DW={w} -DCIN={cin} -DCOUT={cout} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=1e-3, atol=0, equal_nan=True)' )
    print(f"> {h}x{w}x{cin}x{cout}, check result: {check_result}")
    

if __name__ == "__main__":
    #############  h w cin cout
    params = (
            (8,  8,  8, 1),
            )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')


