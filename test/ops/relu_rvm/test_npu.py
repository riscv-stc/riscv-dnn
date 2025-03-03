#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../../../utils") 
from check import from_npu_dump, check_to_txt
from work import do_test


title = "Diffent Optimization levels for add operator"
opt_levels = {"loop=1":"-O2"}

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def relu(num, hin, win, cin, base):
    vs1 = np.random.random((hin, win, cin)).astype('float16') * 2 - 1
    base = np.array(base).astype('float16')
    vd = np.where(vs1 > base, vs1, base)
    vd = vd.astype('float16')

    vs1.tofile(f'build/{num}/src.bin')
    base.tofile(f'build/{num}/base.bin')
    vd.tofile(f'build/{num}/golden.bin')

    return vd



def test(num, params, defs):
    h, w, c, base = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = relu(num, h, w, c, base)

    os.system(f"make ll NUM={num} > build/{num}/test.log 2>&1")
    os.system(f"cp -rf /home/chenbingzheng/WORKSPACE/OPEN/MC2LL/llvm-project/run-matrix2npu build/{num}")
    os.system(f"./build/{num}/run-matrix2npu/run_npu.sh build/{num}/test.ll >> build/{num}/test.log 2>&1")
 
    result = from_npu_dump( f'build/{num}/run-matrix2npu/workspace/npu.dump', golden, 0)

    
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=1e-3, atol=0, equal_nan=True)' )
    print(f"> {h}x{w}x{c}x{base}, check result: {check_result}")
    

if __name__ == "__main__":
    ############# h, w, c, base
    params = (
            # (1, 1, 8, 0),
            # (1, 4, 8, 0),
            # (1, 8, 8, 0),
            # (1, 32, 8, 0),
            # (1, 128, 8, 0),
            (1, 512, 8, 0),
            )
    
    do_test(params, opt_levels, test, title, simulator, simulator!='spike')

    os.system("stty echo")
