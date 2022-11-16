#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from work import do_test


title = "Diffent Optimization levels for matmul operator"

# opt_levels = {"rvv_fp16acc":"-O2 -DFP16_ACC16", "rvv":"-O2"}
# opt_levels = {"rvv":"-O2", "rvm":"-O2 -D__RVM__" }
opt_levels = {"loop=1":"-O2 -D__RVM__", "loop=2":"-O2 -D__RVM__ -DNLOOPS=2"}

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def matmul(num, m, k, n):
    vs1 = np.random.random((m, k)).astype('float16') * 2 - 1
    vs2 = np.random.random((k, n)).astype('float16') * 2 - 1
    vd = np.matmul(vs1, vs2, dtype=np.float16)

    vs1.tofile(f"build/{num}/src1.bin")
    vs2.tofile(f"build/{num}/src2.bin")
    vd.tofile(f'build/{num}/golden.bin')

    return vd


def test(num, params, defs):
    m, k, n = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = matmul(num, m, k, n)
    os.system(f"make DEFS='-DM={m} -DK={k} -DN={n} {defs}' run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

    result = from_txt( f'build/{num}/{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)

    fp16acc = '-DFP16_ACC16' in defs

    # fp16acc use larger tolerances
    if fp16acc:
        rk = k * 1000
        ak = k * 10000
    else:
        rk = k
        ak = k
    check_result = check_to_txt( golden, result, f'check/{num}.data', f'np.allclose( result, golden, rtol={1e-5*rk}, atol={1e-8*ak}, equal_nan=True)' )
    print(f"> {m}x{k}x{n}, check result: {check_result}")
    

if __name__ == "__main__":
    # perf params
    params = (
        #  m k n
        (1, 8, 1),
        (1, 8, 8),
        (8, 8, 8),
        (16, 8, 8),
        (16, 8, 16),
        (16, 8, 32),
        (32, 8, 32),
        (32, 16, 32),
        (64, 8, 64),
    )

    do_test(params, opt_levels, test, title, simulator, simulator!='spike')

