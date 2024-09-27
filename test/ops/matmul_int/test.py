#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd
import math

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from work import do_test


title = "Diffent Optimization levels for matmul operator"

opt_levels = {"loop=1":"-O2 -D__RVM__"}

datatype = "int32"

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)

def safe_mul(x, y):
    result = np.matmul(x.astype(np.int32),y.astype(np.int32),dtype=np.int32)
    
    m = x.shape[0]
    n = x.shape[1]
    z = y.shape[1]
    result2 = np.zeros((m,z),dtype=np.int32)
    for i in range(0,m):
        for k in range(0,z):
            for j in range(0,n):
                result2[i][k] += x[i][j].astype(np.int32)*y[j][k].astype(np.int32);
                if datatype == "int8":
                    sMax = 127
                    sMin = -128
                elif datatype == "int16":
                    sMax = 32767
                    sMin = -32768
                else:
                    sMax = 2147483647
                    sMin = -2147483648
                result2[result2 > sMax] = sMax
                result2[result2 < sMin] = sMin 
    if datatype == "int8":
        return result2.astype(np.int8)
    elif datatype == "int16":
        return result2.astype(np.int16)
    else:
        return result2.astype(np.int32)
    

def matmul(num, m, k, n):

    vs1 = ((np.random.random((m,k)) * 256) - 128).astype('int8')
    vs2 = ((np.random.random((k,n)) * 256)- 128).astype('int8')
    vd =  safe_mul(vs1,vs2)

    vs1.tofile(f"build/{num}/src1.bin")
    vs2.tofile(f"build/{num}/src2.bin")
    vd.tofile(f'build/{num}/golden.bin')

    return vd


def test(num, params, defs):
    m, k, n = params

    os.system(f"rm -rf build/{num} && mkdir -p build/{num}")

    golden = matmul(num, m, k, n)
    if datatype == "int8":
        out_size = hex(math.ceil((m * n )/8)*8)
    elif datatype == "int16":
        out_size = hex(math.ceil((m * n * 2)/8)*8)
    else:
        out_size = hex(math.ceil((m * n * 4)/8)*8)
    os.system(f"make DEFS='-DM={m} -DK={k} -DN={n} {defs}' OUT_SIZE={out_size} run SIM={simulator} NUM={num} >build/{num}/test.log 2>&1")

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
        # (4, 4, 1),
        # (1, 8, 8),
        # (8, 8, 8),
        # (16, 8, 8),
        # (16, 8, 16),
        (16, 8, 32),
        (32, 8, 32),
        (32, 16, 32),
        (64, 8, 64),
        (128,32, 64),
    )

    do_test(params, opt_levels, test, title, simulator, simulator!='spike')

