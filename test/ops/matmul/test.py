#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from perf import gem5_get_perf_data, vcs_get_perf_data, generate_perf_report

title = "Diffent Optimization levels for matmul operator"

# opt_levels = {"rvv_fp16acc":"-O2 -DFP16_ACC16", "rvv":"-O2"}
opt_levels = {"rvv":"-O2", "rvm":"-O2 -D__RVM__" }

cols = ['Workload', 'Cycles', 'IPC', 'Front', 'BS', 'MEM', 'CORE', 'Retire']

simulator = 'spike'
GEM5 = "/home/kening.zhang/stc-exp/simulator/gem5"
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def matmul(m, k, n):
    vs1 = np.random.random((m, k)).astype('float16') * 2 - 1
    vs2 = np.random.random((k, n)).astype('float16') * 2 - 1
    vd = np.matmul(vs1, vs2, dtype=np.float16)

    vs1.tofile("src1.bin")
    vs2.tofile("src2.bin")
    vd.tofile('golden.bin')

    return vd


def test(num, params, defs, fp16acc):
    m, k, n = params

    os.system(f"make clean")

    golden = matmul(m, k, n)
    os.system(f"make DEFS='-DM={m} -DK={k} -DN={n} {defs}' run SIM={simulator}")

    result = from_txt( f'{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)

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
    # # check params
    # params = (
    #     #  m k n
    #     (1, 1, 10),
    #     (10, 10, 1),
    #     (63, 65, 63),
    #     (65, 63, 65),
    # )

    # perf params
    params = (
        #  m k n
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    )
    
    # perf optimization levels
    for key,val in opt_levels.items():
        defs = val
        if simulator == 'vcs': # TODO: support gem5
            defs += ' -DPERF '

        output = pd.DataFrame(columns = cols)
        for i in range(len(params)):
            test(key+'-'+str(i), params[i], defs, key == 'rvv_fp16acc')
            if simulator == 'gem5':
                perf_data = gem5_get_perf_data('m5out')
                perf_data["Workload"] = 'x'.join(map(str, params[i]))
                perf_data = [perf_data[col] for col in cols]
                output.loc[i] = perf_data
            elif simulator == 'vcs':
                perf_data = vcs_get_perf_data()
                perf_data["Workload"] = 'x'.join(map(str, params[i]))
                perf_data = [perf_data[col] for col in cols]
                output.loc[i] = perf_data
        if simulator != 'spike':
            output = output.set_index('Workload')
            os.makedirs('perf', exist_ok=True)
            output.to_csv(f'perf/{key}.csv')

    if simulator != 'spike':
        generate_perf_report(title, [x for x in opt_levels.keys()])
        print('> Perf report generated.')
