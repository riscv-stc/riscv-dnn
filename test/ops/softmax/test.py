#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from perf import gem5_get_perf_data, vcs_get_perf_data, generate_perf_report

title = "Diffent Optimization levels for softmax operator"
opt_levels = {"O0":"-O0", "O2":"-O2", "O2-unroll-loops":"-O2 -funroll-loops"}

cols = ['Workload', 'Cycles', 'IPC', 'Front', 'BS', 'MEM', 'CORE', 'Retire']

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def softmax(hin, win):
    # np.random.seed( 100 )
    vs1 = np.random.random((hin, win)).astype('float32') * 10 - 5 # random  number between -5 and 5
    vd = tf.nn.softmax(vs1.flatten())
    vd = vd.numpy().astype('float32')
    
    vs1.tofile('src.bin')
    vd.tofile('golden.bin')

    return vd


def test(num, params, defs):
    h, w = params

    os.system(f"make clean")

    golden = softmax(h, w)
    os.system(f"make DEFS='-DH={h} -DW={w} {defs}' run SIM={simulator}")
    #os.system(f"make DEFS='-DH={h} -DW={w} {defs}' dump SIM={simulator}")

    result = from_txt( f'{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=2e-3, atol=0, equal_nan=True)' )
    print(f"> {h}x{w}, check result: {check_result}")
    

if __name__ == "__main__":
    #############  h w cin
    params = (
            ( 1, 1001 ),
            # (1, 1, 10),
            # (1, 1, 100),
            # (1000, 1, 1),
            # (16, 16, 16),
            # (32, 32, 32),
            # (64, 64, 64),
            # (65, 300, 66),
            )
    
    # perf optimization levels
    for key,val in opt_levels.items():
        defs = val
        if simulator == 'vcs' or simulator == 'gem5':
            defs += ' -DPERF '

        output = pd.DataFrame(columns = cols)
        for i in range(len(params)):
            test(key+'-'+str(i), params[i], defs)
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




