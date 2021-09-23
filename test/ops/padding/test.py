#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from perf import gem5_get_perf_data, vcs_get_perf_data, generate_perf_report

title = "Diffent Optimization levels for conv operator"
opt_levels = {"O0":"-O0", "O2":"-O2", "O2-unroll-loops":"-O2 -funroll-loops"}

cols = ['Workload', 'Cycles', 'IPC', 'Front', 'BS', 'MEM', 'CORE', 'Retire']

simulator = 'spike'
GEM5 = "/home/kening.zhang/stc-exp/simulator/gem5"
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def padding(hin, win, cin, pt=0, pb=0, pl=0, pr=0):
    shape_input = [hin, win, cin]
    vs1 = np.random.random(shape_input).astype('float16') * 2 - 1
    vd = np.pad(vs1, ((pt, pb), (pl, pr), (0, 0)))

    vs1.tofile("src.bin")
    vd.tofile('golden.bin')

    return vd


def test(num, params, defs):
    h, w, cin, pt, pb, pl , pr, *extras = params

    if extras:
        pt, pb, pl, pr = extras
    else:
        pt, pb, pl, pr = 0, 0, 0, 0

    os.system(f"make clean")

    golden = padding(h, w, cin, pt, pb, pl , pr)
    defines = (
        f'-DHIN={h} -DWIN={w} -DCIN={cin} '
              f'-DPAD_TOP={pt} -DPAD_BOTTOM={pb} -DPAD_LEFT={pl} -DPAD_RIGHT={pr}'
    )

    os.system(f"make DEFS='{defines} {defs}' run SIM={simulator}")

    result = from_txt( f'{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=1e-3, atol=0, equal_nan=True)' )
    print(f"> {num}, check result: {check_result}")
    

if __name__ == "__main__":
    params = (
        (5, 5, 3, 0, 0, 0, 0),
        (7, 7, 3, 1, 1, 1, 1),
        (7, 7, 3, 3, 3, 3, 3),
        (9, 9, 2, 5, 4, 6, 7),
        ( 5, 5, 130, 3, 3, 3, 3),
    )
    
    # perf optimization levels
    for key,val in opt_levels.items():
        defs = val
        if simulator == 'vcs': # TODO: support gem5
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
