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
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def avgpool(hin, win, cin, kh, kw, sh=1, sw=1, pt=0, pb=0, pl=0, pr=0):
    shape_input = [1, hin, win, cin]
    vs1 = np.random.random(shape_input).astype('float16') * 2 - 1
    if pt == 0:
        padding = "VALID"
    else: 
        padding = "SAME"
    vd = tf.nn.avg_pool(vs1, [1, kh, kw, 1], [1, sh, sw, 1], padding)
    vd = vd.numpy()

    vs1.tofile("src.bin")
    vd.tofile('golden.bin')

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
        dh, dw, *extras2 = extras1
    else:
        pt, pb, pl, pr = 0, 0, 0, 0

    if extras2:
        pt, pb, pl, pr = extras2
    else:
        pt, pb, pl, pr = 0, 0, 0, 0

    os.system(f"make clean")

    golden = avgpool(h, w, cin, kh, kw, stride_h, stride_w, pt, pb, pl, pr)
    defines = (
        f'-DHIN={h} -DWIN={w} -DCIN={cin} -DKH={kh} -DKW={kw} '
              f'-DSTRIDE_H={stride_h} -DSTRIDE_W={stride_w} '
              f'-DPAD_TOP={pt} -DPAD_BOTTOM={pb} -DPAD_LEFT={pl} -DPAD_RIGHT={pr}'
    )

    os.system(f"make DEFS='{defines} {defs}' run SIM={simulator}")

    result = from_txt( f'{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=2e-3, atol=0, equal_nan=True)' )
    print(f"> {num}, check result: {check_result}")
    

if __name__ == "__main__":
    # params:
    #
    #   hin, win, cin, kh, kw, 
    #                        sh=1, sw=1
    #                                   pt=0, pb=0, pl=0, pr=0
    params = (
    #   (5, 5, 3, 3, 3),
    #   (7, 7, 3, 3, 3, 2, 2),
    #   (9, 9, 3, 3, 3, 2, 2),
        (16, 16, 16, 5, 5),
        (64, 64, 64, 5, 5),
        (66, 66, 130, 7, 7),
    #   (64, 64, 64, 7, 7),
    #   (16, 16, 130, 5, 5),
    #   (9, 9, 2, 3, 3,  1, 1,  1, 1, 1, 1),
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
