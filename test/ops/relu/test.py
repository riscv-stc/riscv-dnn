#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from perf import gem5_get_perf_data, vcs_get_perf_data, generate_perf_report

title = "Diffent Optimization levels for add operator"
opt_levels = {"O0":"-O0", "O2":"-O2", "O2-loops":"-O2 -funroll-loops"}

cols = ['Workload', 'Cycles', 'IPC', 'Front', 'BS', 'MEM', 'CORE', 'Retire']

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def relu(hin, win, cin, base):
    vs1 = np.random.random((hin, win, cin)).astype('float16') * 2 - 1
    base = np.array(base).astype('float16')
    vd = np.where(vs1 > base, vs1, base)
    vd = vd.astype('float16')

    vs1.tofile('src.bin')
    base.tofile('base.bin')
    vd.tofile('golden.bin')

    return vd



def test(num, params, defs):
    h, w, c, base = params

    os.system(f"make clean")

    golden = relu(h, w, c, base)

    os.system(f"make DEFS='-DH={h} -DW={w} -DC={c} {defs}' run SIM={simulator}")

    result = from_txt( f'{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', 'np.allclose( result, golden, rtol=1e-3, atol=0, equal_nan=True)' )
    print(f"> {h}x{w}x{c}x{base}, check result: {check_result}")
    

if __name__ == "__main__":
    ############# h, w, c, base
    params = (
            (2, 2, 1, 0),
            (5, 5, 5, 0),
            (5, 5, 65, 0),
            (5, 5, 130, 0),
            (2, 2, 1, 0.1),
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

