#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from perf import gem5_get_perf_data, vcs_get_perf_data, generate_perf_report

title = "Diffent Optimization levels for add operator"
opt_levels = {"O0":"-O0", "O2":"-O2", "O2-unroll-loops":"-O2 -funroll-loops"}

cols = ['Workload', 'Cycles', 'IPC', 'Front', 'BS', 'MEM', 'CORE', 'Retire']

simulator = 'spike'
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)


def batchnorm(hin, win, c):
    vs1 = np.random.random((hin, win, c)).astype('float32') * 2 -1
    mean = np.random.random(c).astype('float32') * 2 - 1
    var = np.random.random(c).astype('float32')
    gam = np.random.random(c).astype('float32') * 2 - 1 
    beta = np.random.random(c).astype('float32') * 2 - 1
    eps = np.random.random(1).astype('float32')
    eps_d = eps[0]
    vd = np.multiply(gam, vs1-mean) / np.sqrt(var + eps_d) + beta
    vs1.astype('float16').tofile("src.bin")
    mean.astype('float16').tofile("mean.bin")
    var.astype('float16').tofile("var.bin")
    gam.astype('float16').tofile("gam.bin")
    beta.astype('float16').tofile("beta.bin") 
    eps.astype('float16').tofile("eps.bin")
    vd.astype('float16').tofile('golden.bin')

    return vd.astype('float16')

def batchnorm2(hin, win, c):
    vs1 = np.random.random((hin, win, c)).astype('float16') * 2 -1
    gam = np.random.random(c).astype('float16') * 2 - 1 
    beta = np.random.random(c).astype('float16') * 2 - 1
    vs1 = vs1.astype('float16')
    gam = gam.astype('float16')
    beta = beta.astype('float16')

    vd = np.multiply(gam, vs1)  + beta

    vs1.tofile("src.bin")
    gam.tofile("gam.bin")
    beta.tofile("beta.bin")
    vd.tofile('golden.bin')

    return vd

def test(num, params, defs):
    h, w, c = params

    os.system(f"make clean")

    golden = batchnorm2(h, w, c)

    os.system(f"make DEFS='-DH={h} -DW={w} -DC={c} {defs}' run SIM={simulator}")

    result = from_txt( f'{simulator}.sig', golden, 0 )
    os.makedirs('check', exist_ok=True)
    check_result = check_to_txt( golden, result, f'check/{num}.data', f'np.allclose( result, golden, rtol={1e-3*h*w*c}, atol={1e-8*h*w*c}, equal_nan=True)' )
    print(f"> {h}x{w}x{c}, check result: {check_result}")
    

if __name__ == "__main__":
    #############  h w c
    params = (
            (2, 3, 4),
            (16, 64, 32),
            (16, 16, 130),
            (56, 56, 8)
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

