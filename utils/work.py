import os
import numpy as np
import pandas as pd

import multiprocessing as mp

from check import from_txt, get_sig_addr
from perf import get_perf_data, generate_perf_report
from tma import PlotMetrics

cols = ['Workload', 'Cycles', 'IPC', 'Front', 'BS', 'MEM', 'CORE', 'Retire']

def run(key, i, defs, params, test_func, simulator, enable_perf):
    test_func(key+'-'+str(i), params[i], defs)
    df = pd.DataFrame(columns = cols)
    if enable_perf:
        sig_begin_addrr = get_sig_addr(f'build/{key}-{i}/test.map', 'begin_signature')
        perf_begin_addr = get_sig_addr(f'build/{key}-{i}/test.map', 'begin_perf_data')
        perf_end_addr = get_sig_addr(f'build/{key}-{i}/test.map', 'end_perf_data')
        ndperf = from_txt(f'build/{key}-{i}/{simulator}.sig', np.zeros((perf_end_addr - perf_begin_addr,), dtype=np.uint8), perf_begin_addr - sig_begin_addrr)
        ndperf.tofile(f'build/{key}-{i}/{simulator}.perf_data')
        perf_data = get_perf_data(f'build/{key}-{i}/{simulator}.perf_data', simulator)
        perf_data["Workload"] = 'x'.join(map(str, params[i]))
        perf_data = [perf_data[col] for col in cols]
        df.loc[0] = perf_data
        df.to_pickle(f'build/{key}-{i}/perf.pkl')

        PlotMetrics(f'build/{key}-{i}/{simulator}.perf_data', f'build/{key}-{i}/tma.png', 'x'.join(map(str, params[i])))

def do_test(params, opt_levels, test_func, title, simulator, enable_perf = False):
    args = []
    for key,val in opt_levels.items():
        defs = val
        if enable_perf:
            defs += ' -DPERF '

        for i in range(len(params)):
            args.append((key, i, defs, params, test_func, simulator, enable_perf))

    nproc = mp.cpu_count()
    if len(args) < nproc:
        nproc = len(args)
    with mp.Pool(nproc) as p:
        p.starmap(run, args)

    for key,val in opt_levels.items():
        output = pd.DataFrame(columns = cols)
        for i in range(len(params)):
            if enable_perf:
                df = pd.read_pickle(f'build/{key}-{i}/perf.pkl')
                output.loc[i] = df.loc[0]
        if enable_perf:
            output = output.set_index('Workload')
            os.makedirs('perf', exist_ok=True)
            output.to_csv(f'perf/{key}.csv')

    if enable_perf:
        generate_perf_report(title, [x for x in opt_levels.keys()])
        print('> Perf report generated.')

