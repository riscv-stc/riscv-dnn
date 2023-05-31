#!/bin/bash

#%% 
from posixpath import split
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import re
from decimal import Decimal

from inspect import getmembers
from cffi import FFI

from rich.console import Console
console = Console()

plt.rcParams["figure.dpi"] = 300
plt.rcParams['font.size'] = 8 

title = 'Diffent Optimization levels for add operator'
setups = ['O0', 'O2', 'O3']
metrics = ['Cycles', 'Front', 'BS', 'MEM', 'CORE', 'Retire']

def generate_perf_report(title, setups, metrics=metrics):
    fig, axes = plt.subplots(nrows=1, ncols=len(setups), squeeze=False)

    # read original data
    dfs = [pd.read_csv(f"perf/{s}.csv") for s in setups]
    dfs = [df.set_index('Workload') for df in dfs]


    # TMA metrics plots
    for i in range(len(dfs)):
        df = dfs[i]
        df = df[[*metrics]]
        console.log(df)
        df.plot(ax=axes[0][i], kind="barh", stacked=True, legend=False, title=setups[i])
        if i != 0:
            axes[0][i].get_yaxis().set_visible(False)

    plt.legend(bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()

    fio = io.BytesIO()
    fig.savefig(fio, format='svg')
    svg = base64.b64encode(fio.getvalue()).decode()
    svg_html = f'<img src="data:image/svg+xml;base64,{svg}"></img>'

    # TMA data table
    tdfs = [df[[*metrics]] for df in dfs]
    df = pd.concat(tdfs, axis=1)
    df.columns = pd.MultiIndex.from_product([setups, metrics])
    # display(df)
    tma_table = df.to_html(classes="table is-narrow is-bordered")

    # TMA data table
    tdfs = [df[['Cycles', 'IPC']] for df in dfs]
    df = pd.concat(tdfs, axis=1)
    df.columns = pd.MultiIndex.from_product([setups, ['Cycles', 'IPC']])
    for i in range(len(setups)):
        setup = setups[i]
        df[setup,'Cycles'] = df[setup,'Cycles'].astype(int)
        if i == 0:
            continue
        df[setup,'Up'] = df[setups[0],'Cycles'] / df[setup,'Cycles']
        df[setup].style.background_gradient(subset=pd.IndexSlice[:, pd.IndexSlice[:, 'Up']])
    df.style.apply(lambda x: ['background: lightblue' for i in x])

    # display(df)
    ipc_table = df.to_html(classes="table is-narrow is-bordered")

    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Perf Report</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css">
    </head>
    <body>
    <section class="section">
        <div class="container content">
        <h1 class="title">
            Perf Report
        </h1>
        <p class="subtitle">
            {title}
        </p>
        <h2>TMA</h2>
        <p>{tma_table}</p>
        <p>{svg_html}</p>
        <h2>Cycles & IPC</h2>
        <p>{ipc_table}</p>
        </div>
    </section>
    </body>
    </html>
    '''

    f = open('perf/report.html','w')
    f.write(html)
    f.close()

def gem5_get_core_width(m5out):
    with open(f'{m5out}/config.ini', 'r') as f:
        for dataline in f:
            if "issueWidth" in dataline:
                return int(dataline.split('=')[1][:-1])

def gem5_get_perf_data(datapath, ncores):
    ffi = FFI()

    with open("../../../src/perf-data.h", "r") as f:
        cdefs = f.read()
        ffi.cdef(cdefs)
        f.close()

    perf_data = ffi.new("tma_data_t[]", 8)

    with open(datapath, "rb") as f:
        f.readinto(ffi.buffer(perf_data))

    EVS = cdata_dict(ffi, perf_data)

    CYCLES = []
    for i in range(ncores):
        CYCLES.append(EVS[i]['cycles'])

    EV = EVS[0]

    METRICS = dict()

    CLKS = EV['cycles']
    PipelineWidth = 2
    SLOTS = PipelineWidth * CLKS
    IPC = EV['instret'] / CLKS
    MACHINECLEAR = PipelineWidth * EV['machineClears']
    METRICS['Cycles'] = CLKS
    METRICS['IPC'] = Decimal(IPC).quantize(Decimal("0.00"))
    #level 0
    frontend_bound = EV['fetchBubbles'] / SLOTS * 100
    retiring = EV['instret'] / SLOTS * 100
    bad_speculaton = (EV['renamedInsts'] - EV['instret'] + \
                        EV['squashCycles'] * PipelineWidth) / SLOTS * 100
    backend_bound = 100 - (frontend_bound + bad_speculaton + retiring)

    METRICS['Front'] = Decimal(frontend_bound).quantize(Decimal("0.00")) # frontend_bound
    METRICS['Retire'] = Decimal(retiring).quantize(Decimal("0.00")) # retiring
    METRICS['BS'] = Decimal(bad_speculaton).quantize(Decimal("0.00")) # bad_speculaton
    METRICS['backend_bound'] = Decimal(backend_bound).quantize(Decimal("0.00"))
    
    #level 1
    fetch_latency_bound = EV['defetchLatencyCycles'] / CLKS * 100
    decode_latency_bound = (EV['refetchLatencyCycles'] - EV['defetchLatencyCycles']) \
                            / CLKS * 100
    fetch_bandwidth_bound = frontend_bound - (fetch_latency_bound + decode_latency_bound)
    br_mispred_fraction = EV['brMispredRetired'] / \
                            (EV['brMispredRetired'] + MACHINECLEAR)
    branch_mispredicts = br_mispred_fraction * bad_speculaton
    machine_clears = bad_speculaton - branch_mispredicts

    lsFraction = (EV['memStallsAnyLoad'] + EV['memStallsStores']) / \
                    EV['exeStallCycles']
    memory_bound = backend_bound * lsFraction
    core_bound = backend_bound * (1 - lsFraction)

    METRICS['fetch_latency_bound'] = Decimal(fetch_latency_bound).quantize(Decimal("0.00"))
    METRICS['fetch_bandwidth_bound'] = Decimal(fetch_bandwidth_bound).quantize(Decimal("0.00"))
    METRICS['decode_bound'] = Decimal(decode_latency_bound).quantize(Decimal("0.00"))
    METRICS['branch_mispredicts'] = Decimal(branch_mispredicts).quantize(Decimal("0.00"))
    METRICS['machine_clears'] = Decimal(machine_clears).quantize(Decimal("0.00"))

    METRICS['micro_sequencer'] = 0
    METRICS['base'] = METRICS['Retire']

    METRICS['MEM'] = Decimal(memory_bound).quantize(Decimal("0.00")) # memory_bound
    METRICS['CORE'] = Decimal(core_bound).quantize(Decimal("0.00")) # core_bound

    return METRICS, CYCLES

def cdata_dict(ffi, cd):
    if isinstance(cd, ffi.CData):
        try:
            return ffi.string(cd)
        except TypeError:
            try:
                return [cdata_dict(ffi, x) for x in cd]
            except TypeError:
                return {k: cdata_dict(ffi, v) for k, v in getmembers(cd)}
    else:
        return cd

def vcs_get_perf_data(datapath, ncores):
    ffi = FFI()

    with open("../../../src/perf-data.h", "r") as f:
        cdefs = f.read()
        ffi.cdef(cdefs)
        f.close()

    perf_data = ffi.new("tma_data_t[]", 8)

    with open(datapath, "rb") as f:
        f.readinto(ffi.buffer(perf_data))

    EVS = cdata_dict(ffi, perf_data)

    EV = EVS[0]
    
    METRICS = dict()

    PipelineWidth = 2
    CLKS = EV['cycles']
    IPC = EV['instret'] / CLKS
    SLOTS = PipelineWidth * CLKS
    MACHINE_CLEAR = PipelineWidth * EV['machineClears']
    RECOVERY_BUBBLES = PipelineWidth * EV['recoveryCycles']
    BR_MISPRED_FRACTION  = EV['brMispredRetired'] / (EV['brMispredRetired'] + EV['machineClears'])
    FETCH_LATENCY_CYCLES = EV['iCacheStallCycles'] + EV['iTLBStallCycles'] + EV['badResteers'] + EV['unknowBanchCycles']
    MEM_LATENCY_CYCLES   = EV['memStallsStores'] + EV['memStallsAnyLoad']
    METRICS['Cycles'] = CLKS
    METRICS['IPC'] = Decimal(IPC).quantize(Decimal("0.00"))
    #level 0
    frontend_bound = EV['fetchBubbles'] / SLOTS * 100
    retiring = EV['instret'] / SLOTS * 100
    backend_bound  = 100 - frontend_bound - (EV['slotsIssed'] + RECOVERY_BUBBLES) * 100 / SLOTS
    bad_speculaton = 100 - (frontend_bound + retiring + backend_bound)

    METRICS['Front'] = Decimal(frontend_bound).quantize(Decimal("0.00")) # frontend_bound
    METRICS['Retire'] = Decimal(retiring).quantize(Decimal("0.00")) # retiring
    METRICS['BS'] = Decimal(bad_speculaton).quantize(Decimal("0.00")) # bad_speculaton
    METRICS['backend_bound'] = Decimal(backend_bound).quantize(Decimal("0.00"))
    # level-1
    lsFraction = (EV['memStallsAnyLoad'] + EV['memStallsStores']) / EV['exeStallCycles']
    memory_bound = backend_bound * lsFraction
    core_bound = backend_bound * (1 - lsFraction)
    METRICS['MEM'] = Decimal(memory_bound).quantize(Decimal("0.00")) # memory_bound
    METRICS['CORE'] = Decimal(core_bound).quantize(Decimal("0.00")) # core_bound
    
    return METRICS

def get_perf_data(datapath, simulator, ncores):
    if simulator == 'vcs':
        return vcs_get_perf_data(datapath, ncores)
    elif simulator == 'gem5':
        return gem5_get_perf_data(datapath, ncores)

if __name__ == "__main__":
    generate_perf_report(title, setups)
# %%
