#!/usr/bin/python3
import numpy as np
import os
from decimal import Decimal

wdic = {16: 4, 32: 8}
floatdic = {16: np.float16, 32: np.float32}
intdic = {16: np.int16, 32: np.int32}

t = 'f'

def check(num, acc, res='dst.bin', golden='golden.bin', dbits=16):
    res = np.fromfile(res, dtype=floatdic[dbits])
    golden = np.fromfile(golden, dtype=floatdic[dbits])
    w = wdic[dbits]

    if np.size(res)!= np.size(golden):
        print("Error size %d vs %d" %(np.size(res), np.size(golden)))
        exit(-1)
    reshex = res.copy()
    goldenhex = golden.copy()
    reshex.dtype = intdic[dbits]
    goldenhex.dtype = intdic[dbits]
    errh = np.abs(res - golden)
    errhex = np.abs(reshex - goldenhex)

    checkFile = open('check' + num + '.dat', 'w')
    error = res - golden
    maxerr = np.max(np.abs(error))
    relerr = np.abs( (res-golden) / golden)
    relerrmax = np.max(np.where(relerr < np.inf, relerr, 0))
    error = error.flatten().astype(np.float32)
    variance = np.sum(error * error) / len(error)
    res_max_index = np.argwhere(res == np.max(res)).flatten().tolist()
    golden_max_index = np.argwhere(golden == np.max(golden)).flatten().tolist()
    print(f'max error : %f' % maxerr, file=checkFile)
    print(f'variance  : %f' % variance, file=checkFile)
    print(f'RVV max   : %f - ' % np.max(res),  res_max_index, file=checkFile)
    print(f'Python max: %f - ' % np.max(golden),  golden_max_index, file=checkFile)

    print( f'\n         %{2*w+12}s  %{2*w+12}s  %{2*w+12}s' % ( 'RVV', 'Python', 'Err' ), file=checkFile)
    isfailed = False

    for i in range(np.size(res)):
        resh = res[i]
        goldenh = golden[i]
        if errh[i] > 0.0005 * acc and errhex[i] > 1:
            print(f'%8d: %{w+10}{t}(%0{w}x), %{w+10}{t}(%0{w}x), %{w+10}{t}(%0{w}x), mismatch' % (i, resh, abs(reshex[i]), goldenh, abs(goldenhex[i]), errh[i], abs(errhex[i])), file=checkFile)
            isfailed = True
        else:
            print(f'%8d: %{w+10}{t}(%0{w}x), %{w+10}{t}(%0{w}x), %{w+10}{t}(%0{w}x)' % (i, resh, abs(reshex[i]), goldenh, abs(goldenhex[i]), errh[i], abs(errhex[i])), file=checkFile)
            

    checkFile.close()
    if isfailed:
        print(num + " FAILED")
    else:
        print(num + " PASS")
    
    # print(num, maxerr, variance, res_max_index, golden_max_index)

def get_perf(fname, start_flag, end_flag) :
    start_list = []
    end_list = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            if start_flag in line:
                start_list.append(line)
            if end_flag in line:
                end_list.append(line)
    start_tick = int(start_list[0].split(":")[0])
    end_tick = int(end_list[-1].split(":")[0])
    ticks = end_tick - start_tick
    return ticks


def get_PipelineWidth(stats_file):
    dirpath = os.path.dirname(stats_file)
    config_ini = os.path.join(dirpath, "config.ini")
    with open(config_ini, 'r') as f:
        for dataline in f:
            if "issueWidth" in dataline:
                PipelineWidth=int(dataline.split('=')[1][:-1])
                break
    return PipelineWidth

def get_tma(stats_file):
    EV = dict() 
    METRICS = dict()
    with open(stats_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.isspace():
                continue
            else:
                data = line.split()
                if len(data) > 1 and data[1] != '' and data[1].isdigit():
                    EV[data[0]] = float(data[1])
    CLKS = EV['system.cpu.numCycles']
    PipelineWidth = get_PipelineWidth(stats_file)
    SLOTS = PipelineWidth * CLKS
    IPC = EV['system.cpu.committedOps'] / CLKS
    MACHINECLEAR = PipelineWidth * EV['system.cpu.fetch.machineClearCycles']
    METRICS['Cycles'] = CLKS
    METRICS['IPC'] = Decimal(IPC).quantize(Decimal("0.00"))
    #level 0
    frontend_bound = EV['system.cpu.rename.fetchBubblesInsts'] / SLOTS * 100
    retiring = EV['system.cpu.commit.opsCommitted'] / SLOTS * 100
    bad_speculaton = (EV['system.cpu.rename.renamedInsts'] - EV['system.cpu.commit.opsCommitted'] + \
                        EV['system.cpu.rename.squashCycles'] * PipelineWidth) / SLOTS * 100
    backend_bound = 100 - (frontend_bound + bad_speculaton + retiring)

    METRICS['Front'] = Decimal(frontend_bound).quantize(Decimal("0.00")) # frontend_bound
    METRICS['Retire'] = Decimal(retiring).quantize(Decimal("0.00")) # retiring
    METRICS['BS'] = Decimal(bad_speculaton).quantize(Decimal("0.00")) # bad_speculaton
    METRICS['backend_bound'] = Decimal(backend_bound).quantize(Decimal("0.00"))
    
    #level 1
    fetch_latency_bound = EV['system.cpu.decode.fetchLatencyCycles'] / CLKS * 100
    decode_latency_bound = (EV['system.cpu.rename.fetchLatencyCycles'] - EV['system.cpu.decode.fetchLatencyCycles']) \
                            / CLKS * 100
    fetch_bandwidth_bound = frontend_bound - (fetch_latency_bound + decode_latency_bound)
    br_mispred_fraction = EV['system.cpu.commit.branchMispredicts'] / \
                            (EV['system.cpu.commit.branchMispredicts'] + MACHINECLEAR)
    branch_mispredicts = br_mispred_fraction * bad_speculaton
    machine_clears = bad_speculaton - branch_mispredicts

    lsFraction = (EV['system.cpu.iewAnyLoadStallCycles'] + EV['system.cpu.iewStoresStallCycles']) / \
                    EV['system.cpu.iewExecStallCycles']
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

    return METRICS

def get_tma_log(log_file):
    '''
    0:csr_cycle  1:csr_opsCommitted   2:csr_machineClearCycles  3:csr_defetchLatencyCycles  4:csr_refetchLatencyCycles  
    5:csr_fetchBubblesInsts    6:csr_renamedInsts        7:csr_squashCycles         8:csr_iewExecStallCycle       ,
    9:csr_iewAnyLoadStallCycles   10:csr_iewStoresStallCycles   11:csr_branchMispredicts
    '''
    METRICS = dict()
    perf = []
    with open(log_file, 'r') as f:
        for lines in f:
            if "Perf" in lines:
                print(lines)
                perf_str = lines.split(":")[1].split()
                break
    for s in perf_str:
        perf.append(int(s))
    
    CLKS = perf[0]
    METRICS['Cycles'] = perf[0]
    PipelineWidth = get_PipelineWidth('m5out/config.ini')
    SLOTS = PipelineWidth * perf[0]
    IPC = perf[1] / CLKS
    MACHINECLEAR = PipelineWidth * perf[2]
    METRICS['IPC'] = Decimal(IPC).quantize(Decimal("0.00"))
    #level 0
    frontend_bound = perf[4] / SLOTS * 100
    retiring = perf[1] / SLOTS * 100
    bad_speculaton = (perf[6] - perf[1] + perf[7] * PipelineWidth) / SLOTS * 100
    backend_bound = 100 - (frontend_bound + bad_speculaton + retiring)

    METRICS['Front'] = Decimal(frontend_bound).quantize(Decimal("0.00")) # frontend_bound
    METRICS['Retire'] = Decimal(retiring).quantize(Decimal("0.00")) # retiring
    METRICS['BS'] = Decimal(bad_speculaton).quantize(Decimal("0.00")) # bad_speculaton
    METRICS['backend_bound'] = Decimal(backend_bound).quantize(Decimal("0.00"))
    
    #level 1
    fetch_latency_bound = perf[3] / CLKS * 100
    decode_latency_bound = (perf[4] - perf[3]) / CLKS * 100
    fetch_bandwidth_bound = frontend_bound - (fetch_latency_bound + decode_latency_bound)
    br_mispred_fraction = perf[11] / (perf[11] + MACHINECLEAR)
    branch_mispredicts = br_mispred_fraction * bad_speculaton
    machine_clears = bad_speculaton - branch_mispredicts

    lsFraction = (perf[9] + perf[10]) / perf[8]
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

    return METRICS





    
