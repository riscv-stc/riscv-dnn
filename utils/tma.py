#!/usr/bin/python3
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import math
import re
import os

def GetEV(retpath):
    EV = dict()
    with open(retpath, "r") as f:
        lines = f.readlines()
        for line in lines:
          if re.search(r"\w:(-*)\d+$", line):
            data = line.split(":")
            EV[data[0]] = abs(int(data[1].rstrip()))
    return EV

def GetMetrics():
    logpath = 'vcs.log'
    EV = GetEV(logpath)
    METRICS = dict()

    PipelineWidth = 2
    CLKS  = EV['cycles']
    IPC   = EV['instret'] / CLKS
    SLOTS = PipelineWidth * CLKS
    MACHINE_CLEAR = PipelineWidth * EV['machineClears']
    RECOVERY_BUBBLES = PipelineWidth * EV['recoveryCycles']
    BR_MISPRED_FRACTION  = EV['brMispredRetired'] / (EV['brMispredRetired'] + EV['machineClears'])
    FETCH_LATENCY_CYCLES = EV['iCacheStallCycles'] + EV['iTLBStallCycles'] + EV['badResteers'] + EV['unknowBanchCycles']
    MEM_LATENCY_CYCLES   = EV['memStallsStores'] + EV['memStallsAnyLoad']
    METRICS['Cycles'] = CLKS
    METRICS['IPC'] = round(IPC, 3)
    #level 0
    METRICS['frontend_bound'] = round(EV['fetchBubbles'] * 100 / SLOTS, 2)
    METRICS['retiring']       = round(EV['instret'] * 100 / SLOTS, 2)
    METRICS['backend_bound']  = max(0, round(100 - METRICS['frontend_bound'] - (EV['slotsIssed'] + RECOVERY_BUBBLES) * 100 / SLOTS, 2))
    METRICS['bad_speculaton'] = max(0, round(100 - (METRICS['frontend_bound'] + METRICS['backend_bound'] + METRICS['retiring']), 2))

    #level 1
    #METRICS['fetch_latency_bound']   = METRICS['frontend_bound']
    METRICS['fetch_latency_bound']   = 100

    METRICS['branch_mispredicts']  = round(BR_MISPRED_FRACTION * 100, 2)
    METRICS['machine_clears']      = round(100 - METRICS['branch_mispredicts'], 2)

    METRICS['retiredIntRatio']     = round(EV['intTotalRetired'] * 100 / EV['instret'], 2)
    METRICS['retiredFloatRatio']   = round(EV['fpTotalRetired']  * 100 / EV['instret'], 2)
    METRICS['retiredRvvRatio']     = round(EV['rvvTotalRetired'] * 100 / EV['instret'], 2)

    METRICS['memory_bound']     = round((EV['memStallsAnyLoad'] + EV['memStallsStores']) * 100 / EV['exeStallCycles'], 2)
    METRICS['core_bound']       = max(0, round(100 - METRICS['memory_bound'], 2))

    # leve 2
    METRICS['icache_miss']      = round(EV['iCacheStallCycles'] * 100 / FETCH_LATENCY_CYCLES, 2)
    METRICS['itlb_miss']        = round(EV['iTLBStallCycles'] * 100 / FETCH_LATENCY_CYCLES, 2)
    METRICS['branch_resteers']  = max(round((EV['badResteers'] + EV['unknowBanchCycles']) * 100 / FETCH_LATENCY_CYCLES, 2), 0)

    METRICS['branchRatio']      = round(EV['branchRetired'] * 100 / EV['intTotalRetired'], 2)
    METRICS['dividerRatio']     = round(EV['intDividerRetired'] * 100/ EV['intTotalRetired'], 2)
    METRICS['intOtherRatio']    = round(100 - METRICS['branchRatio'] - METRICS['dividerRatio'], 2)

    METRICS['fpDividerRatio']   = round(EV['fpDividerRetired'] * 100 / EV['fpTotalRetired'], 2)
    METRICS['fpOtherRatio']     = round(100 - METRICS['fpDividerRatio'], 2)

    METRICS['rvvVsetRatio']     = round(EV['rvvVsetRetired'] * 100  / EV['rvvTotalRetired'], 2)
    METRICS['rvvLoadRatio']     = round(EV['rvvLoadRetired'] * 100  / EV['rvvTotalRetired'], 2)
    METRICS['rvvStoreRatio']    = round(EV['rvvStoreRetired'] * 100 / EV['rvvTotalRetired'], 2)
    METRICS['rvvOtherRatio']    = round(100 - METRICS['rvvVsetRatio'] - METRICS['rvvLoadRatio'] - METRICS['rvvStoreRatio'], 2)

    METRICS['divider']          = round(EV['divBusyCycles'] * 100 / CLKS / METRICS['core_bound'], 2)
    METRICS['exe_ports_util']   = round(100 - METRICS['divider'], 2)
    #METRICS['exe_ports_util']   = round((CORE_BOUND_CYCLES() - EV['divBusyCycles']) * 100 / CLKS(), 2)

    METRICS['store_bound']      = round(EV['memStallsStores'] * 100 / MEM_LATENCY_CYCLES, 2)
    METRICS['l1_bound']         = round((EV['memStallsAnyLoad'] - EV['memStallsL1Miss']) * 100 / MEM_LATENCY_CYCLES, 2)
    METRICS['ext_memory_bound'] = round((EV['memStallsL1Miss'] * 100 / MEM_LATENCY_CYCLES), 2)

    #level 3
    METRICS['mispredicts_resteers'] = round(BR_MISPRED_FRACTION * EV['badResteers'] * 100 / CLKS, 2)
    METRICS['Mclear_resteers']      = round((1 - BR_MISPRED_FRACTION) * EV['badResteers'] * 100 / CLKS, 2)
    METRICS['unknow_branches']      = round(EV['unknowBanchCycles'] * 100 / CLKS, 2)

    #METRICS['serializing_Op'] = round(EV['robStallCycles'] * 100 / CLKS(), 2)
    METRICS['memUnitUtil'] = round(EV['memUnitUtilization'] * 100 / CLKS, 2)
    METRICS['jmpUnitUtil'] = round(EV['jmpUnitUtilization'] * 100 / CLKS, 2)
    METRICS['aluUnitUtil'] = round(EV['aluUnitUtilization'] * 100 / CLKS, 2)
    METRICS['fpuUnitUtil'] = round(EV['fpuUnitUtilization'] * 100 / CLKS, 2)
    METRICS['vecUnitUtil'] = round(EV['vecUnitUtilization'] * 100 / CLKS, 2)
    METRICS['vmxUnitUtil'] = round(EV['vmxUnitUtilization'] * 100 / CLKS, 2)

    METRICS['mem_latency']   = round(EV['memLatency'] * 100 / EV['memStallsL1Miss'], 2)
    METRICS['mem_bandwidth'] = round(100 - METRICS['mem_latency'], 2)

    return METRICS

def label(x, y, width, height, rotation, text, text_size, dy = 1):
    xx = x + width / 2
    yy = y + height / 2 - dy
    plt.text(xx, yy, text, ha="center", family='sans-serif', rotation = rotation, size=text_size)

def PlotMetrics(case_title):
    METRICS = GetMetrics()
    XYLevel0 = dict()
    XYLevel1 = dict()
    XYLevel2 = dict()
    XYLevel3 = dict()
    XYLevel4 = dict()

    fig, ax = plt.subplots()
    pads = 0.2
    smallPads = 3 * pads
    largePads = 4 * pads

    ###################
    #level0
    ###################
    height = 5
    width = 12
    text_size = 6
    rotation = 0

    x = 0
    y = 0
    XYLevel0['frontend_bound'] = (x, y, width, height)
    frontend_bound = mpathes.FancyBboxPatch((x, y),width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(frontend_bound)
    text = "Frontend" +"\n" + "Bound" + "\n" + str(METRICS['frontend_bound']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 2)


    x = x + width + largePads
    XYLevel0['bad_speculaton'] = (x, y, width, height)
    bad_speculation = mpathes.FancyBboxPatch((x, y), width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(bad_speculation)
    text = "Bad" +"\n" + "Speculation" + "\n" + str(METRICS['bad_speculaton']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 2)


    x = x + width + largePads
    XYLevel0['retiring'] = (x, y, 1.8 * width, height)
    retiring = mpathes.FancyBboxPatch((x, y), 1.8 * width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(retiring)
    text = "Retiring" + "\n" + str(METRICS['retiring']) + "%"
    label(x, y, 1.8 * width, height, rotation, text, text_size)


    x = x + 1.8 * width + largePads
    XYLevel0['backend_bound'] = (x, y, 1.8 * width, height)
    backend_bound = mpathes.FancyBboxPatch((x, y), 1.8 * width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(backend_bound)
    text = "Backend Bound" + "\n" + str(METRICS['backend_bound']) + "%"
    label(x, y, 1.8 * width, height, rotation, text, text_size)

    ###################
    #level1
    ###################
    level = 1
    text_size = 6
    rotation = 0

    width = XYLevel0['frontend_bound'][2]
    x = XYLevel0['frontend_bound'][0]
    y = y - height - largePads
    XYLevel1['fetch_latency'] = (x, y,  width, height)
    fetch_latency = mpathes.FancyBboxPatch((x, y), width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(fetch_latency)
    text = "Fetch" +"\n" + "Latency" + "\n" + str(METRICS['fetch_latency_bound']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 2)


    #------------------
    width = (XYLevel0['bad_speculaton'][2] - 1 * largePads) / 2
    x = XYLevel0['bad_speculaton'][0]
    XYLevel1['branch_mispredicts'] = (x, y, width, height)
    branch_mispredicts = mpathes.FancyBboxPatch((x, y), width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(branch_mispredicts)
    text = "Branch" +"\n" + "Mispred" + "\n" + str(METRICS['branch_mispredicts']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 2)

    x = x + width + largePads
    XYLevel1['machine_clears'] = (x, y, width, height)
    machine_clears = mpathes.FancyBboxPatch((x, y), width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(machine_clears)
    text = "Machine" +"\n" + "Clears" + "\n" + str(METRICS['machine_clears']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 2)


    #------------------
    width = (XYLevel0['retiring'][2] - 2 * largePads) * 3 / 9
    x = XYLevel0['retiring'][0]
    XYLevel1['scalarInt'] = (x, y, width, height)
    intRetiredRatio = mpathes.FancyBboxPatch((x, y), width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(intRetiredRatio)
    text = "Int" +"\n" + str(METRICS['retiredIntRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size)

    x = x + width + largePads
    width = (XYLevel0['retiring'][2] - 2 * largePads) * 2 / 9
    XYLevel1['scalarFloat'] = (x, y, width, height)
    fpRetiredRatio = mpathes.FancyBboxPatch((x, y), width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(fpRetiredRatio)
    text = "Float" +"\n" + str(METRICS['retiredFloatRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size)

    x = x + width + largePads
    width = (XYLevel0['retiring'][2] - 2 * largePads) * 4 / 9
    XYLevel1['vector'] = (x, y, width, height)
    rvvRetiredRatio = mpathes.FancyBboxPatch((x, y), width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(rvvRetiredRatio)
    text = "Vector" +"\n" + str(METRICS['retiredRvvRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size)

    #------------------
    x = XYLevel0['backend_bound'][0]
    width = (XYLevel0['backend_bound'][2] - 1 * largePads) * 7 / 11
    XYLevel1['core_bound'] = (x, y, width, height)
    core_bound = mpathes.FancyBboxPatch((x, y), width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(core_bound)
    text = "Core" +"\n" + "Bound" + "\n" + str(METRICS['core_bound']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 2)

    x = x + width + largePads
    width = (XYLevel0['backend_bound'][2] - 1 * largePads) * 4 / 11
    XYLevel1['memory_bound'] = (x, y, width, height)
    memory_bound = mpathes.FancyBboxPatch((x, y), width, height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(memory_bound)
    text = "Memory" +"\n" + "Bound" + "\n" + str(METRICS['memory_bound']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 2)


    ###################
    #level2
    ###################
    level = 2
    text_size = 5
    rotation = 90

    width = (XYLevel1['fetch_latency'][2] - 2 * smallPads)*1 / 5
    x = XYLevel1['fetch_latency'][0]
    y = y - 2.2 * height - largePads
    XYLevel2['itlb_miss'] = (x, y, width, height)
    itlb_miss = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(itlb_miss)
    text = "iTLB Miss " + str(METRICS['itlb_miss']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel2['icache_miss'] = (x, y, width, height)
    icache_miss = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(icache_miss)
    text = "iCache Miss " + str(METRICS['icache_miss']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    width = width * 3
    XYLevel2['branch_resteers'] = (x, y, width, height)
    branch_resteers = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(branch_resteers)
    text = "Branch Resteers\n" + str(METRICS['branch_resteers']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    #------------------
    width = (XYLevel1['scalarInt'][2] -  2 * smallPads) / 3
    x = XYLevel1['scalarInt'][0]
    XYLevel2['branchRatio'] = (x, y, width, height)
    branch_arith = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(branch_arith)
    text = "branch  " + str(METRICS['branchRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel2['dividerRatio'] = (x, y, width, height)
    divider_ratio = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(divider_ratio)
    text = "divider " + str(METRICS['dividerRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel2['intOtherRatio'] = (x, y, width, height)
    int_other_ratio = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(int_other_ratio)
    text = "others  " + str(METRICS['intOtherRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    #------------------
    width = (XYLevel1['scalarFloat'][2] - smallPads) / 2
    x = XYLevel1['scalarFloat'][0]
    XYLevel2['fpDividerRatio'] = (x, y, width, height)
    fp_divider_ratio = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(fp_divider_ratio)
    text = "fdiv " + str(METRICS['fpDividerRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel2['fpOtherRatio'] = (x, y, width, height)
    fp_other_ratio = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(fp_other_ratio)
    text = "fpu  " + str(METRICS['fpOtherRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    #------------------
    width = (XYLevel1['vector'][2] - 3* smallPads) / 4
    x = XYLevel1['vector'][0]
    XYLevel2['rvvVsetRatio'] = (x, y, width, height)
    rvv_vset_ratio = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(rvv_vset_ratio)
    text = "vset   " + str(METRICS['rvvVsetRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel2['rvvLoadRatio'] = (x, y, width, height)
    rvv_load_ratio = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(rvv_load_ratio)
    text = "vload  " + str(METRICS['rvvLoadRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel2['rvvStoreRatio'] = (x, y, width, height)
    rvv_store_ratio = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(rvv_store_ratio)
    text = "vstore " + str(METRICS['rvvStoreRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel2['rvvOtherRatio'] = (x, y, width, height)
    rvv_other_ratio = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(rvv_other_ratio)
    text = "others " + str(METRICS['rvvOtherRatio']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    #------------------
    width = (XYLevel1['core_bound'][2] - smallPads) / 7
    x = XYLevel1['core_bound'][0]
    XYLevel2['divider'] = (x, y, width, height)
    divider_busy_cycles = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(divider_busy_cycles)
    text = "Divider " + str(METRICS['divider']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    width = (XYLevel1['core_bound'][2] - smallPads) * 6 / 7
    rotation = 0
    XYLevel2['exe_ports_util'] = (x, y, width, height)
    exe_ports_util = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(exe_ports_util)
    text = "Execution \n Ports \n Utilization\n" + str(METRICS['exe_ports_util']) + "%\n\n"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    #------------------
    rotation = 90
    width = (XYLevel1['memory_bound'][2] -  2 * smallPads) / 5
    x = XYLevel1['memory_bound'][0]
    XYLevel2['store_bound'] = (x, y, width, height)
    store_bound = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(store_bound)
    text = "store bound " + str(METRICS['store_bound']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel2['l1_bound'] = (x, y, width, height)
    l1_bound = mpathes.FancyBboxPatch((x, y), width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(l1_bound)
    text = "L1 bound " + str(METRICS['l1_bound']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    rotation = 0
    x = x + width + smallPads
    XYLevel2['ext_memory_bound'] = (x, y, 3 * width, height)
    ext_memory_bound = mpathes.FancyBboxPatch((x, y), 3 * width, 2.2 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(ext_memory_bound)
    text = "ext mem\n" + "bound \n" + str(METRICS['ext_memory_bound']) + "%\n\n"
    label(x, y, 3 * width, height, rotation, text, text_size, dy = 1.5)


    ###################
    #level3
    ###################
    level = 3
    text_size = 5
    rotation = 90
    y = y - 2.5 * height - largePads

    #------------------
    # width = (XYLevel2['branch_resteers'][2] - 2*smallPads) / 3
    # x = XYLevel2['branch_resteers'][0]
    # XYLevel3['mispredicts_resteers'] = (x, y, width, height)
    # mispredicts_resteers = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    # ax.add_patch(mispredicts_resteers)
    # text = "Misp_Resteers " + str(METRICS['mispredicts_resteers']) + "%"
    # label(x, y, width, height, rotation, text, text_size)

    # x = x + width + smallPads
    # XYLevel3['Mclear_resteers'] = (x, y, width, height)
    # Mclear_resteers = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    # ax.add_patch(Mclear_resteers)
    # text = "Mclear_resteers " + str(METRICS['Mclear_resteers']) + "%"
    # label(x, y, width, height, rotation, text, text_size)

    # x = x + width + smallPads
    # XYLevel3['unknow_branches'] = (x, y, width, height)
    # unknow_branches = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    # ax.add_patch(unknow_branches)
    # text = "unknow_branches " + str(METRICS['unknow_branches']) + "%"
    # label(x, y, width, height, rotation, text, text_size)

    #---------
    width = (XYLevel2['exe_ports_util'][2] - 5 * smallPads) / 6
    x = XYLevel2['exe_ports_util'][0]
    XYLevel3['memUnitUtil'] = (x, y, width, height)
    mem_unit_util = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(mem_unit_util)
    text = "mem unit " + str(METRICS['memUnitUtil']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel3['jmpUnitUtil'] = (x, y, width, height)
    jmp_unit_util = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(jmp_unit_util)
    text = "jmp unit " + str(METRICS['jmpUnitUtil']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel3['aluUnitUtil'] = (x, y, width, height)
    alu_unit_util = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(alu_unit_util)
    text = "alu unit " + str(METRICS['aluUnitUtil']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel3['fpuUnitUtil'] = (x, y, width, height)
    fpu_unit_util = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(fpu_unit_util)
    text = "fpu unit " + str(METRICS['fpuUnitUtil']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel3['vecUnitUtil'] = (x, y, width, height)
    vec_unit_util = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(vec_unit_util)
    text = "vec unit " + str(METRICS['vecUnitUtil']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel3['vmxUnitUtil'] = (x, y, width, height)
    vmx_unit_util = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(vmx_unit_util)
    text = "vmx unit " + str(METRICS['vmxUnitUtil']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    #------------------
    width = (XYLevel2['ext_memory_bound'][2] -  1 * smallPads) / 2
    x = XYLevel2['ext_memory_bound'][0]
    XYLevel3['mem_latency'] = (x, y, width, height)
    mem_latency = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(mem_latency)
    text = "latency  " + str(METRICS['mem_latency']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    x = x + width + smallPads
    XYLevel3['mem_bandwidth'] = (x, y, width, height)
    mem_bandwidth = mpathes.FancyBboxPatch((x, y), width, 2.5 * height, color='r', alpha=0.3, boxstyle=mpathes.BoxStyle("Round", pad=pads))
    ax.add_patch(mem_bandwidth)
    text = "bandwidth " + str(METRICS['mem_bandwidth']) + "%"
    label(x, y, width, height, rotation, text, text_size, dy = 1.5)

    plt.axis('equal')
    plt.axis('off')
    plt.suptitle(case_title, y = 0.9)
    plt.title("IPC = " + str(METRICS['IPC']), y = 0)
    plt.savefig('test.png', dpi=300)
