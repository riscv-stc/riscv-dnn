#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from decimal import Decimal
sys.path.append("..") 
from check import check,get_perf,get_tma

TEST_OPTION = {"func": False, "perf": True}

opt_level = {"O0":0, "O2":2}


perf = ['Charis', 'Cycles', 'IPC', 'Front', 'BS', 'MEM', 'CORE', 'Retire', 'Up']
cols = ['operate',] + perf * len(opt_level)
output = pd.DataFrame(columns = cols)

simulator = 'spike'
GEM5 = "/home/kening.zhang/stc-exp/simulator/gem5"
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)

def cast_f32_to_f16(hin, win, cin, cout):
    vs1 = np.random.random((hin, win, cin, cout)).astype('float32') * 2 - 1
    vd = vs1.copy().astype('float16')

    vs1.tofile('src.bin')
    vd.tofile('golden.bin')

def test_cast_f32_to_f16(num, caseSize):
    h = caseSize[0]
    w = caseSize[1]
    cin = caseSize[2]
    cout = caseSize[2]
    cast_f32_to_f16(h, w, cin, cout)
    if simulator == 'spike':
        os.system("spike --isa=rv64gcv_zfh --varch=vlen:1024,elen:64,slen:1024 pk rvv-test \
                  %d %d %d %d" % (h, w, cin, cout))
    else:
        # --debug-flags=O3PipeView --debug-start=78580000 --debug-file=trace.out \
        os.system('''%s/build/RISCV/gem5.opt --tmaPipelineWidth=5 \
                    %s/configs/example/se.py  --cpu-type=StcBoom --bp-type=LTAGE \
                    --num-cpu=1 --mem-channels=1 --mem-size=3072MB --caches --l1d_size=64kB --l1i_size=64kB \
                    --cacheline_size=128 --l1i_assoc=8 --l1d_assoc=8 --l2cache --l2_size=64MB  --cmd=rvv-test \
                    -o "%d %d %d %d" > log 2>&1''' 
                    % (GEM5, GEM5, h, w, cin, cout))
        os.system(''' %s/util/o3-pipeview.py -c 500 -o pipeview.out --color m5out/trace.out'''
                    % GEM5)
        if TEST_OPTION['perf']:
            return get_tma('m5out/stats.txt')
    if TEST_OPTION['func']:
        check(str(num), w1)

if __name__ == "__main__":
    case_sizes = (
                  (16, 16, 16, 1),
                  (32, 32, 32, 16),
                  (64, 64, 64, 64),
                  (65, 300, 66, 2),
                  )
    for i in range(len(case_sizes)):
        perf_data = ["Cast", ]
        chara_start = 1
        base_perf = 1
        for key,val in opt_level.items():
            os.system("clang++ -g --target=riscv64-unknown-elf  -march=rv64gv0p10zfh0p1 \
                        -menable-experimental-extensions -mllvm  -riscv-v-vector-bits-min=128 \
                        -O%d -o rvv-test test_cast.cpp  ../../src/mat.cpp" % val)
            
            tma = test_cast_f32_to_f16(key+'-'+str(i), case_sizes[i])
            if TEST_OPTION['perf']:    
                tma["Charis"] = key
                if val == 0:
                    base_perf = tma["Cycles"]
                    tma["Up"] = 1.0
                else:
                    up = base_perf / tma["Cycles"]
                    tma['Up'] = Decimal(up).quantize(Decimal("0.000"))
                for col in perf:
                    perf_data.append(tma[col])
        if TEST_OPTION['perf']:
            output.loc[i] = perf_data
    if TEST_OPTION['perf']:
        output.to_csv('perf.csv')


