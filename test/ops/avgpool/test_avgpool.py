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

def test_avgpool(num, caseSize):
    stride_h = 1
    stride_w = 1
    pt = 0
    pb = 0
    pl = 0
    pr = 0
    if len(caseSize) >= 5:
        h = caseSize[0]
        w = caseSize[1]
        cin = caseSize[2]
        kh = caseSize[3]
        kw = caseSize[4]
    if len(caseSize) >= 7:
        stride_h = caseSize[5]
        stride_w = caseSize[6]
    if len(caseSize) >=11:
        pt = caseSize[7]
        pb = caseSize[8]
        pl = caseSize[9]
        pr = caseSize[10]
    avgpool(h, w, cin, kh, kw, stride_h, stride_w, pt, pb, pl , pr)
    if simulator == 'spike':
        os.system("spike --isa=rv64gcv_zfh --varch=vlen:1024,elen:64,slen:1024 pk rvv-test \
                   %d %d %d %d %d %d %d %d %d %d %d" \
                   % (h, w, cin, kh, kw, stride_h, stride_w, pt, pb, pl, pr))
    else:
       # --debug-flags=O3PipeView --debug-start=78580000 --debug-file=trace.out \
        os.system('''%s/build/RISCV/gem5.opt --tmaPipelineWidth=5 \
                    %s/configs/example/se.py  --cpu-type=StcBoom --bp-type=LTAGE \
                    --num-cpu=1 --mem-channels=1 --mem-size=3072MB --caches --l1d_size=64kB --l1i_size=64kB \
                    --cacheline_size=128 --l1i_assoc=8 --l1d_assoc=8 --l2cache --l2_size=64MB  --cmd=rvv-test \
                    -o "%d %d %d %d %d %d %d %d %d %d %d" > log 2>&1''' 
                    % (GEM5, GEM5, h, w, cin, kh, kw, stride_h, stride_w, pt, pb, pl, pr))
        os.system(''' %s/util/o3-pipeview.py -c 500 -o pipeview.out --color m5out/trace.out'''
                    % GEM5)
        if TEST_OPTION['perf']:
            return get_tma('m5out/stats.txt')
    if TEST_OPTION['func']:
        check(str(num), kh*kw)

if __name__ == "__main__":
    case_sizes = (
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
    for i in range(len(case_sizes)):
        perf_data = ["Avgpool", ]
        chara_start = 1
        base_perf = 1
        for key,val in opt_level.items():
            os.system("clang++ -g --target=riscv64-unknown-elf  -march=rv64gv0p10zfh0p1 \
                        -menable-experimental-extensions -mllvm  -riscv-v-vector-bits-min=128 \
                        -O%d -o rvv-test test_avgpool.cpp  ../../src/mat.cpp" % val)
            
            tma = test_avgpool(key+'-'+str(i), case_sizes[i])
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
