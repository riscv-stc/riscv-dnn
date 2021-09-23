#!/home/kening.zhang/stc-v2/stc-verification/isa/riscv-tests/isa/.env/bin/python
import os
import sys
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from decimal import Decimal
sys.path.append("..") 
from check import check,get_perf,get_tma

opt_level = {"O0":0, "O2":2}

perf = ['Charis', 'Cycles', 'IPC', 'Front', 'BS', 'MEM', 'CORE', 'Retire', 'Up']
cols = ['operate',] + perf * len(opt_level)
output = pd.DataFrame(columns = cols)

simulator = 'spike'
GEM5 = "/home/kening.zhang/stc-exp/simulator/gem5"
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)

def batchnorm(hin, win, cin):
    vs1 = np.random.random((hin, win, cin)).astype('float32') * 2 -1
    mean = np.random.random(cin).astype('float32') * 2 - 1
    var = np.random.random(cin).astype('float32')
    gam = np.random.random(cin).astype('float32') * 2 - 1 
    beta = np.random.random(cin).astype('float32') * 2 - 1
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

def batchnorm2(hin, win, cin):
    vs1 = np.random.random((hin, win, cin)).astype('float16') * 2 -1
    gam = np.random.random(cin).astype('float16') * 2 - 1 
    beta = np.random.random(cin).astype('float16') * 2 - 1
    vd = np.multiply(gam, vs1)  + beta
    vs1.astype('float16').tofile("src.bin")
    gam.astype('float16').tofile("gam.bin")
    beta.astype('float16').tofile("beta.bin")
    vd.astype('float16').tofile('golden.bin')

def test_batchnorm(num, caseSize):
    h = caseSize[0]
    w = caseSize[1]
    cin = caseSize[2]
    # batchnorm(h, w, cin)
    batchnorm2(h, w, cin)
    if simulator == 'spike':
        os.system("spike --isa=rv64gcv_zfh --varch=vlen:1024,elen:64,slen:1024 pk rvv-test \
                    %d %d %d" % (h, w, cin))
    else:
        # --debug-flags=O3PipeView --debug-start=78580000 --debug-file=trace.out \
        os.system('''%s/build/RISCV/gem5.opt --tmaPipelineWidth=5 \
                    %s/configs/example/se.py  --cpu-type=StcBoom --bp-type=LTAGE \
                    --num-cpu=1 --mem-channels=1 --mem-size=3072MB --caches --l1d_size=64kB --l1i_size=64kB \
                    --cacheline_size=128 --l1i_assoc=8 --l1d_assoc=8 --l2cache --l2_size=64MB  --cmd=rvv-test \
                    -o "%d %d %d" > log 2>&1''' 
                    % (GEM5, GEM5, h, w, cin))
        return get_tma('m5out/stats.txt')
    # check(str(num), w*cin*256)
    
if __name__ == "__main__":
    case_sizes = (
                  (16, 16, 16),
                  (64, 64, 64),
                  (16, 16, 130),
                  (128, 128, 130),
                  (56, 56, 64)
                  )
    for i in range(len(case_sizes)):
        perf_data = ["BatchNorm", ]
        chara_start = 1
        base_perf = 1
        for key,val in opt_level.items():
            os.system("clang++ -g --target=riscv64-unknown-elf  -march=rv64gv0p10zfh0p1 \
                        -menable-experimental-extensions -mllvm  -riscv-v-vector-bits-min=128 \
                        -O%d -o rvv-test test_batchnorm.cpp  ../../src/mat.cpp" % val)
            tma = test_batchnorm(key+'-'+str(i), case_sizes[i])
            tma["Charis"] = key
            if val == 0:
                base_perf = tma["Cycles"]
                tma["Up"] = 1.0
            else:
                up = base_perf / tma["Cycles"]
                tma['Up'] = Decimal(up).quantize(Decimal("0.000"))
            for col in perf:
                perf_data.append(tma[col])
        output.loc[i] = perf_data
    output.to_csv('perf.csv')




