#!/usr/bin/python3
import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append("..") 
from check import check

o_level = 2
fun_start = "Z4reluR3MatS0" if o_level == 0 else "main+400"
fun_end = fun_start if o_level == 0 else "main+492"

simulator = 'spike'
GEM5 = "/home/kening.zhang/stc-exp/simulator/gem5"
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)

def relu(hin, win, cin, base):
    vs1 = np.random.random((hin, win, cin)).astype('float16') * 2 - 1
    base = np.array(base).astype('float16')
    vd = np.where(vs1 > base, vs1, base)
    vd.astype('float16')

    vs1.tofile('src.bin')
    base.tofile('base.bin')
    vd.tofile('golden.bin')

def test_relu(num, h, w, cin, base):
    relu(h, w, cin, base)
    if simulator == 'spike':
        os.system("spike --isa=rv64gcv_zfh --varch=vlen:1024,elen:64,slen:1024 pk rvv-test %d %d %d" % (h, w, cin))
    else:
        os.system('''%s/build/RISCV/gem5.opt --debug-flags=Exec %s/configs/example/se.py  --cpu-type=BoomCPU --bp-type=LTAGE \
                    --num-cpu=1 --mem-channels=1 --mem-size=3072MB --caches --l1d_size=32kB --l1i_size=32kB \
                    --cacheline_size=64 --l1i_assoc=8 --l1d_assoc=8 --l2cache --l2_size=512kB --cmd=rvv-test \
                    -o "%d %d %d" > log 2>&1''' 
                    % (GEM5, GEM5, h, w, cin))
    check('Case'+str(num), 0)
    os.system('''
                 a=`grep -inr %s log | head -1 | awk -F ':' '{print $2}'`
                 b=`grep -inr %s log | tail -1 | awk -F ':' '{print $2}'`
                 echo "Cycles: "$(($b-$a))
              ''' % (fun_start, fun_end))

os.system("clang++  --target=riscv64-unknown-elf  -march=rv64gv0p10zfh0p1 -menable-experimental-extensions -mllvm -riscv-v-vector-bits-min=128 -O%d -o rvv-test test_relu.cpp  ../../src/mat.cpp" % o_level)
test_relu(2, 2, 2, 1, 0)
test_relu(3, 5, 5, 5, 0)
test_relu(4, 5, 5, 65, 0)
test_relu(4, 5, 5, 130, 0)

test_relu(5, 2, 2, 1, 0.1)
test_relu(6, 5, 5, 5, -0.2)
test_relu(7, 5, 5, 65, 0.5)
test_relu(8, 5, 5, 130, 0.8)
test_relu(9, 5, 5, 130, -0.8)

test_relu(10, 5, 5, 513, 0)
test_relu(11, 5, 5, 1024, 0)


