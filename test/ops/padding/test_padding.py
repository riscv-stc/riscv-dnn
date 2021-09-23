#!/usr/bin/python3
import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append("..") 
from check import check

o_level = 2
fun_start = "Z7paddingR3" if o_level == 0 else "main+416"
fun_end = fun_start if o_level == 0 else "main+616"

simulator = 'spike'
GEM5 = "/home/kening.zhang/stc-exp/simulator/gem5"
if len(sys.argv) > 1:
    simulator = sys.argv[1]
print("run on %s" % simulator)

def padding(hin, win, cin, pt=0, pb=0, pl=0, pr=0):
    shape_input = [hin, win, cin]
    vs1 = np.random.random(shape_input).astype('float16') * 2 - 1
    vd = np.pad(vs1, ((pt, pb), (pl, pr), (0, 0)))

    vs1.tofile("src.bin")
    vd.tofile('golden.bin')

def test_padding(num, h, w, cin, pt=0, pb=0, pl=0, pr=0):
    padding(h, w, cin, pt, pb, pl , pr)
    if simulator == 'spike':
        os.system("spike --isa=rv64gcv_zfh --varch=vlen:1024,elen:64,slen:1024 pk rvv-test %d %d %d %d %d %d %d" % (h, w, cin, pt, pb, pl, pr))
    else:
        os.system('''%s/build/RISCV/gem5.opt --debug-flags=Exec %s/configs/example/se.py  --cpu-type=BoomCPU --bp-type=LTAGE \
                    --num-cpu=1 --mem-channels=1 --mem-size=3072MB --caches --l1d_size=32kB --l1i_size=32kB \
                    --cacheline_size=64 --l1i_assoc=8 --l1d_assoc=8 --l2cache --l2_size=512kB --cmd=rvv-test \
                    -o "%d %d %d %d %d %d %d" > log 2>&1''' 
                    % (GEM5, GEM5, h, w, cin, pt, pb, pl, pr))
    check(str(num), 0)
    os.system('''
                 a=`grep -inr %s log | head -1 | awk -F ':' '{print $2}'`
                 b=`grep -inr %s log | tail -1 | awk -F ':' '{print $2}'`
                 echo "Cycles: "$(($b-$a))
              ''' % (fun_start, fun_end))

os.system("clang++  --target=riscv64-unknown-elf  -march=rv64gv0p10zfh0p1 -menable-experimental-extensions -mllvm -riscv-v-vector-bits-min=128 -O%d -o rvv-test test_padding.cpp  ../../src/mat.cpp" % o_level)
test_padding(2, 5, 5, 3, 0, 0, 0, 0)
test_padding(3, 7, 7, 3, 1, 1, 1, 1)
test_padding(4, 7, 7, 3, 3, 3, 3, 3)
test_padding(5, 9, 9, 2, 5, 4, 6, 7)
test_padding(6, 5, 5, 130, 3, 3, 3, 3)
