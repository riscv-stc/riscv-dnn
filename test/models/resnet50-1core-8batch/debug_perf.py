#!/home/kening.zhang/stc-v2/stc-verification/isa/riscv-tests/isa/.env/bin/python

import os
import numpy as np

param = "numPhysVecRegs"
origin = 256
decr = 32

old = origin
new = origin

ThreadHold = 1475800

fperf = open("perf.data", 'w')

while True:
    print(f"{param} = {new} Running......")
    os.system("cd /home/kening.zhang/work/stc-exp/gem5-temp && bsub -Ip ./build-ubuntu.sh > log 2>&1 && cd -")
    os.system('''bsub -Ip /home/kening.zhang/work/stc-exp/gem5-temp/build/RISCV/gem5.opt --outdir=m5out-tmep \\
                --listener-mode=off  /home/kening.zhang/work/stc-exp/gem5-temp/configs/example/fs.py \\
                --cpu-type=StcBoom --bp-type=LTAGE --num-cpu=1 \\
                --mem-channels=1 --mem-size=3072MB --caches --cacheline_size=128  --enable_vcache  \\
                --l1d_size=64kB --l1i_size=64kB --l1i_assoc=8 --l1d_assoc=8 --l3cache \\
                --l3_size=8MB --l2_size=512kB --l1v_num_banks=8 --l2_num_banks=8 --l3_num_banks=8 \\
                --vmuLoadPorts=4 --vmuStorePorts=2  --kernel=build/loop4-0/test.elf > log 2>&1''')

    with open("m5out-tmep/system.cpu.terminal", 'r') as f:
        lines= f.readlines()
        fperf.write(f"{param} = {new} :\n")
        for line in lines:
            fperf.write(line)
        cycles = int(lines[-1])
        print(f"\tPerf: {cycles}")
        # if cycles > 1475800:
        if old <= 128:
            print(f"The best {param} is {old}")
            break

    new = old - decr

    start=False
    file_data=""
    with open("/home/kening.zhang/work/stc-exp/gem5-temp/src/cpu/o3/O3CPU.py", 'r') as fcfg:
        for line in fcfg:
            if "class StcBoom" not in line:
                start=True
            if start:
                if f"{param} = {old}" in line:
                    line = line.replace(f"{param} = {old}", f"{param} = {new}")
            file_data+=line

    with open("/home/kening.zhang/work/stc-exp/gem5-temp/src/cpu/o3/O3CPU.py", 'w') as fcfg:
        fcfg.write(file_data)

    old = new
    
fperf.close()