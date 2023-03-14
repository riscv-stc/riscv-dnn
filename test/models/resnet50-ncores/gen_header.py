#!/home/kening.zhang/stc-v2/stc-verification/isa/riscv-tests/isa/.env/bin/python

import os
import numpy as np


outdir = "bindata/"

inhead = '''
#ifndef __RESNET50V2_WEIGHTDATA_
#define __RESNET50V2_WEIGHTDATA_

#include <stdint.h>
#include "../../../include/incbin.h"

'''


infile =''

section = ''' ".scdata.params" '''


for i in range(49):
    fhead = outdir + "batch_normalization_"
    gamend = "gamma.bin"
    betaend = "beta.bin"
    meanend = "moving_mean.bin"
    varend = "moving_variance.bin"
    epsd = 0.000010009999641624745
    if i == 0:
        gamd = np.fromfile(fhead+gamend, dtype=np.float32).astype(np.float64)
        betad = np.fromfile(fhead+betaend, dtype=np.float32).astype(np.float64)
        meand = np.fromfile(fhead+meanend, dtype=np.float32).astype(np.float64)
        vard = np.fromfile(fhead+varend, dtype=np.float32).astype(np.float64)
        outalpha = outdir + "batch_normalization_new_alpha.bin"
        outbeta = outdir + "batch_normalization_new_beta.bin"
    else:
        gamd = np.fromfile(fhead + str(i) + "_" + gamend, dtype=np.float32).astype(np.float64)
        betad = np.fromfile(fhead + str(i) + "_" + betaend, dtype=np.float32).astype(np.float64)
        meand = np.fromfile(fhead + str(i) + "_" + meanend, dtype=np.float32).astype(np.float64)
        vard = np.fromfile(fhead + str(i) + "_" + varend, dtype=np.float32).astype(np.float64)
        outalpha = outdir + "batch_normalization_" + str(i) + "_new_alpha.bin"
        outbeta = outdir + "batch_normalization_" + str(i) + "_new_beta.bin"
    
    alphad = np.multiply(gamd, np.reciprocal(np.sqrt(vard + epsd)))
    betanew = np.subtract(betad, np.multiply(meand, alphad))
    alphad.astype('float16').tofile(outalpha)
    betanew.astype('float16').tofile(outbeta)


with open('resnet50_parameters.h', 'w') as fhead:
    fhead.write(inhead)
    for f in os.listdir(outdir):
        if '.bin' in f:
            var = f.split('.')[0] + '_data'
            if "kernel" in f:
                fhead.write("extern uint8_t " + var + "[];\n")
            else:
                fhead.write("extern uint8_t " + var + "[];\n")
            fhead.write("INCBIN(" + var + ", \"" + f + "\", " + section + ");\n\n")


    fhead.write(infile)
    fhead.write("\n#endif\n")
