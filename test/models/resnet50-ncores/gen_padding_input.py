#python

import numpy as np

origin = np.fromfile("bindata/imagenet_pic_data.bin", dtype=np.int16)
origin = origin.reshape((64, 224, 224, 3))

output = np.pad(origin, ((0,0), (3,3), (3,3),(0,0)), 'constant', constant_values=(0, 0))

output.tofile("bindata/imagenet_pic_data_pad.bin")