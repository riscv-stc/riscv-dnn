import os

os.system('rm *.bin')

os.system("touch conv2d.bin batch_normalization.bin Relu.bin add.bin max_pooling2d.bin")

for i in range(1, 53):
    os.system("touch conv2d_%d.bin" %i)

for i in range(1, 49):
    os.system("touch batch_normalization_%d.bin Relu_%d.bin" %(i, i))

for i in range(1, 16):
    os.system("touch add_%d.bin" % i)

os.system("touch Mean.bin MatMul.bin BiasAdd.bin softmax_tensor_fp16.bin")

    