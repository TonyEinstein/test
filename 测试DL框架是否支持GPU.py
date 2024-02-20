#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2024/2/20 15:40
# file: 测试DL框架是否支持GPU.py
# author: chenruhai
# email: ruhai.chen@qq.com



try:
    # 判断是否安装了cuda
    import torch
    print("torch：是否已安装CUDA：\t",torch.cuda.is_available())  # 返回True则说明已经安装了cuda
    # 判断是否安装了cuDNN
    from torch.backends import cudnn
    print("torch：是否安装cudnn：\t",cudnn.is_available())  # 返回True则说明已经安装了cuDNN

    ngpu = 1
    # 判断是要在GPU还是在CPU上运行，如果GPU可用那么在GPU。
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("torch：程序运行计算的设备编号：\t",device)
    print("torch：程序运行计算的设备名称：\t",torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())
except Exception as e:
    print(e)

print("--------------"*5)
try:
    import tensorflow as tf

    print("tensorflow：可用的GPU数量: \t", len(tf.config.experimental.list_physical_devices('GPU')))

    a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用
    # b = tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)  # 判断GPU是否可以用
    if tf.config.list_physical_devices('GPU'):
        b = True
    else:
        b = False

    print("tensorflow：CUDA是否可用：\t",a)
    print("tensorflow：GPU是否可用：\t",b)
    print("tensorflow：GPU详细信息：：\t",tf.config.list_physical_devices('GPU'))
except Exception as e:
    print(e)


