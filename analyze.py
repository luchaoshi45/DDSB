import numpy as np
from os import path
import os
import matplotlib.pyplot as plt
from matplotlib import cm,ticker
import json
import warnings

def analyze(path, interval=[0,1]):
    '''
    待分析 log txt 文件路径
    interval: 显示区间
    '''
    out = open(path, encoding='utf-8')
    lines = out.readlines()
    
    #提取trainLoss和validationLoss
    trainLoss=[]
    valloss = []

    if path[-4:] == ".txt":
        for line in lines:
            if "train_loss" in line:
                val = json.loads(line)['train_loss']
                trainLoss.append(val)
    elif path[-4:] == ".log":
        for line in lines:
            if "Epoch:"in line and "loss:" in line:
                info = line
                index = info.find('loss:')##从下标0开始，查找在字符串里第一个出现的子串，返回结果：0
                str = line[index+6:index+12]
                trainLoss.append(float(str))
            if "Test:"in line and "loss:" in line:
                info = line
                index = info.find('loss:')##从下标0开始，查找在字符串里第一个出现的子串，返回结果：0
                str = line[index+6:index+12]
                valloss.append(float(str))
    else:
        warnings.warn("文件类型错误", UserWarning)
        print("文件类型错误: {}".format(path))

    
    trainLoss = trainLoss[int(interval[0]*len(trainLoss)): int(interval[1]*len(trainLoss))]
    if len(valloss) != 0:
        valloss = valloss[int(interval[0]*len(valloss)): int(interval[1]*len(valloss))]
        
    #绘图
    
    if len(valloss) != 0:
        fig=plt.figure()
        plt.subplot(211)
        epochNum=len(trainLoss)
        xs=np.arange(epochNum)
        #plt.yticks(np.arange(-1,0,0.1))
        plt.plot(xs, trainLoss, color='coral', label="train loss")
        plt.legend()

        plt.subplot(212)
        epochNum=len(valloss)
        xs=np.arange(epochNum)
        plt.plot(xs, valloss, color='cyan', label="val loss")
        plt.legend()
    else:
        fig=plt.figure()
        epochNum=len(trainLoss)
        xs=np.arange(epochNum)
        #plt.yticks(np.arange(-1,0,0.1))
        plt.plot(xs, trainLoss, color='coral', label="train loss")
        plt.legend()

    plt.savefig(path[:-4] + "_loss.png")
    #plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('analyze', add_help=False)
    parser.add_argument('--base_dir', default='runs/analyze', type=str)
    args = parser.parse_args()
    base_dir = args.base_dir
    
    for name in os.listdir(base_dir):
        if '.log' in name or '.txt' in name:
            analyze(base_dir+'/'+name, [0.,1])  

 
 
 