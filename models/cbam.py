import torch.nn as nn
import torch

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16) -> None:
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction # 中间通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # B C H W   C = 256~1000 多 没有太大计算量
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=channel,out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel,out_features=channel),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avgx = self.mlp(self.avg_pool(x).view(x.size(0), -1)).view(x.size(0),-1,1,1)
        maxx = self.mlp(self.max_pool(x).view(x.size(0), -1)).view(x.size(0),-1,1,1)
        return self.sigmoid(avgx + maxx)  # B C 1 1

class SpatialAttentionModule(nn.Module):
    def __init__(self) -> None:
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgx = torch.mean(x, dim=1, keepdim=True) # B 1 W H
        maxx, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv(torch.cat([avgx, maxx], dim=1))
        return self.sigmoid(x)
 

class CBAM(nn.Module):
    def __init__(self, channel) -> None:
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x # 之前这里忘记赋值了
        return x

class CBAQM(nn.Module):
    def __init__(self, in_channel, in_h, in_w, num_qery=300, out_channel=512) -> None:
        super(CBAQM, self).__init__()
        self.cbam = CBAM(in_channel)
        self.linear1 = nn.Linear(in_h*in_w, num_qery) # 256 维度
        self.linear2 = nn.Linear(in_channel, out_channel) # 256*2 维度

    def forward(self, x):
        '''
        path = 'out'
        import os 
        if not os.path.exists(path):
            os.mkdir(path)
        for i in range(x.shape[1]):
            ansx(x, i, path+'/in'+str(i))
        '''

        x = self.cbam(x) # B C H W
        x = self.linear1(x.view(x.size(0), x.size(1), -1)) # B C 300
        x = self.linear2(torch.transpose(x,1,2)) # B 300 512

        '''
        with open('cbaqm.txt', 'w') as f:
                  for i in x[0]:
                    f.write(str(i))
        '''
        return x
        

def ansx(x, i, name):
    showx = x[0].permute(2,1,0) # C H W
    showx = showx.cpu().numpy()
    import cv2
    import numpy as np

    showx = showx - np.min(showx) # 正数
    showx = showx/np.max(showx)-np.min(showx) # 归一化
    cv2.imwrite(name + '.png', showx[:,:,i]*255)



if __name__ == '__main__':
    '''
    cam = ChannelAttentionModule(256).to('cuda')
    sam = SpatialAttentionModule().to('cuda')
    cabm = CBAM(256).to('cuda') '''

    H_W = 28

    cbaqm =  CBAQM(256, H_W, H_W).to('cuda')
    x = torch.randn(2,256,H_W,H_W, dtype=torch.float32, device='cuda')
    
    import time
    ts = time.time()
    for i in range(100):
        out = cbaqm(x)
    t = time.time() - ts
    print(t/100)
    
    out = cbaqm(x)
    
    out = 1