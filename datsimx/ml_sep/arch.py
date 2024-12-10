
import torch
from torch import nn
from copy import deepcopy as decop

    
def pad_from_ks(ks):
    return int(round(ks/2))-1

def conv_norm(*args, **kwargs):
    """accepts all conv2d args, 
    returns a batch normalized conv with relu activation"""
    conv = nn.Conv2d(*args, **kwargs)
    bn = nn.BatchNorm2d(conv.out_channels)
    relu = nn.ReLU()
    return nn.Sequential(conv, bn, relu)


class squeezeBlock(nn.Module):

    def __init__(self, inchan, squeeze_chan, outchan):
        super().__init__()
        convA = conv_norm(inchan,squeeze_chan,kernel_size=1,padding=0,bias=False)
        convB = conv_norm(squeeze_chan,squeeze_chan, kernel_size=3, padding=1,bias=False)
        convC = conv_norm(squeeze_chan,outchan,kernel_size=1,padding=0,bias=False)
        self.convs = nn.Sequential(convA, convB, convC)
        if inchan == outchan:
            self.input_mod = nn.Identity()
        else:
            self.input_mod = conv_norm(inchan, outchan, kernel_size=1, bias=False)

    def forward(self, x):
        return self.convs(x)


class resnetDown(nn.Module):
    def __init__(self, wts=None, ks=24):
        if wts is None:
            wts = [1,1,2,2,2,1]
        assert all (w > 0 for w in wts)
        assert len(wts) == 6
        assert ks >= 7 and ks <= 32
        super().__init__()

        self.mx = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        # treat the input image, expects either 832 x 832 images or 512 x 512 images 
        self.input_mod = nn.Sequential(conv_norm(1, 32, kernel_size=ks, stride=2, padding=pad_from_ks(ks)), self.mx)

        # make the residual blocks ...
        self.sb64 = [squeezeBlock (32, 32, 128)] + [squeezeBlock(128,32,128)]*wts[0]
        self.sb256 =  [squeezeBlock(128, 64, 256)] +  [squeezeBlock(256,64,256)]*wts[1]
        self.sb512 =  [squeezeBlock(256, 128, 512)] +  [squeezeBlock(512,128,512)]*wts[2]
        self.sb1024 = [squeezeBlock(512, 256, 1024)] +  [squeezeBlock(1024,256,1024)]*wts[3]
        self.sb2048 = [squeezeBlock(1024, 512, 2048)] +  [squeezeBlock(2048,512,2048)]*wts[4]
        self.sb4096 = [squeezeBlock(2048, 1024, 4096)] +  [squeezeBlock(4096,1024,4096)]*wts[5]

    def forward( self, x):
        x = self.input_mod(x)
        for blocks in [self.sb64, self.sb256, self.sb512, self.sb1024, self.sb2048, self.sb4096]:
            for calc_resid in blocks:
                x = calc_resid(x) + calc_resid.input_mod(x)
            x = self.mx(x)
        x = self.avg(x)
        return x.flatten(1,-1)

class predictMulti(resnetDown):
    def __init__(self, wts=None, in_feat=4096, hidden_feat=1024, out_feat=1):
        super().__init__(wts)
        fc_hidden = nn.Linear(in_feat, hidden_feat)
        relu = nn.ReLU()
        fc_out = nn.Linear(hidden_feat, out_feat)
        self.predict = nn.Sequential(fc_hidden, relu, fc_out)

    def forward(self, x):
        x = super().forward(x)
        x = self.predict(x)
        return x 

