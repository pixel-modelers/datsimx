
import torch
from torch import nn

    
def pad_from_ks(ks):
    return int(round(ks/2))-1

def conv_norm(*args, **kwargs):
    """accepts all conv2d args, 
    returns a batch normalized conv with relu activation"""
    conv = nn.Conv2d(*args, **kwargs)
    bn = nn.BatchNorm2d(conv.out_channels)
    relu = nn.ReLU()
    return nn.Sequential(conv, bn, relu)

def convtrans_norm(*args, **kwargs):
    """accepts all conv2d args, 
    returns a batch normalized conv with relu activation"""
    conv = nn.ConvTranspose2d(*args, **kwargs)
    bn = nn.BatchNorm2d(conv.out_channels)
    relu = nn.ReLU()
    return nn.Sequential(conv, bn, relu)


class inflateBlock(nn.Module):

    def __init__(self, inchan, inflate_chan, outchan):
        super().__init__()
        convtA = convtrans_norm(inchan,inflate_chan,kernel_size=3,padding=1,bias=False)
        convtB = convtrans_norm(inflate_chan,inflate_chan, kernel_size=1, padding=0,bias=False)
        convtC = convtrans_norm(inflate_chan,outchan,kernel_size=3,padding=1,bias=False)
        self.convs = nn.Sequential(convtA, convtB, convtC)

    def forward(self, x):
        return self.convs(x)


class residualBlock(nn.Module):

    def __init__(self, inchan, squeeze_chan, outchan):
        super().__init__()
        convA = conv_norm(inchan,squeeze_chan,kernel_size=1,padding=0,bias=False)
        convB = conv_norm(squeeze_chan,squeeze_chan, kernel_size=3, padding=1,bias=False)
        convC = conv_norm(squeeze_chan,outchan,kernel_size=1,padding=0,bias=False)
        self.convs = nn.Sequential(convA, convB, convC)


    def forward(self, x):
        return self.convs(x)


class resnetDown(nn.Module):
    def __init__(self, ks=7):
        super().__init__()

        self.mx = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        # make the residual blocks ...
        self.input_conv = conv_norm(1, 64, kernel_size=ks, stride=2, padding=pad_from_ks(ks))
        # make the residual blocks ...
        start_block_args = [(64, 64, 256),
                            (256, 128, 512), 
                            (512, 256, 1024), 
                            (1024, 512, 2048)] 
        Nmores = [2,3,5,2]
        block_args = [(256,64,256),
                      (512,128,512),
                      (1024,256,1024),
                      (2048,512,2048)]
        channel_dims = 256,512,1024,2048

        for i_dim, (Nmore, bl_arg) in enumerate(zip(Nmores, block_args)):
            start_bl_arg = start_block_args[i_dim]
            blocks = [residualBlock(*start_bl_arg)]
            mod_blocks = [conv_norm(start_bl_arg[0], start_bl_arg[2], kernel_size=1, bias=False)]
            for _ in range(Nmore):
                blocks.append(residualBlock(*bl_arg))
                mod_blocks.append( nn.Identity() )
            setattr(self, f"resid{channel_dims[i_dim]}", nn.ModuleList(blocks))
            setattr(self, f"mod{channel_dims[i_dim]}", nn.ModuleList(mod_blocks))
        
        self.x1_skip = selfx64_skip = self.x256_skip = self.x512_skip = self.x1024_skip = self.x2048_skip = None

    def unet_down(self, x):
        # initial conv and max pool:
        self.x1_skip = x
        x = self.input_conv(x)
        self.x64_skip = x
        x = self.mx(x)

        for i in range(len(self.resid256)):
            x = self.resid256[i](x) + self.mod256[i](x)
        self.x256_skip = x
        x = self.mx(x)

        for i in range(len(self.resid512)):
            x = self.resid512[i](x) + self.mod512[i](x)
        self.x512_skip = x
        x = self.mx(x)

        for i in range(len(self.resid1024)):
            x = self.resid1024[i](x) + self.mod1024[i](x)
        self.x1024_skip = x
        x = self.mx(x)

        for i in range(len(self.resid2048)):
            x = self.resid2048[i](x) + self.mod2048[i](x)
        self.x2048_skip = x
        x = self.mx(x)

        x = self.avg(x)
        return x

    def forward( self, x):
        x = self.unet_down(x)
        return x


class resnetU(nn.Module):
    def __init__(self, ks=7, down=None):
        super().__init__()
        if down is None:
            self.down = resnetDown(ks=ks)
        else:
            self.down = down

        self.up = nn.Upsample(scale_factor=(2,2))
        self.inv_avg = convtrans_norm(2048, 2048, kernel_size=13) # note, its 13 of 6 depending on whether input image is 832 or 512 ... 
        self.out_conv = convtrans_norm(64,1,kernel_size=ks+1,stride=2,padding=pad_from_ks(ks+1), bias=False)

        # define the decoder layers...
        inv_finish_block_args = [(2048, 4096, 1024),
                                (1024, 2048, 512), 
                                (512, 1024, 256), 
                                (256, 512, 64)]
        inv_block_args =[(2048,4096,2048),
                         (1024,2048,1024),
                         (512,1024,512),
                         (256,512,256)]
        inv_Nmores = [2,5,3,2]

        inv_channel_dims = 2048, 1024, 512, 256
        for i_dim, (Nmore, bl_arg) in enumerate(zip(inv_Nmores, inv_block_args)):
            blocks = []
            mod_blocks = []
            for _ in range(Nmore):
                blocks.append(inflateBlock(*bl_arg))
                mod_blocks.append( nn.Identity() )
            finish_arg = inv_finish_block_args[i_dim] 
            blocks.append( inflateBlock(*finish_arg))
            mod_blocks.append(convtrans_norm(finish_arg[0],finish_arg[2],kernel_size=1, bias=False))
            setattr(self, f"inv_resid{inv_channel_dims[i_dim]}",nn.ModuleList(blocks))
            setattr(self, f"inv_mod{inv_channel_dims[i_dim]}",nn.ModuleList(blocks))

    def forward(self, x):
        x = self.down(x)
        x = self.inv_avg(x)
        x = self.up(x)
        for i in range(len(self.inv_resid2048)):
            x = self.inv_resid2048[i](x) + self.inv_mod2048[i](x)*self.inv_mod2048[i](self.down.x2048_skip)

        x = self.up(x)
        for i in range(len(self.inv_resid1024)):
            x = self.inv_resid1024[i](x) + self.inv_mod1024[i](x)*self.inv_mod1024[i](self.down.x1024_skip)
        
        x = self.up(x)
        for i in range(len(self.inv_resid512)):
            x = self.inv_resid512[i](x) + self.inv_mod512[i](x)*self.inv_mod512[i](self.down.x512_skip)
        
        x = self.up(x)
        for i in range(len(self.inv_resid256)):
            x = self.inv_resid256[i](x) + self.inv_mod256[i](x)*self.inv_mod256[i](self.down.x256_skip)

        x = self.up(x)
        x = x + self.down.x64_skip
        x = self.out_conv(x)
        x = x + self.down.x1_skip
        return x


class predictMulti(nn.Module):
    def __init__(self, ks=7, in_feat=2048):
        super().__init__()
        self.rs = resnetDown(ks=ks)
        self.hidden1024 = nn.Linear(in_feat, 1000)
        self.relu = nn.ReLU()
        self.hidden128 = nn.Linear(1000, 100)
        self.out = nn.Linear(100,1)

    def forward(self, x):
        x = self.rs(x)
        x = x.flatten(1,-1)
        x = self.hidden1024(x)
        x = self.relu(x)
        x = self.hidden128(x)
        x = self.out(x)
        return x

