import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        print(f"proj_query: {proj_query.shape}")
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        print(f"proj_key: {proj_key.shape}")
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        print(f"attention: {attention.shape}")
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        print(f"proj_value: {proj_value.shape}")
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        print(f"out: {out.shape}")
        out = out.view(m_batchsize, C, height, width)
        print(f"out: {out.shape}")
        out = self.gamma*out + x
        print(f"final: {out.shape}")
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



if __name__ == '__main__':
    in_channel = 256
    dropout = 0

    pa = PAM_Module(256)
    paremeters = sum(p.numel() for p in pa.parameters() if p.requires_grad)
    print(f"paremeters: {paremeters}")

    ca = CAM_Module(256)
    paremeters = sum(p.numel() for p in ca.parameters() if p.requires_grad)
    print(f"paremeters: {paremeters}")


    x = torch.rand((2, in_channel, 56, 56))

    # obj_feat = torch.Size([200, 256])

    print(f"Input: {x.shape}")
    out = pa(x)
    print(f"out: {out.shape}")