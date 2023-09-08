import torch
import torch.functional as F
from src.models.components import torch_4DVarNN_dinAE as NN_4DVar

DimAE = 20
dW = 2
genSuffixModel = '_GENN_%02d_%02d ' %(DimAE ,dW)
shapeData = [1, 40, 1]


class Encoder_L96(torch.nn.Module):
    def __init__(self):
        super(Encoder_L96, self).__init__()
        self.F = torch.nn.Parameter(torch.Tensor([8.]))
        self.dt = 0.01
        self.IntScheme = 1
        # self.stdTr = stdTr
        # self.meanTr = meanTr
        # self.conv1     = torch.nn.Conv2d(1,shapeData[0],1,padding=0,bias=False)

        self.conv1 = torch.nn.Conv2d(1, 1, (5, 1), padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(1, 1, (3, 1), padding=0, bias=False)

        # predefined parameters
        K = torch.Tensor([-1., 0., 0., 1., 0.]).view(1, 1, 5, 1)
        self.conv1.weight = torch.nn.Parameter(K)
        K = torch.Tensor([1., 0., 0.]).view(1, 1, 3, 1)
        self.conv2.weight = torch.nn.Parameter(K)

    def _odeL96(self, xin):
        x_1 = torch.cat((xin[:, :, xin.size(2) - 2:, :], xin, xin[:, :, 0:2, :]), dim=2)
        # x_1 = x_1.view(-1,1,xin.size(1)+4,xin.size(2))
        x_1 = self.conv1(x_1)
        # x_1 = x_1.view(-1,xin.size(1),xin.size(2))

        x_2 = torch.cat((xin[:, :, xin.size(2) - 1:, :], xin, xin[:, :, 0:1, :]), dim=2)
        # x_2 = x_2.view(-1,1,xin.size(1)+2,xin.size(2))
        x_2 = self.conv2(x_2)
        # x_2 = x_2.view(-1,xin.size(1),xin.size(2))

        dpred = x_1 * x_2 - xin + self.F

        return dpred.view(-1, xin.size(1), xin.size(2), xin.size(3))

    def _EulerSolver(self, x):
        return x + self.dt * self._odeL96(x)

    def _RK4Solver(self, x):
        k1 = self._odeL96(x)
        x2 = x + 0.5 * self.dt * k1
        k2 = self._odeL96(x2)

        x3 = x + 0.5 * self.dt * k2
        k3 = self._odeL96(x3)

        x4 = x + self.dt * k3
        k4 = self._odeL96(x4)

        return x + self.dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.

    def forward(self, x):
        # X = stdTr * x
        # X = X + meanTr
        X = x
        if self.IntScheme == 0:
            xpred = self._EulerSolver(X[:, :, :, :])
        else:
            xpred = self._RK4Solver(X[:, :, :, :])

        # xpred = xpred - meanTr
        # xpred = xpred / stdTr

        xnew = torch.cat((x[:, :, :, 0].view(-1, 1, x.size(2), 1), xpred), dim=3)
        return xnew

class Encoder(torch.nn.Module):
    def __init__(self, shapeData):
        super(Encoder, self).__init__()
        self.conv1  = NN_4DVar.ConstrainedConv2d(1,DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
        self.conv21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
        self.conv22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
        self.conv23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
        self.conv3  = torch.nn.Conv2d(2*DimAE,1,(1,1),padding=0,bias=False)
        #self.conv4 = torch.nn.Conv1d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,1,padding=0,bias=False)
        self.shapeData = shapeData
        #self.conv2Tr = torch.nn.ConvTranspose1d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,4,stride=4,bias=False)
        #self.conv5 = torch.nn.Conv1d(8*shapeData[0]*DimAE,16*shapeData[0]*DimAE,3,padding=1,bias=False)
        #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)

    def forward(self, xin):
        x_1 = torch.cat((xin[:,:,xin.size(2)-dW:,:],xin,xin[:,:,0:dW,:]),dim=2)
        #x_1 = x_1.view(-1,1,xin.size(1)+2*dW,xin.size(2))
        x   = self.conv1( x_1 )
        x   = x[:,:,dW:xin.size(2)+dW,:]
        x   = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
        x   = self.conv3( x )
        #x = self.conv4( F.relu(x) )
        x = x.view(-1,self.shapeData[0],self.shapeData[1],self.shapeData[2])
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        return torch.mul(1. ,x)

class Model_AE(torch.nn.Module):
    def __init__(self, shapeData):
        super(Model_AE, self).__init__()
        self.encoder = Encoder(shapeData)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder( x )
        x = self.decoder( x )
        return x

class Model_L96(torch.nn.Module):
    def __init__(self):
        super(Model_L96, self).__init__()
        self.encoder = Encoder_L96()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x