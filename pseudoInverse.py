import torch
from torch.autograd import Variable

class pseudoInverse(object):
    def __init__(self, params, C = 1e-2):
        self.params=list(params)
        self.C=C
        self.w = self.params[len(self.params) - 1]
        # print(self.w)   # [torch.FloatTensor of size 10x1000]
        self.w.data.fill_(0.0)

        # For sequential learning in OS-ELM
        self.dimInput=self.params[len(self.params)-1].data.size()[1]
        print('1) dimInput = ', self.dimInput)  # 1) dimInput =  1000
        self.M = Variable(torch.inverse(self.C * torch.eye(self.dimInput)))


    def train(self, inputs, targets):
        oneHotTarget        = self.oneHotVectorize(targets=targets)
        dimInput            = inputs.size()[1]
        print('2) dimInput  = ', dimInput)

        xtx= torch.mm(inputs.t(), inputs)   # ATA
        print(xtx.size(), ' = ', inputs.t().size(), ' X ', inputs.size())

        I = Variable(torch.eye(dimInput))
        print('I = ', I.size())

        self.M = Variable(torch.inverse(xtx.data + self.C*I.data))    # (ATA)-1
        w = torch.mm(self.M, inputs.t())    # (ATA)-1AT
        print('1) w = ',  w.size())
        w = torch.mm(w, oneHotTarget)       # A+B = X
        print('2) w = ', w.size())

        self.w.data = w.t().data
        print('3) w = ', w.size())

    def oneHotVectorize(self,targets):
        oneHotTarget = torch.zeros(targets.size()[0],targets.max().data[0]+1)
        print('oneHot = ', oneHotTarget.size())

        for i in range(targets.size()[0]):
            oneHotTarget[i][targets[i].data[0]]=1

        oneHotTarget = Variable(oneHotTarget)

        return oneHotTarget