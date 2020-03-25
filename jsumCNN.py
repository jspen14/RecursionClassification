import torchvision.models as models
from torch import nn

class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()

    def forward(self, v1, v2):
        pass

class meanAttention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()

    def forward(self, v1, v2):
        pass

class cycleGenerator(nn.Module):
    def __init__(self, data, numClasses):
        super(ConvNetwork, self).__init__()
        x, y = data[0]
        channels, height, width = x.size()
        output = numClasses

        self.sourceImgNet = nn.Sequential()
        self.batchImgNet = nn.Sequential()

    def forward(self,x):
        return self.net(x)
    
    def freezeLayers(numLayers):
        pass


