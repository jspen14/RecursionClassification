import torchvision.models as models
from torch import nn

class ConvNetwork(nn.Module):
    def __init__(self, data, numClasses):
        super(ConvNetwork, self).__init__()
        x, y = data[0]
        channels, height, width = x.size()
        output = numClasses

        self.net = models.vgg19(pretrained=True)
        self.net.features[0] = nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.net.classifier[6] = nn.Linear(4096, numClasses)
#        self.net = nn.Sequential(
#            nn.Conv2d(channels, 10, (3,3), padding=(1,1)),
#            nn.ReLU(),
#            nn.Conv2d(10, 100, (5,5), padding=(1,1)),
#            nn.ReLU(),
#            nn.Conv2d(100, 300, (5,5), padding=(1,1)),
#            nn.ReLU(),
#            nn.Linear()
#        )

    def forward(self,x):
        return self.net(x)
    
    def freezeLayers(numLayers):
        pass


