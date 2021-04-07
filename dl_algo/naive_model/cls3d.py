import torch
import torch.nn as nn
import math


class NaiveClsModel(nn.Module):

    def __init__(self, num_classes=2):
        super(NaiveClsModel, self).__init__()

        self.conv_layer1 = self._make_conv_layer(1, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 64)
        
        self.avg_pool = nn.AdaptiveAvgPool3d((2, 1, 1))
        self.fc = nn.Linear(128*2, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(2, 3, 3), padding=1),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = NaiveClsModel(num_classes=2)
    input = torch.zeros((3, 1, 38, 320, 416))

    model(input)

