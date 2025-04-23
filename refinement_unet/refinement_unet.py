import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Double Convolution Block used in UNet ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

# --- Basic UNet Refiner: Improves segmentation in high-uncertainty regions ---
class BasicUNetRefiner(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128]):
        super(BasicUNetRefiner, self).__init__()
        # Encoder path.
        self.encoder1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = DoubleConv(features[1], features[2])
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck.
        self.bottleneck = DoubleConv(features[2], features[2]*2)
        
        # Decoder path.
        self.up3 = nn.ConvTranspose3d(features[2]*2, features[2], kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features[2]*2, features[2])
        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features[1]*2, features[1])
        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features[0]*2, features[0])
        
        self.conv_last = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder.
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # Bottleneck.
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder.
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.conv_last(dec1)
        return torch.sigmoid(out)
