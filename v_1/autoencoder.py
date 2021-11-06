import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
  def __init__(self, code_size=1000):
    super().__init__()
    self.code_size = code_size

    def init_normal(m):
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform(m.weight)  
        m.bias.data.fill_(0.)

    self.encoder = nn.Sequential(self.conv_block(in_channels=3, out_channels=32),
                                 self.conv_block(in_channels=32, out_channels=64),
                                 nn.Dropout2d(p=0.2), 
                                 self.conv_block(in_channels=64, out_channels=128),
                                 self.conv_block(in_channels=128, out_channels=256), 
                                 nn.Conv2d(in_channels=256, out_channels=self.code_size, kernel_size=3), 
                                 nn.ELU(),
                                 nn.Flatten())
    
    self.decoder = nn.Sequential(nn.Unflatten(1, (self.code_size, 1, 1)), 
                                 nn.ConvTranspose2d(in_channels=self.code_size, out_channels=256, kernel_size=3),
                                 nn.ELU(),
                                 self.deconv_block(in_channels=256, out_channels=128),
                                 self.deconv_block(in_channels=128, out_channels=64),
                                 nn.Dropout2d(p=0.2),
                                 self.deconv_block(in_channels=64, out_channels=32),
                                 self.deconv_block(in_channels=32, out_channels=3))
    self.apply(init_normal)

  def forward(self, x):
    latent_code = self.encoder(x)
    reconstruction = self.decoder(latent_code)
    return reconstruction, latent_code
    # return self.encoder(x)

  def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, k_pooling=2):
    block = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding), 
                          nn.BatchNorm2d(num_features=out_channels), 
                          nn.MaxPool2d(kernel_size=k_pooling),
                          nn.ELU())
    return block

  def deconv_block(self, in_channels, out_channels, scale_factor=2, kernel_size=3, stride=1, padding=1):
    block = nn.Sequential(nn.Upsample(scale_factor=scale_factor),
                          nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding), 
                          nn.BatchNorm2d(out_channels), 
                          nn.ELU())
    return block