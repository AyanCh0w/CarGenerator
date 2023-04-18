print("START")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import gradio as gr
os.chdir('C:/Users/blubo/OneDrive/Documents/GitHub/CarGenerator/Final App')
print("IMPORT")
inputChannels = 3
NDF = 64 #num discriminator features
#DEFINE MODEL

def initilize_weights(layer):
  layer_name = layer.__class__.__name__
  if layer_name.find("Conv") != -1:
    nn.init.normal_(layer.weight.data, 0, 0.02)
  if layer_name.find("BatchNorm") != -1:
    nn.init.normal_(layer.weight.data, 1, 0.02)
    nn.init.constant_(layer.bias.data, 0)

noiseChannels = 100
NGF = 128
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.block1 = nn.Sequential(nn.ConvTranspose2d(noiseChannels, NGF*8, 4, 1, 0, bias = False), nn.BatchNorm2d(NGF*8), nn.ReLU())
    self.block2 = nn.Sequential(nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias = False), nn.BatchNorm2d(NGF*4), nn.ReLU(), nn.Conv2d(NGF*4, NGF*4, 3, 1, 1, bias = False), nn.BatchNorm2d(NGF*4), nn.ReLU())
    self.block3 = nn.Sequential(nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias = False), nn.BatchNorm2d(NGF*2), nn.ReLU(), nn.Conv2d(NGF*2, NGF*2, 3, 1, 1, bias = False), nn.BatchNorm2d(NGF*2), nn.ReLU())
    self.block4 = nn.Sequential(nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias = False), nn.BatchNorm2d(NGF), nn.ReLU(), nn.Conv2d(NGF, NGF, 3, 1, 1, bias = False), nn.BatchNorm2d(NGF), nn.ReLU())
    self.block5 = nn.Sequential(nn.ConvTranspose2d(NGF, inputChannels, 4, 2, 1, bias = False), nn.Tanh())

  def forward(self, x):
    x = self.block1(x)
    #print("GEN", x.shape)
    x = self.block2(x)
    #print("GEN", x.shape)
    x = self.block3(x)
    #print("GEN", x.shape)
    x = self.block4(x)
    #print("GEN", x.shape)
    x = self.block5(x)
    #print("-------------------------")
    return x
  
class SuperRes(nn.Module):
  def __init__(self):
    super(SuperRes, self).__init__()

    self.layer1 = nn.Sequential(
      nn.ConvTranspose2d(3, 16, 4, 2, 1),
      nn.LeakyReLU(0.2),
    )

    self.layer2 = nn.Sequential(
      nn.ConvTranspose2d(16, 32, 4, 2, 1),
      nn.LeakyReLU(0.2),
    )

    self.layer3 = nn.Sequential(
      nn.ConvTranspose2d(32, 64, 4, 2, 1),
      nn.LeakyReLU(0.2),
    )

    self.layer4 = nn.Sequential(
      nn.ConvTranspose2d(64, 3, 4, 2, 1),
      nn.Tanh()
    )

  def forward(self, x):
    output = self.layer1(x)
    output = self.layer2(output)
    output = self.layer3(output)
    output = self.layer4(output)
    return output
print("MODEL")
device = "cuda"
res = SuperRes().to(device)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(res.parameters())
print("DEVICE")
noiseChannels = 100
genModel = torch.load("./genSave.model")
resModel = torch.load("./superresSave.model")
genModel.eval()
resModel.eval()
print("LOAD")

def car(Batch):
  output = (genModel(torch.randn(1, noiseChannels, 1, 1, device = "cuda")).cpu().detach()[0].permute([1, 2, 0]) + 1) / 2
  output = output.permute(2, 0, 1)
  output = output.unsqueeze(0)
  output = output.to("cuda")
  output = resModel(output)
  final = output.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
  return final

demo = gr.Blocks()
with demo:
  gr.Markdown("""
    # Car Image Gen
  """
  )
  btn = gr.Button(value="Click For Image of Car")
  output = gr.Image()
  output.style(width=512, height=512)

  btn.click(fn=car, outputs=output)

demo.launch(share=True)