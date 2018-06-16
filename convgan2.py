import random
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

seed = random.randint(0, 10000)
torch.manual_seed(seed)
torch.set_num_threads(8)

bs = 16

#training_set_transform = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#training_set = datasets.CIFAR10(root='CIFAR10_data', transform=training_set_transform,
#                              download=True)
#training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True)
training_set_transform = transforms.Compose([
                            transforms.Resize(70),
                            transforms.CenterCrop(64),
                            transforms.ToTensor()])
training_set = datasets.ImageFolder(root='data', 
                                    transform=training_set_transform)
training_set_loader = torch.utils.data.DataLoader(training_set, 
                                                  batch_size=bs, 
                                                  shuffle=True)

def show_images(batch):
  images = torchvision.utils.make_grid(batch, normalize=True)
  images = np.transpose(images.numpy(), (1, 2, 0))
  plt.imshow(images)

def random_sample(bs):
    return torch.randn(bs, 100, 1, 1)

def weights_init(m):
    if type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(100, 512, 4)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.tanh(self.conv5(x))
        return x
    

class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, 4)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = F.sigmoid(self.conv5(x)).view(x.size(0), -1)
        return x
    
G = Generator()
D = Discriminator()
G.apply(weights_init)
D.apply(weights_init)
criterion = nn.BCELoss()
G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

d_steps = 1
g_steps = 1
epochs = 100

real_target = Variable(torch.ones(bs, 1))
fake_target = Variable(torch.zeros(bs, 1))

for epoch in range(1, epochs+1):
    
    dataiter = iter(training_set_loader)
    for batch in range(int(len(training_set_loader)/d_steps)-1):
        
        drl = 0.0
        dfl = 0.0
        gl = 0.0
        
        for _ in range(d_steps):
            D.zero_grad()
            
            d_real_data = Variable(next(dataiter)[0])
            d_real_data_decision = D(d_real_data)
            d_real_data_loss = criterion(d_real_data_decision, real_target)
            d_real_data_loss.backward()
            drl += d_real_data_loss.data[0]
            
            d_fake_data = G(Variable(random_sample(bs)))
            d_fake_data_decision = D(d_fake_data)
            d_fake_data_loss = criterion(d_fake_data_decision, fake_target)
            d_fake_data_loss.backward()
            dfl += d_fake_data_loss.data[0]
            
            D_optim.step()
            
        for _ in range(g_steps):
            G.zero_grad()
            
            g_data = G(Variable(random_sample(bs)))
            g_data_decision = D(g_data)
            g_data_loss = criterion(g_data_decision, real_target)
            g_data_loss.backward()
            gl += g_data_loss.data[0]
            
            G_optim.step()
            
        print("Epoch: %d, Batch: %d, D_R_loss: %f, D_F_loss: %f, G_Loss: %f" 
              % (epoch, batch, drl/d_steps, dfl/d_steps, gl/g_steps))