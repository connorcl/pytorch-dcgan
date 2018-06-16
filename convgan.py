import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.nn.functional as F
from unidecode import unidecode
from PIL import Image, ImageDraw, ImageFont

batch_size = 1

f = open('grimms3.txt')
text = unidecode(f.read())

all_characters = []
for c in text:
    if c not in all_characters:
        all_characters.append(c)
all_characters.sort()

char2index = {}
for i, c in enumerate(all_characters):
    char2index[c] = float(i)
    
index2char = {}
for i, c in enumerate(all_characters):
    index2char[float(i)] = c

def string2tensor(string):
    tensor = torch.Tensor(len(string))
    for i, c in enumerate(string):
        tensor[i] = char2index[c]
    tensor = tensor / 53
    return tensor.unsqueeze(0).unsqueeze(0)

def tensor2string(tensor):
    tensor = tensor.squeeze(0).squeeze(0) * 53
    s = []
    for i in tensor:
        s.append(index2char[round(i, 0)])
    return ''.join(s)

SEQUENCE_LENGTH = 380
STRIDE = 4

def sample_string(i, length, text):
    return text[i:i+length]

dataset = []

for i in range(len(text)-SEQUENCE_LENGTH+1):
    if i % STRIDE == 0:
        dataset.append(sample_string(i, SEQUENCE_LENGTH, text))
        
def random_sample(bs):
    return torch.randn(bs, 100, 1)

font = ImageFont.truetype('font.ttf', 14)

def make_img(string):
    img = Image.open('bg.jpg')
    img_draw = ImageDraw.Draw(img)
    img_draw.text((1, 1), string, font=font, fill="black")
    return img
        
        
class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(100, 1024, 8)
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.ConvTranspose1d(1024, 512, 8, 2, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.ConvTranspose1d(512, 256, 8, 2, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.ConvTranspose1d(256, 128, 8, 2, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.ConvTranspose1d(128, 64, 8, 2 ,1)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = nn.ConvTranspose1d(64, 1, 8, 2, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.sigmoid(self.conv6(x))
        return x
    

class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 8, 2, 1)
        self.conv2 = nn.Conv1d(32, 64, 8, 2, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 8, 2, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 8, 2, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, 8, 2, 1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 1, 8)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)
        x = F.sigmoid(self.conv6(x)).view(x.size(0), -1)
        return x
    
def weights_init(m):
    if type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d:
        m.weight.data.normal_(mean=0.0, std=0.02)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(mean=1.0, std=0.02)
        m.bias.data.fill_(0)
    
G = Generator()
G = G.apply(weights_init)
D = Discriminator()
D = D.apply(weights_init)
G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.99))
D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.99))
criterion = nn.BCELoss()

g_steps = 1
d_steps = 1
epochs = 50
for epoch in range(1, epochs+1):
    dataiter = iter(dataset)
    
    for batch in range(len(dataset)):
    
        d_running_real_loss = 0.0
        d_running_fake_loss = 0.0
        g_running_loss = 0.0
        # train d
        for _ in range(d_steps):
            D.zero_grad()
            # first on real data
            d_real_data = Variable(string2tensor(next(dataiter)))
            d_real_data_decision = D(d_real_data)
            d_real_data_loss = criterion(d_real_data_decision, Variable(torch.ones(batch_size, 1)))
            d_real_data_loss.backward()
            d_running_real_loss += d_real_data_loss.data[0]
            # then on fake data
            d_fake_data = G(Variable(random_sample(batch_size)))
            d_fake_data_decision = D(d_fake_data)
            d_fake_data_loss = criterion(d_fake_data_decision, Variable(torch.zeros(batch_size, 1)))
            d_fake_data_loss.backward()
            d_running_fake_loss += d_fake_data_loss.data[0]
            D_optimizer.step()
        
        # train g
        for _ in range(g_steps):
            G.zero_grad()
            g_fake_data = G(Variable(random_sample(batch_size)))
            g_d_fake_data_decision = D(g_fake_data)
            g_fake_data_loss = criterion(g_d_fake_data_decision, Variable(torch.ones(batch_size, 1)))
            g_fake_data_loss.backward()
            g_running_loss += g_fake_data_loss.data[0]
            G_optimizer.step()
    
        print("Epoch: %d, Batch: %d, D_R_loss: %f, D_F_loss: %f, G_Loss: %f" % (epoch, batch, d_running_real_loss/d_steps, d_running_fake_loss/d_steps, g_running_loss/g_steps))







        