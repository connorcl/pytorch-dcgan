import random
from unidecode import unidecode
from gensim.models import Word2Vec
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch import optim

f = open("grimms.txt")
text = unidecode(f.read()).lower()
sentences = text.split('.')
for i in range(len(sentences)):
    sentences[i] = list(sentences[i])
text = text.replace('.', '')
    
model = Word2Vec(sentences=sentences, size=32, window=8, min_count=1, iter=500)

SEQ_LENGTH = 64

def sample_text():
    i = random.randint(0, len(text)-(SEQ_LENGTH))
    return text[i:i+SEQ_LENGTH]

def string2tensor(string):
    V = []
    for c in string:
        V.append(torch.from_numpy(model.wv[c]).unsqueeze(0).unsqueeze(-1))
    return torch.cat(V, dim=2)

def tensor2string(tensor):
    tensor = tensor.squeeze(0)
    vecs = [c.squeeze(1).numpy() for c in torch.chunk(tensor, SEQ_LENGTH, dim=-1)]
    string = ''.join([model.wv.similar_by_vector(vec)[0][0] for vec in vecs])
    return string
    
def random_sample(bs):
    return torch.randn(bs, 100, 1)

def weights_init(m):
    if type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(100, 512, 4)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.ConvTranspose1d(512, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.ConvTranspose1d(256, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.ConvTranspose1d(128, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.ConvTranspose1d(64, 32, 4, 2, 1)
        
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
        self.conv1 = nn.Conv1d(32, 64, 4, 2, 1)
        self.conv2 = nn.Conv1d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1, 4)
        
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

real_target = Variable(torch.ones(1, 1))
fake_target = Variable(torch.zeros(1, 1))

for epoch in range(1, epochs+1):
    
    for batch in range(int(len(text)/64)):
        
        drl = 0.0
        dfl = 0.0
        gl = 0.0
        
        for _ in range(d_steps):
            D.zero_grad()
            
            d_real_data = Variable(string2tensor(sample_text()))
            d_real_data_decision = D(d_real_data)
            d_real_data_loss = criterion(d_real_data_decision, real_target)
            d_real_data_loss.backward()
            drl += d_real_data_loss.data[0]
            
            d_fake_data = G(Variable(random_sample(1)))
            d_fake_data_decision = D(d_fake_data)
            d_fake_data_loss = criterion(d_fake_data_decision, fake_target)
            d_fake_data_loss.backward()
            dfl += d_fake_data_loss.data[0]
            
            D_optim.step()
            
        for _ in range(g_steps):
            G.zero_grad()
            
            g_data = G(Variable(random_sample(1)))
            g_data_decision = D(g_data)
            g_data_loss = criterion(g_data_decision, real_target)
            g_data_loss.backward()
            gl += g_data_loss.data[0]
            
            G_optim.step()
            
        print("Epoch: %d, Batch: %d, D_R_loss: %f, D_F_loss: %f, G_Loss: %f" 
              % (epoch, batch, drl/d_steps, dfl/d_steps, gl/g_steps))