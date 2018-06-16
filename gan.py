import random
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.nn.functional as F
from unidecode import unidecode

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
    
def char2tensor(c):
    t = torch.zeros(len(all_characters))
    i = all_characters.index(c)
    t[i] = 1
    return t
    
def tensor2char(t):
    i = t.max(0)[1][0]
    return all_characters[i]
    
def string2tensor(string):
    return torch.cat([char2tensor(c) for c in string])

def tensor2string(tensor):
    s = [tensor2char(t) for t in torch.chunk(tensor, int(tensor.size()[0]/len(all_characters)))]
    return ''.join(s)

SEQUENCE_LENGTH = 100
STRIDE = 4

def random_sample():
    return torch.rand(256)

def sample_string(i, length, text):
    return text[i:i+length]

dataset = []

for i in range(len(text)-SEQUENCE_LENGTH+1):
    if i % STRIDE == 0:
        dataset.append(string2tensor(sample_string(i, SEQUENCE_LENGTH, text)))
        



class Generator(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
        

class Discriminator(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


g = Generator(256, 1024, SEQUENCE_LENGTH*len(all_characters))
d = Discriminator(SEQUENCE_LENGTH*len(all_characters), 8, 2)
criterion = nn.CrossEntropyLoss()
g_optimizer = optim.Adam(g.parameters())
d_optimizer = optim.Adam(d.parameters())
    
epochs = 10000
g_steps = 4
d_steps = 1
for epoch in range(1, epochs+1):
    
    d_running_real_loss = 0.0
    d_running_fake_loss = 0.0
    g_running_loss = 0.0
    for step in range(d_steps):
        d.zero_grad()
        
        d_real_data = Variable(random.choice(dataset)).unsqueeze(0)
        d_real_decision = d(d_real_data)
        d_real_loss = criterion(d_real_decision, Variable(torch.LongTensor([0])))
        d_real_loss.backward()
        d_running_real_loss += d_real_loss
        
        d_fake_data = g(Variable(random_sample()).unsqueeze(0))
        d_fake_decision = d(d_fake_data)
        d_fake_loss = criterion(d_fake_decision, Variable(torch.LongTensor([1])))
        d_fake_loss.backward()
        d_running_fake_loss += d_fake_loss
        
        d_optimizer.step()
    for step in range(g_steps):
        g.zero_grad()
        g_fake_data = g(Variable(random_sample()).unsqueeze(0))
        dg_fake_decision = d(g_fake_data)
        g_loss = criterion(dg_fake_decision, Variable(torch.LongTensor([0])))
        g_loss.backward()
        g_running_loss += g_loss
        g_optimizer.step()
    print("Epoch: %d, D_R_loss: %f, D_F_loss: %f, G_Loss: %f" % (epoch, d_running_real_loss/d_steps, d_running_fake_loss/d_steps, g_running_loss/g_steps))
        