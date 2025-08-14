import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import tqdm
from util import load_data_from_txt
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

GPU_ID = 7
def calc_device(gpu_id: int):
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
DEVICE = calc_device(GPU_ID)
data_type = "torch"
DATA_PATH = '../../data/gset/gset_22.txt'
BATCH_SIZE = 2**5
data= load_data_from_txt(DATA_PATH)
src = data['adj_matrix']

NUM_NODES = src.shape[0]

def cal_obj(adj_matrix,x):
    delta_x = x * 2 - 1
    XA = torch.matmul(delta_x, adj_matrix)
    energy_x = -0.25 * (XA * delta_x).sum(dim=1)+0.25*torch.sum(adj_matrix)
    return energy_x
#input_dim:输入的解向量的维度

class Solver(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        prediction = self.sigmoid(prediction)

        prediction = (prediction - 0.5) * 0.999999 + 0.5
        return prediction, hidden, cell

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sample_from_prob(prob):
    prob=prob.detach()
    sample = 1-torch.bernoulli(prob)
    return sample

def get_return(value,prob,sample):
    sample = sample.detach()
    log_prob = torch.sum((sample * prob + (1 - sample) * (1 - prob)).log(),dim=1)
    objective = (log_prob * value.detach()).mean()
    return objective

def train_fn(model, src,sample, optimizer,sol):
    model.train()
    obj1 = cal_obj(src, sample)
    hidden = torch.zeros(1, BATCH_SIZE, 256, device=DEVICE)
    cell = torch.zeros(1, BATCH_SIZE, 256, device=DEVICE)
    for i in range(2000):
        optimizer.zero_grad()
        output,hidden,cell = model(sample,hidden.detach(), cell.detach())
        sample = sample_from_prob(output)
        obj = cal_obj(src,sample)
        if i ==1999:
            print(obj.max())
        obj = (torch.sum(src)/2+obj)/2
        value = obj-obj.mean()
        loss = get_return(value,output,sample)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loss.detach()
    return loss

sol=1



input_dim = NUM_NODES
output_dim = NUM_NODES

hidden_dim = 256
n_layers = 1
encoder_dropout = 0.5
decoder_dropout = 0.5
device = DEVICE
sample = torch.randint(0,2,(BATCH_SIZE,NUM_NODES),device=DEVICE,dtype=torch.float)
model = Solver(
    output_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
).to(device)

model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
n_epochs = 10000
clip = 1.0
best_valid_loss = float("inf")
teacher_forcing_ratio = 0.5
print(f"The model has {count_parameters(model):,} trainable parameters")


for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        src,
        sample,
        optimizer,
        sol
    )

    if (epoch %100) ==0 :
        print(f"\tTrain Loss: {train_loss:7.3f} ")
        # print(f"\tValid Loss: {valid_loss:7.3f}")

model.load_state_dict(torch.load("tut1-model.pt"))

