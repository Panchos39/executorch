import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SelfAttentiveEncoder(nn.Module):

    def __init__(self):
        super(SelfAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.ws1 = nn.Linear(256, 20, bias=False)
        self.ws2 = nn.Linear(20, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
#        self.init_weights()
        self.attention_hops = 1

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, outp):
        size = outp.size()  # [bsz, len, nhid] 50,25,128
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp) #bsz,hop,nhid

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)

class A_LSTM(nn.Module):
    def __init__(self, sequence_size=12):
        super(A_LSTM, self).__init__()
        # input size?
#         self.lnlstm = LNLSTM(30,64,2)#input,output,layer num_layers=1
        self.gru = nn.GRU(input_size=39, hidden_size=128,num_layers=2,bidirectional=True)
        #self.bn1 = nn.BatchNorm1d(24)
        self.bn1 = nn.BatchNorm1d(sequence_size)
        #self.selfattention = SelfAttention(128)
        # fix attention output size to 25
        self.selfattention = SelfAttentiveEncoder()
        self.fc1 = nn.Linear(256, 320)
        self.bn2 = nn.BatchNorm1d(320)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(320, 320)
        self.bn3 = nn.BatchNorm1d(320)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(320, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(39)
        
    def forward(self, x):#input should be length,bsz,30
        length,bsz,feature = x.shape
        x = x.contiguous().view(-1,feature)
        x = self.bn5(x)
        x = x.contiguous().view(length,bsz,feature)
        x,hidden = self.gru(x)
        #print("hidden", hidden.size())
        #output 25,bsz,128
        x = x.permute(0,2,1)#25,128,bsz
        #  x = self.maxpooling(x)#25,128,bsz/2
        #print("pooling",x.shape)
        x = x.permute(2,0,1)#bsz/2,25,128
        #print("pooling",x.shape)
        x = self.bn1(x)
        #print("bn1",x)
        # x = self.selfattention(x).squeeze()
        x = self.selfattention(x).squeeze(1)
        #print("attention",x.size())
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn4(x)
        #print("embedding", x.size())
        #l2_norm = torch.norm(x,p=2,dim=1)#bsz
        #print("l2_norm", l2_norm.size())
        #return l2_norm
        return x
    
    def gen_embedding(self, x) :
        length,bsz,feature = x.shape
        x = x.contiguous().view(-1,feature)
        x = self.bn5(x)
        x = x.contiguous().view(length,bsz,feature)
        x,hidden = self.gru(x)
        #print("hidden", hidden.size())
        #output 25,bsz,128
        x = x.permute(0,2,1)#25,128,bsz
        #  x = self.maxpooling(x)#25,128,bsz/2
        #print("pooling",x.shape)
        x = x.permute(2,0,1)#bsz/2,25,128
        #print("pooling",x.shape)
        x = self.bn1(x)
        #print("bn1",x)
        # x = self.selfattention(x).squeeze()
        x = self.selfattention(x).squeeze(1)
        #print("attention",x.size())
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn4(x)
        return x
        

class MMD_NCA_Net(nn.Module):
    def __init__(self, sequence_size):
        super(MMD_NCA_Net, self).__init__()
        self.A_LSTM = A_LSTM(sequence_size)
        self.seq_emb_size = sequence_size

    def forward(self, x):
        return self.A_LSTM(x)
    
    def load_weights(self, weights_path, device) :
        self.load_state_dict((torch.load(weights_path)))
        self.to(device)
        self.eval()
        self.device = device
    
    def gen_embedding(self, x) :
        tensor = Variable(torch.Tensor(x)).float()\
                .to(self.device).squeeze()\
                .view(-1,self.seq_emb_size,39).permute(1,0,2)
        return self.A_LSTM.gen_embedding(tensor)

