import torch.nn.functional as F
from torch.nn import GRU,MultiheadAttention, LayerNorm, Linear, Sequential, ReLU,Conv1d, BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, NNConv
import torch 

# Set up GNN model
class Weather_GNN(torch.nn.Module):
  def __init__(self,In,HGNN,Out,Ts):
    super(Weather_GNN,self).__init__()
    # Convolution Layers
    self.conv1d = Sequential(Conv1d(In,HGNN*4,kernel_size=24,padding=12),
                             BatchNorm1d(HGNN*4),ReLU(),
                         Conv1d(HGNN*4,HGNN,kernel_size=8 ,padding=7),
                         ReLU(),BatchNorm1d(HGNN),Conv1d(HGNN,HGNN,kernel_size=4 ,padding=3),
                         BatchNorm1d(HGNN))
    self.gat1 = GATv2Conv(HGNN,HGNN*4,heads=2,concat=True,edge_dim=1)
    self.gat2 = GATv2Conv(HGNN*8,HGNN*8,heads=2,concat=True,edge_dim=1)
    self.gat3 = GATv2Conv(HGNN*16,HGNN*4,heads=1,concat=True,edge_dim=1)
    self.lin = Sequential(Linear(HGNN*4,HGNN*2),ReLU(),Linear(HGNN*2,Out))
    self.time_step = Ts
    # output
  def forward(self, x, edge_index,edge_attr):
    nNodes = x.size(0)
    x = x.view(nNodes,self.time_step,-1)
    x = x.permute(0,2,1)
    x = self.conv1d(x)
    x = torch.max(x,dim=2).values
    x = F.relu(x)
    x = F.dropout(x,p=0.1,training=self.training)
    x = self.gat1(x,edge_index,edge_attr)
    x = F.relu(x)
    x = F.dropout(x,p=0.1,training=self.training)
    x = self.gat2(x, edge_index,edge_attr)
    x = F.relu(x)
    x = F.dropout(x,p=0.1,training=self.training)
    x = self.gat3(x,edge_index,edge_attr)
    x = F.relu(x)
    x = self.lin(x) 
    x = x.view(nNodes,12,2)
    return x