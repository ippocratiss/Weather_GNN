import torch.nn.functional as F
from torch.nn import GRU,MultiheadAttention, LayerNorm, Linear, Sequential, ReLU,Conv1d, BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, NNConv
import torch 

# Set up GNN model 
class Weather_GNN(torch.nn.Module):
  def __init__(self,In,HGNN,Out,Ts):
    super(Weather_GNN,self).__init__()
    # Convolution Layers
    self.conv1 = Conv1d(In,HGNN*7,kernel_size=12,padding='same')
    self.conv2 = Conv1d(In,HGNN*7,kernel_size=6,padding='same')
    self.conv3 = Conv1d(In,HGNN*7,kernel_size=3,padding='same')
    self.conv4 = Conv1d(In,HGNN*7,kernel_size=1,padding=0)
    self.mlp1 = Linear(HGNN*28,HGNN)
    #linear layer to reduce channel size
    self.LN = LayerNorm(HGNN)
    #GNN layers
    self.gat1 = GATv2Conv(HGNN,HGNN//8,heads=8,concat=True,edge_dim=1,dropout=0.1)
    self.gat2 = GATv2Conv(HGNN,HGNN//8,heads=8,concat=True,edge_dim=1,dropout=0.1)
    self.gat3 = GATv2Conv(HGNN,HGNN//8,heads=8,concat=True,edge_dim=1,dropout = 0.1)
    #output layer 
    self.lin = Linear(HGNN,Out)

    self.time_step = Ts
    self.nfeat = In
    self.pred_window = Out // In 
  def forward(self, x, edge_index,edge_attr,):
    nNodes = x.size(0)
    x = x.view(nNodes,self.time_step,self.nfeat).permute(0,2,1)
    x1 = F.gelu(self.conv1(x))
    x2 = F.gelu(self.conv2(x))
    x3 = F.gelu(self.conv3(x))
    x4 = F.gelu(self.conv4(x))
    x1 = torch.max(x1,dim=2).values
    x2 = torch.max(x2,dim=2).values
    x3 = torch.max(x3,dim=2).values
    x4 = torch.max(x4,dim=2).values
    x = torch.cat([x1,x2,x3,x4],dim = 1)
    x = self.LN(F.gelu(self.mlp1(x)))
    
    
    x = x + F.dropout(F.gelu(self.gat1(x, edge_index, edge_attr)), p=0.12, training=self.training)
    x = x + F.dropout(F.gelu(self.gat2(x, edge_index, edge_attr)), p=0.12, training=self.training)
    x = x + F.dropout(F.gelu(self.gat3(x, edge_index, edge_attr)), p=0.12, training=self.training)

    x = self.lin(x) 
    x = x.view(nNodes, self.pred_window, self.nfeat) 
    return x