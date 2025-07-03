import torch.nn.functional as F
from torch.nn import GRU
from torch_geometric.nn import GCNConv, GATConv
import torch 

# Set up GNN model
class Weather_GNN(torch.nn.Module):
  def __init__(self,In,HGRU,HGNN,Out,Ts):
    super(Weather_GNN,self).__init__()
    # Convolution Layers
    self.gru = GRU(In,HGRU,num_layers=2,batch_first=True)
    self.conv1 = GATConv(HGRU,HGNN)
    self.conv2 = GATConv(HGNN,HGNN)
    self.time_step = Ts
    # output
    self.lin = torch.nn.Linear(HGNN, Out)
  def forward(self, x, edge_index):
    nNodes = x.size(0)
    x = x.view(nNodes, self.time_step,-1)
    _,HS = self.gru(x)
    x = HS[0]
    #print(x.shape)
    x = self.conv1(x, edge_index)
    x = F.leaky_relu(x,negative_slope=0.1)
    x = self.conv2(x, edge_index)
    x = F.leaky_relu(x,negative_slope=0.1)
    #x = F.dropout(x,p=0.1,training=self.training)
    x = self.lin(x)
    return x