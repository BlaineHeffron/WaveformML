import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv, knn_graph

from src.models.BasicNetwork import *

#see https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
#try https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(x, edge_index)


class GraphNet(BasicNetwork):
    def __init__(self, config):
        super(GraphNet, self).__init__(config)
        #self.conv1 = GCNConv(dataset.num_node_features, 16)
        #self.conv2 = GCNConv(16, dataset.num_classes)
        self.conv1 = DynamicEdgeConv(self.config.system_config.n_samples, 16)
        self.conv2 = DynamicEdgeConv(16, self.config.ntype)

    def forward(self, data):
        x, edge_index = data[1], data[0]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphNet().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))