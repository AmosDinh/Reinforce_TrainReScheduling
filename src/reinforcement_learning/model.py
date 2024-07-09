import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool

class DuelingQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(DuelingQNetwork, self).__init__()

        # value network
        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc4_val = nn.Linear(hidsize2, 1)

        # advantage network
        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc4_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc4_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidsize1, hidsize2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidsize1)
        self.fc2 = nn.Linear(hidsize1, hidsize2)
        self.fc3 = nn.Linear(hidsize2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class QGINWithPooling(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_layers=2):
        super(QGINWithPooling, self).__init__()
        
        self.convs = nn.ModuleList()
        
        mlp = Sequential(
            Linear(num_features, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp, train_eps=False))
        
        for _ in range(num_layers - 1):
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=False))
        
        self.out = Linear(hidden_dim, num_classes)
        self.att_train_k = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.att_target_k = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.att_train_k)
        nn.init.xavier_uniform_(self.att_target_k)
        self.att_train_q = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.att_target_q = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.att_train_q)
        nn.init.xavier_uniform_(self.att_target_q)

    def forward(self, state):
        x, edge_index, train_index, target_index = state[0].x, state[0].edge_index, state[1], state[2]
        # Apply GIN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # Global pooling
        # attention to the train location and the target location
        if train_index == -1:
            x_train = torch.zeros(1, x.shape[1])
        else:
            x_train = x[train_index,:].unsqueeze(0)

        if target_index == -1: # if not on map
            x_target = torch.zeros(1, x.shape[1])
        else: 
            x_target = x[target_index,:].unsqueeze(0)

        q_train = torch.matmul(x_train, self.att_train_q)
        q_target = torch.matmul(x_target, self.att_target_q)

        k_train = torch.matmul(x, self.att_train_k)
        k_target = torch.matmul(x, self.att_target_k)
        train = torch.matmul(q_train, k_train.T)
        target = torch.matmul(q_target, k_target.T)

        # softmax
        train = F.softmax(train, dim=0)
        target = F.softmax(target, dim=0)
        x1 = torch.matmul(train, x)  # like global sum pooling but with attention weights
        x2 = torch.matmul(target, x)
        x = x1 + x2
        # batch = torch.zeros(x.shape[0], 1)  # all in the same batch
        # x = global_add_pool(x, batch)
        
        # Apply final linear layer
        x = self.out(x)
        
        return x