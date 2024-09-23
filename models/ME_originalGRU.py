import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
    ):
    # outputsize可以不用，把hidden_size加一个linear转成output_size 
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,  
            batch_first=True,  
        )
        self.fc = nn.Linear(hidden_size, output_size)  # output_size 为输出的维度

    def forward(self, input):
        output, _ = self.gru(input)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出

        return output


# 创建模型实例
input_size = 30
hidden_size = 64
num_layers = 2
output_size = 10  # 假设全连接层的输出维度为 10
model = GRUModel(
    input_size,
    hidden_size,
    num_layers,
    output_size,
)


class GRUGNNCell_xlstmV1(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, n_heads=3,
                 n_layers=1, dropout=0.1, gnn_layer="graphconv", edge_dim=None):
        super().__init__()

    def reset_parameters(self):

    def forward(self, x, edge_index, h=None, edge_attr=None):