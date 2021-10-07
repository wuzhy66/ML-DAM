from torch import nn
from nets.graph_encoder import GraphAttentionEncoder
import torch
import math

class CriticNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        encoder_normalization
    ):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim

        self.encoder = GraphAttentionEncoder(
            node_dim=input_dim,
            n_heads=8,
            embed_dim=embedding_dim,
            n_layers=n_layers,
            normalization=encoder_normalization
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """

        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """
        _, graph_embeddings = self.encoder(inputs)
        return self.value_head(graph_embeddings)
class ModifiedPointer_CriticNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        encoder_normalization
    ):
        super(ModifiedPointer_CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim

        # self.encoder = GraphAttentionEncoder(
        #     node_dim=input_dim,
        #     n_heads=8,
        #     embed_dim=embedding_dim,
        #     n_layers=n_layers,
        #     normalization=encoder_normalization
        # )
        """
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.init_hx = self.init_hidden(hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.encoder = nn.Conv1d(4, embedding_dim, kernel_size=1, stride=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        """
        self.encoder = nn.Conv1d(input_dim, embedding_dim, kernel_size=1, stride=1)
        self.decoder = nn.Sequential(
            nn.Conv1d(embedding_dim, 20, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(20, 20, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(20, 1, kernel_size=1, stride=1)
        )
    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        std = 1. / math.sqrt(hidden_dim)
        enc_init_hx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_hx.data.uniform_(-std, std)
        return enc_init_hx

    def forward(self, inputs):
        """

        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """
        batch_size = inputs.size(0)
        # print(inputs.size())
        embed_inputs = self.encoder(inputs.permute(0, 2, 1))
        # embed_inputs = embed_inputs.permute(0, 2, 1)
        # h0 = self.init_hx.unsqueeze(0).repeat(batch_size, 1).unsqueeze(0)
        # _, hx = self.rnn(embed_inputs, h0)
        # hx = hx.squeeze()
        # _, graph_embeddings = self.encoder(inputs)
        # return self.value_head(graph_embeddings)
        output = self.decoder(embed_inputs)
        output = output.sum(dim=2)
        return output.squeeze()