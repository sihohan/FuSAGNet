import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class GraphLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
        **kwargs
    ):
        super(GraphLayer, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and (not concat):
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        # Add self-loops to the adjacency matrix
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))

        # Linearly transform node feature matrix
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        # Start propagating messages
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            embedding=embedding,
            edges=edge_index,
            return_attention_weights=return_attention_weights,
        )
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(
        self, x_i, x_j, edge_index_i, size_i, embedding, edges, return_attention_weights
    ):
        x_i = x_i.view(-1, self.heads, self.out_channels)  # Target node
        x_j = x_j.view(-1, self.heads, self.out_channels)  # Source node

        if embedding is not None:
            embedding_i, embedding_j = (
                embedding[edge_index_i],  # edge_index_i = edges[1], i.e., parent nodes
                embedding[edges[0]],  # edges[0], i.e., child nodes
            )
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            # Eq. 6 in Deng and Hooi (2021)
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        # Eq. 7 in Deng and Hooi (2021)
        attended_key_i = torch.einsum("nhd,mhd->nhd", key_i, cat_att_i)
        attended_key_j = torch.einsum("nhd,mhd->nhd", key_j, cat_att_j)
        alpha = attended_key_i.sum(dim=-1) + attended_key_j.sum(dim=-1)
        alpha = alpha.view(-1, self.heads, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Eq. 8 in Deng and Hooi (2021)
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if return_attention_weights:
            self.__alpha__ = alpha

        return torch.einsum("nhc,nhd->nhd", alpha, x_j)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
