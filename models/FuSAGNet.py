from random import uniform

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.time import *
from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i * edge_num : (i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_size, out_size, layer_num):
        super(OutLayer, self).__init__()
        modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_size if layer_num == 1 else out_size, 1))
            else:
                layer_in_num = in_size if i == 0 else out_size
                modules.append(nn.Linear(layer_in_num, out_size))
                modules.append(nn.BatchNorm1d(out_size))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for module in self.mlp:
            if isinstance(module, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = module(out)
                out = out.permute(0, 2, 1)
            else:
                out = module(out)

        return out


class SparseEncoder(nn.Module):
    def __init__(
        self, in_size, latent_size, num_layers, use_bn=False, use_dropout=False
    ):
        super().__init__()
        self.in_size = in_size
        self.latent_size = latent_size
        layer_sizes = [in_size] + [
            in_size - (i + 1) * (in_size - latent_size) // num_layers
            for i in range(num_layers)
        ]
        self.encoder = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )

        self.use_bn = use_bn
        self.use_dropout = use_dropout
        if use_bn:
            self.bns = nn.ModuleList(
                [
                    nn.BatchNorm1d(self.encoder[i].out_features)
                    for i in range(len(self.encoder))
                ]
            )

    def forward(self, x):
        B = x.size(0)
        if x.dim() > 2:
            x = x.view(B, -1)

        feature_maps = []
        for i, module in enumerate(self.encoder):
            x = module(x)
            if self.use_bn:
                x = self.bns[i](x)

            x = torch.sigmoid(x)
            feature_maps.append(x)

        return x, feature_maps


class SparseDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        out_size,
        n_features,
        num_layers,
        use_bn=False,
        use_dropout=False,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.out_size = out_size
        layer_sizes = [out_size] + [
            out_size - (i + 1) * (out_size - latent_size) // num_layers
            for i in range(num_layers)
        ]
        layer_sizes = layer_sizes[::-1]
        self.decoder = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )

        self.use_bn = use_bn
        self.use_dropout = use_dropout
        if use_bn:
            self.bns = nn.ModuleList(
                [
                    nn.BatchNorm1d(self.decoder[i].out_features)
                    for i in range(len(self.decoder))
                ]
            )

    def forward(self, x):
        feature_maps = []
        for i, module in enumerate(self.decoder):
            x = module(x)
            if self.use_bn:
                x = self.bns[i](x)

            x = torch.sigmoid(x)
            feature_maps.append(x)

        return x, feature_maps


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, heads=1):
        super(GNNLayer, self).__init__()
        self.gnn = GraphLayer(
            in_channels=in_channel, out_channels=out_channel, heads=heads, concat=False
        )
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, embedding=None):
        out, (_, _) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        out = self.bn(out)
        return self.relu(out)


class FuSAGNet(nn.Module):
    def __init__(
        self,
        edge_index_sets,
        node_num,
        dim=16,
        out_layer_inter_dim=16,
        window_size=16,
        out_layer_num=1,
        topk=15,
        latent_size=16,
        n_layers=2,
        process_dict=None,
    ):
        super(FuSAGNet, self).__init__()
        self.edge_index_sets = edge_index_sets
        self.embed_dim = dim
        self.node_num = node_num
        sensor_f = 0
        embedding_modules = []
        for process in process_dict:
            sensor_i = sensor_f
            n_processes = process_dict.get(process)
            sensor_f += n_processes
            embedding_modules.append(nn.Embedding(sensor_f - sensor_i, self.embed_dim))

        self.embeddings = nn.ModuleList(embedding_modules)

        n_rnn_layers = 3
        self.rnn_embedding_modules = nn.ModuleList(
            [
                nn.GRU(
                    self.embed_dim,
                    self.embed_dim // 2,
                    bidirectional=True,
                    num_layers=n_rnn_layers,
                    dropout=0.2,
                )
                for _ in range(len(process_dict))
            ]
        )
        self.bn_outlayer_in = nn.BatchNorm1d(self.embed_dim)

        edge_set_num = len(edge_index_sets)

        self.topk = topk
        self.learned_graph = None

        self.latent_size = window_size
        num_layers = n_layers
        self.encoder = SparseEncoder(
            in_size=node_num * window_size,
            latent_size=node_num * self.latent_size,
            num_layers=num_layers,
            use_bn=True,
            use_dropout=True,
        )
        self.decoder = SparseDecoder(
            latent_size=node_num * self.latent_size,
            out_size=node_num * window_size,
            n_features=node_num,
            num_layers=num_layers,
            use_bn=True,
            use_dropout=True,
        )
        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(in_channel=self.latent_size, out_channel=dim, heads=1)
                for _ in range(edge_set_num)
            ]
        )
        self.out_layer = OutLayer(
            in_size=dim * edge_set_num,
            out_size=out_layer_inter_dim,
            layer_num=out_layer_num,
        )

        self.dp = nn.Dropout(0.2)
        self.init_params(init_method="kaiming_uniform")

    def init_params(self, init_method):
        for embedding in self.embeddings:
            if init_method == "uniform":
                nn.init.uniform_(embedding.weight, a=0.0, b=1.0)
            elif init_method == "kaiming_uniform":
                nn.init.kaiming_uniform_(embedding.weight, a=0.0)
            elif init_method == "xavier_uniform":
                nn.init.xavier_uniform_(embedding.weight, gain=1.0)
            elif init_method == "normal":
                nn.init.normal_(embedding.weight, mean=0.0, std=1.0)
            elif init_method == "kaiming_normal":
                nn.init.kaiming_normal_(embedding.weight, a=0.0)
            elif init_method == "xavier_normal":
                nn.init.xavier_normal_(embedding.weight, gain=1.0)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, data, target, org_edge_index):
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets
        device = data.device
        batch_num, node_num, _ = x.shape
        mu, log_var = None, None

        z, enc_feature_maps = self.encoder(x)
        z_out = z.view(-1, self.node_num, self.latent_size).clone().detach()
        x_recon, dec_feature_maps = self.decoder(z)
        x_recon = x_recon.view(x.size())
        rhos = (
            torch.FloatTensor([uniform(1e-5, 1e-2) for _ in range(z.size(1))])
            .unsqueeze(0)
            .to(device)
        )
        rho_hat = torch.sum(z, dim=0, keepdim=True)
        enc_fmaps, dec_fmaps = (
            enc_feature_maps[:-1],
            dec_feature_maps[:-1][::-1],
        )
        z = z.view(-1, self.latent_size).contiguous()

        gcn_outs = []
        for i, _ in enumerate(edge_index_sets):
            embedded_sensors = []
            y_process = []
            for j, embedding in enumerate(self.embeddings):
                sensors = torch.arange(embedding.num_embeddings).to(device)
                embedded = embedding(sensors)
                embedded = embedded.unsqueeze(0)
                embedded, _ = self.rnn_embedding_modules[j](embedded)
                embedded = embedded.squeeze()
                embedded_sensors.append(embedded)
                y_process.extend(batch_num * [j for _ in range(embedded.size(0))])

            y_process = torch.tensor(y_process).to(device)
            all_embeddings = torch.cat(embedded_sensors)
            embeds = all_embeddings.view(node_num, -1)
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            cos_ji_mat = torch.matmul(embeds, embeds.T)
            cos_ji_mat /= torch.matmul(
                embeds.norm(dim=-1).view(-1, 1), embeds.norm(dim=-1).view(1, -1)
            )
            topk_num = self.topk
            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            gated_i = (
                torch.arange(0, node_num)
                .unsqueeze(1)
                .repeat(1, topk_num)
                .flatten()
                .to(device)
                .unsqueeze(0)
            )
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            batch_gated_edge_index = get_batch_edge_index(
                gated_edge_index, batch_num, node_num
            ).to(device)

            gcn_out = self.gnn_layers[i](
                x=z, edge_index=batch_gated_edge_index, embedding=all_embeddings
            )
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        out = torch.mul(x, embeds)
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)
        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
        x_frcst = out
        return (
            x_frcst,
            x_recon,
            z_out,
            mu,
            log_var,
            enc_fmaps,
            dec_fmaps,
            all_embeddings,
            y_process,
            rhos,
            rho_hat,
        )
