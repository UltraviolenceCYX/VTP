import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
import torch.nn.functional as F

from sinkhorn_transformer import SinkhornTransformer

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SingleDeconv3DBlock, self).__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(SingleConv3DBlock, self).__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Conv3DBlock, self).__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Deconv3DBlock, self).__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, 3),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)



class Embeddings(nn.Module):
    def  __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super(Embeddings, self).__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings




class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super(Transformer, self).__init__()
        ## flatten
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        self.attenmode = SinkhornTransformer(
            dim=embed_dim,
            heads=num_heads,
            depth=num_layers,
            bucket_size=120
        ).cuda()



    def forward(self, x):
        hidden_states = self.embeddings(x)
        extract_layers = self.attenmode(hidden_states)
        return extract_layers


class VTP(nn.Module):
    def __init__(self, cfg):
        super(VTP, self).__init__()
        self.input_dim = cfg.NETWORK.NUM_JOINTS
        self.output_dim = cfg.NETWORK.NUM_JOINTS
        self.embed_dim = cfg.TRANSFORMER.EMBEDDING_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE
        self.patch_size = cfg.TRANSFORMER.PATCH_SIZE
        self.num_heads = cfg.TRANSFORMER.NUM_HEADS
        self.num_trans_layers = 1
        self.dropout = cfg.TRANSFORMER.DROPOUT
        self.extract_layers = [1]
        self.num_extract_layers = len(self.extract_layers)
        self.patch_dim = [int(cube/self.patch_size) for cube in self.cube_size]
        # Transformer Encoder
        self.transformer = Transformer(self.input_dim, self.embed_dim, self.cube_size, self.patch_size,
                                       self.num_heads, self.num_trans_layers, self.dropout, self.extract_layers)
        self.decoder = nn.Sequential(
            Conv3DBlock(512, self.output_dim, 3),
            Conv3DBlock(self.output_dim, self.output_dim, 3)

        )
        self.encoder = nn.Sequential(
            Conv3DBlock(self.input_dim, 128, 3),
            Conv3DBlock(128, 256, 3)
        )

    def forward(self, x):
        t = self.transformer(x)
        t = t.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        c = self.encoder(x)
        res = self.decoder(torch.cat((c, t), dim=1))

        return res