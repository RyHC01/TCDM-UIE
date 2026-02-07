import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
# Diffusion Color Restoration Block (DCRB)
class DCRB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DCRB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
# Diffusion Partial-Convolution Feed-Forward Network (DPFN)
class DPFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DPFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        p_dim = int(hidden_features * 0.25)
        self.conv_0 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_2 = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.act = nn.GELU()
        self.p_dim = p_dim
        self.hidden_features = hidden_features

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x1, x2 = torch.split(x, [self.p_dim, self.hidden_features - self.p_dim], dim=1)
        x1 = self.conv_1(x1)
        x1 = self.act(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_2(x)
        return x


##########################################################################
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[:, np.newaxis, :]
    return torch.tensor(pos_encoding, dtype=torch.float32)


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x * torch.sigmoid(x)


class DTBs(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, da=False):
        super(DTBs, self).__init__()

        # DA
        if da:
            self.adap_pool = nn.AdaptiveAvgPool2d(1)
            self.pro_extractor = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(dim // 4, 10)
            )
            self.d_adap = nn.Embedding(10, dim)

        else:
            self.adap_pool = None
            self.d_adap = None
            self.pro_extractor = None

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DCRB(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DPFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, t):
        if len(t.size()) == 1:
            t = nonlinearity(get_timestep_embedding(t, x.size(1)))[:, :, None, None]

        if self.d_adap is not None:
            # B, C, H, W = x.shape
            x_adap = self.adap_pool(x).squeeze(-1).squeeze(-1)  # E x C
            prototype = torch.argmax(F.softmax(self.pro_extractor(x_adap), dim=1), dim=1)  # E
            pro_embed = self.d_adap(prototype)  # E x C

            x = x + pro_embed[..., None, None]

        x = x + self.attn(self.norm1(x) + t)
        x = x + self.ffn(self.norm2(x) + t)

        return x, t


##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


def concat_with_padding(skip_tensor, upsampled_tensor):
    _, _, H1, W1 = skip_tensor.shape
    _, _, H2, W2 = upsampled_tensor.shape
    dh, dw = H2 - H1, W2 - W1
    assert dh >= 0 and dw >= 0, f"skip_tensor must not be larger: ({H1},{W1}) vs ({H2},{W2})"
    pad = [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
    return torch.cat([upsampled_tensor, F.pad(skip_tensor, pad)], dim=1)


class Sequentials(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


##########################################################################
# Diffusion Transformer (DT)
class DT(nn.Module):
    def __init__(self,
                 inp_channels=6,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 4, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False
                 ):

        super(DT, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = Sequentials(*[
            DTBs(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                 LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = Sequentials(*[
            DTBs(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                 bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = Sequentials(*[
            DTBs(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                 bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = Sequentials(*[
            DTBs(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                 bias=bias, LayerNorm_type=LayerNorm_type, da=(i >= num_blocks[3] // 2)) for i in
            range(num_blocks[3])])  # da

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = Sequentials(*[
            DTBs(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                 bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = Sequentials(*[
            DTBs(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                 bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = Sequentials(*[
            DTBs(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                 bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, t):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1, _ = self.encoder_level1(inp_enc_level1, t)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, _ = self.encoder_level2(inp_enc_level2, t)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, _ = self.encoder_level3(inp_enc_level3, t)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, _ = self.latent(inp_enc_level4, t)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = concat_with_padding(inp_dec_level3, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _ = self.decoder_level3(inp_dec_level3, t)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = concat_with_padding(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _ = self.decoder_level2(inp_dec_level2, t)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = concat_with_padding(inp_dec_level1, out_enc_level1)
        out_dec_level1, _ = self.decoder_level1(inp_dec_level1, t)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:, -3:, :, :]

        return out_dec_level1
