import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import sys
# classes

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        #print('trigger',x.shape)
        #sys.exit()
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        #print('out shape 1 {}',out.shape)
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)
        #print('out shape 2 {}', out.shape)
        # concat back the cls token
        out = torch.cat((cls_out, out), dim = 1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        #print('out shape 3 {}', out.shape)
        # combine heads out
        return self.to_out(out)

# main classes

class TimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        image_size = 224,
        patch_size = 16,
        channels = 3,
        depth = 12,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_emb = nn.Embedding(num_positions + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))
        self.relinear = nn.Linear(dim, patch_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout),)

            ]))

        self.to_out = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p) * (w // p)

        #print(video.shape)
        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)
        #print(video.shape)
        tokens = self.to_patch_embedding(video)
        #print('token shape {}',tokens.shape)
        cls_token = repeat(self.cls_token, 'n d -> b n d', b = b)
        #print('cls_token shape {}', cls_token.shape)
        x =  torch.cat((cls_token, tokens), dim = 1)
        x += self.pos_emb(torch.arange(x.shape[1], device = device))
        #print('x.shape 1{}',x.shape)
        for (time_attn, spatial_attn, ff) in self.layers:
            #x = time_attn(x, 'b (f n) d', '(b n) f d', n = n) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f) + x
            x = ff(x) + x
        # for (time_attn, spatial_attn, ff,ff) in self.layers:
        #     x_t = time_attn(x, 'b (f n) d', '(b n) f d', n = n)
        #     x_s = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f)
        #     x_s = ff(x_s)
        #     x_t = ff(x_t)
        #     x = x_t + x_s + x
        #print('x.shape 2{}', x.shape)
        #cls_token = x[:, 0]
        h=int(h/p)
        w=int(w/p)

        x = self.relinear(x[:, 1:, :])
        x = rearrange(x, ' b (f h w) (p1 p2 c)-> b f c (h p1) (w p2)', f=f, h=h, w=w, p1=p, p2=p)
        #print(cls_token.shape)
        #return self.to_out(cls_token)
        return x


