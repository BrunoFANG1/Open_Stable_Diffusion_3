import torch
import torch.nn as nn
import einops
import math

def modulate(norm_func, x, scale, shift):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)
    return x

class RMSNorm(nn.Module):
    def __init__(self, d, length, epsilon=1e-8):
        super().__init__()
        self.norm_dim = d
        self.dim_len = length
        self.epsilon = epsilon
        self.g = nn.Parameter(torch.zeros(self.dim_len))

    def forward(self, x):

        RMS = x.norm(2, dim=-1, keepdim = True)* (self.dim_len)**(-0.5)
        x = x/(RMS + self.epsilon) * self.g
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)
        return x

class MultiHeadAttention(nn.Module): #### Consider add FlashAttention
    def __init__(
            self,
            emb_size: int,
            num_heads: int=8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: nn.Module = nn.LayerNorm,
            enable_flashattn: bool = False
    ):
        super().__init__()
        assert emb_size % num_heads == 0, "embedding size must be divisible by num of heads"
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn

        self.qkv = nn.Linear(emb_size, emb_size*3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(emb_size, emb_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, Q, K, V):
        # Q, K, V = self.qkv(x).chunk(3, dim=-1)
        Q = self.q_norm(einops.rearrange(Q, 'b p (h e) -> b h p e', h = self.num_heads))
        K = self.k_norm(einops.rearrange(K, 'b p (h e) -> b h p e', h = self.num_heads))
        V = einops.rearrange(V, 'b p (h e) -> b h p e', h = self.num_heads)

        att = torch.einsum('bhij, bhkl -> bhik', Q, K)*self.scale
        dtype = Q.dtype
        att = att.to(torch.float32)       
        att = att.softmax(dim=-1)
        att = self.attn_drop(att.to(dtype))
        att = torch.einsum('bhii, bhjk -> bhik', att, V)
        output = einops.rearrange(att, ' b h p e-> b p (h e)')
        output = self.proj(output)
        output = self.proj_drop(output)
        return output

    


def test():
    input = torch.rand(10, 20, 768)
    att = MultiHeadAttention(emb_size=768)
    output = att(input)
    print(output.shape)
    return None

if __name__ == "__main__":
    test()