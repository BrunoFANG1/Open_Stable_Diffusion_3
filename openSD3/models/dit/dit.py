import torch
import torch.nn as nn
import einops
from openSD3.models.layers.blocks import modulate, RMSNorm, MultiHeadAttention

class MMDiTBlock(nn.Module):
    """
    A MM-DiT block in Figure 2 (b)
    """
    def __init__(
            self,
            emb_size,
            num_heads,
            mlp_ratio=4,
            enable_RMS_Norm =False
    ):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(emb_size, 12* emb_size, bias=True))
        self.norm1_c = nn.LayerNorm(emb_size, 1e-6, bias=False)
        self.norm2_c = nn.LayerNorm(emb_size, 1e-6, bias=False)
        self.norm1_x = nn.LayerNorm(emb_size, 1e-6, bias=False)
        self.norm2_x = nn.LayerNorm(emb_size, 1e-6, bias=False)
        self.QKV_c = nn.Linear(emb_size, 3*emb_size)
        self.QKV_x = nn.Linear(emb_size, 3*emb_size)
        self.enable_RMS_Norm = enable_RMS_Norm
        self.norm_c_Q = RMSNorm(-1, emb_size)
        self.norm_c_K = RMSNorm(-1, emb_size)
        self.norm_x_Q = RMSNorm(-1, emb_size)
        self.norm_x_K = RMSNorm(-1, emb_size)
        self.att = MultiHeadAttention(
            emb_size = emb_size
        )
        self.split_c = nn.Linear(emb_size, emb_size)
        self.split_x = nn.Linear(emb_size, emb_size)
        self.MLP_c = nn.Sequential(
            nn.Linear(emb_size, mlp_ratio*emb_size),
            nn.Linear(mlp_ratio*emb_size, emb_size)
        )
        self.MLP_x = nn.Sequential(
            nn.Linear(emb_size, mlp_ratio*emb_size),
            nn.Linear(mlp_ratio*emb_size, emb_size)
        )


    def forward(self, y, c, x):
        # y -> [batch_size, emd_size] ??? Not sure why
        # c -> [batch_size, seq_len_caption, emd_size]
        # x -> [batch_size, seq_len_img, emd_size]
        seq_len_c = c.shape[1]
        alpha_c, beta_c, gamma_c, delta_c, epsilon_c, zeta_c, alpha_x, beta_x, gamma_x, delta_x, epsilon_x, zeta_x = self.adaLN_modulation(y).chunk(12, dim=1) # [batch_size, emd_size]
        Q_c, K_c, V_c = self.QKV_c(modulate(self.norm1_c, c, alpha_c, beta_c)).chunk(3, dim=-1) 
        Q_x, K_x, V_x = self.QKV_x(modulate(self.norm1_x, x, alpha_x, beta_x)).chunk(3, dim=-1)
        
        if self.enable_RMS_Norm == True: # ?????Why it did not perform any norm to V matrix????
            Q_c = self.norm_c_Q(Q_c)
            Q_x = self.norm_x_Q(Q_x)
            K_c = self.norm_c_K(K_c)
            K_x = self.norm_x_K(K_x) 

        Q = torch.cat((Q_c, Q_x), dim=1) # [batch_size, seq_len_caption+seq_len_img, emd_size]
        K = torch.cat((K_c, K_x), dim=1) # [batch_size, seq_len_caption+seq_len_img, emd_size]
        V = torch.cat((V_c, V_x), dim=1) # [batch_size, seq_len_caption+seq_len_img, emd_size]

        att = self.att(Q, K, V)

        c = c + gamma_c.unsqueeze(1) * self.split_c(att[:, 0:seq_len_c, :])
        c = c + zeta_c.unsqueeze(1) * self.MLP_c(modulate(self.norm2_c, c, delta_c, epsilon_c))
        x = x + gamma_x.unsqueeze(1) * self.split_x(att[:, seq_len_c:, :])
        x = x + zeta_x.unsqueeze(1) * self.MLP_x(modulate(self.norm2_x, x, delta_x, epsilon_x))
        return c, x


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xx = torch.rand(10,25,768).to(device)
    cc = torch.rand(10,28,768).to(device)
    yy = torch.rand(10,768).to(device)
    pack = MMDiTBlock(768, 8).to(device)
    output_c, output_x = pack(yy, cc, xx)
    print(output_c.shape)
    print(output_x.shape)

if __name__ == "__main__":
    test()