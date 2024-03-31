import torch
import torch.nn as nn
import einops
import clip
import numpy as np
from openSD3.models.layers.blocks import (
    modulate,
    RMSNorm,
    MultiHeadAttention, 
    FinalLayer,
    TimestepEmbedder,
)

from transformers import (
    CLIPProcessor, 
    CLIPTextModel,
    T5Tokenizer, 
    T5EncoderModel,
)

from timm.models.vision_transformer import Mlp

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


class DiT(nn.Module):
    """
    Diffusion Transformer
    """
    def __init__(
            self,
            input_size = (16, 32, 32),
            in_channels = 4,
            patch_size=(1, 2, 2),
            emb_size=768,
            depth=2,
            num_heads=8,
            mlp_ratio=4,
            class_dropout_prob=0.1,
            learn_sigma=True,
            condition="text",
            no_temporal_pos_emb=False,
            caption_channels=1024,
            model_max_length=77,
            dtype=torch.float32,
            enable_flashattn=False,
            enable_layernorm_kernel=False,
            enable_sequence_parallelism=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels   ##### Why input chanel is 4 instead of 3 (RGB)
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.dtype = dtype
        self.t_embedder = TimestepEmbedder(emb_size)
        self.text_tokenizer_1 = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model_1 = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_tokenizer_2 = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_model_2 = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.text_tokenizer_3 = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.text_model_3 = T5EncoderModel.from_pretrained("google/flan-t5-large")
        self.Linear_c = nn.Linear(caption_channels, emb_size)
        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    emb_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    enable_RMS_Norm =False
                )
                for _ in range(depth)
            ]
        )
        self.t_model = TimestepEmbedder(emb_size)
        self.final_layer = FinalLayer(emb_size, np.prod(self.patch_size), self.out_channels) ###???
    
    def unpatchify(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = einops.rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def x_embedding(self, noised_latent):
        return x

    def y_embedding(self, Caption, t):
        t = torch.tensor(t, dtype=torch.int)

        cap_emb_1 = self.text_tokenizer_1(text=Caption, return_tensors="pt", padding='max_length', max_length=77, truncation=True)
        cap_emb_1 = self.text_model_1(**cap_emb_1)
        text_emb_1 = cap_emb_1.last_hidden_state
        # print(cap_emb_1.last_hidden_state.shape) # [2, 77, 512]

        cap_emb_2 = self.text_tokenizer_2(text=Caption, return_tensors="pt", padding='max_length', max_length=77, truncation=True)
        cap_emb_2 = self.text_model_1(**cap_emb_2)
        text_emb_2 = cap_emb_2.last_hidden_state
        # print(cap_emb_2.last_hidden_state.shape) # [2, 77, 512]

        y_emb = torch.cat([text_emb_1, text_emb_2], dim=-1)
        self.mlp_y = Mlp(
            in_features=text_emb_1.shape[2] + text_emb_2.shape[2], hidden_features=self.emb_size, out_features=self.emb_size,  drop=0
        )
        y = self.mlp_y(y_emb)
        t = self.t_embedder(t, dtype=y.dtype) ### There are some device issues; make sure to fix it in the future

        return y + t.unsqueeze(1) 
    
    def c_embedding(self, Caption):
        cap_emb_1 = self.text_tokenizer_1(text=Caption, return_tensors="pt", padding='max_length', max_length=77, truncation=True)
        cap_emb_1 = self.text_model_1(**cap_emb_1)
        text_emb_1 = cap_emb_1.last_hidden_state
        # print(cap_emb_1.last_hidden_state.shape) # [2, 77, 512]

        cap_emb_2 = self.text_tokenizer_2(text=Caption, return_tensors="pt", padding='max_length', max_length=77, truncation=True)
        cap_emb_2 = self.text_model_1(**cap_emb_2)
        text_emb_2 = cap_emb_2.last_hidden_state
        # print(cap_emb_2.last_hidden_state.shape) # [2, 77, 512]

        cap_emb_3 = self.text_tokenizer_3(text=Caption, return_tensors="pt", padding='max_length', max_length=77, truncation=True) 
        cap_emb_3 = self.text_model_3(input_ids=cap_emb_3['input_ids'], attention_mask=cap_emb_3['attention_mask'])       
        # print(cap_emb_3.last_hidden_state.shape) # [2, 77, 1024]
        text_emb_3 = cap_emb_3.last_hidden_state

        assert  text_emb_1.shape[2] + text_emb_2.shape[2] == text_emb_3.shape[2] , 'You need to make sure their embedding size is the same (Due to memery issue, I did not follow the same T5-xxl model in the orignal paper)'
        output_embedding = torch.cat([torch.cat([text_emb_1, text_emb_2], dim=2), text_emb_3], dim=1)

        c = self.Linear_c(output_embedding)

        return c

    def forward(self, x, t, Caption):
        return None

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xx = torch.rand(10,25,768).to(device)
    cc = torch.rand(10,28,768).to(device)
    yy = torch.rand(10,768).to(device)
    Caption = ["A photo of a cat", "A photo of a dog"]
    
    pack = DiT()
    # output = pack.c_embedding(Caption)
    # output = pack.y_embedding(Caption, t=[2,7])

    print(output.shape)
    # print(output_x.shape)
    return None

if __name__ == "__main__":
    test()