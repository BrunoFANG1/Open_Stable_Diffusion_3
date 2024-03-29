import torch
import torch.nn as nn
import einops

from diffusers.models import AutoencoderKL

# Consider rebuild VAE by myself

class VideoAutoencoderKL(nn.Module):
    def __init__(self, latent_channels, from_pretrained=None, micro_batch_size=None):
        super().__init__()
        self.module = AutoencoderKL(latent_channels = latent_channels).from_pretrained(from_pretrained) # The original paper uses latent_channel = 16, I cannot find a pretrained model with the same channels.
        self.micro_batch_size = micro_batch_size

    def encode(self, x): # x -> [batch size, channel, time , H, W]
        batch_size = x.shape[0]
        x = einops.rearrange(x, 'B C T H W -> (B T) C H W')
        bs = self.micro_batch_size
        x_out = []

        for i in range(0, x.shape[0], bs):
            x_bs = x[i:i+bs]
            x_bs = self.module.encode(x_bs).latent_dist.sample().mul_(0.18215)
            x_out.append(x_bs)
        x = torch.cat(x_out, dim=0)

        x = einops.rearrange(x, '(B T) C H W -> B C T H W ', B = batch_size)

        return x
    
    def decode(self, x): # x -> [batch_size, latent_channel, time, H, W]
        batch_size = x.shape[0]
        x = einops.rearrange(x, 'B C T H W -> (B T) C H W')
        bs = self.micro_batch_size
        x_out = []

        for i in range(0, x.shape[0], bs):
            x_bs = x[i:i+bs]
            x_bs = self.module.decode(x_bs/0.18215).sample
            x_out.append(x_bs)
        x = torch.cat(x_out, dim=0)
        x = einops.rearrange(x, '(B T) C H W -> B C T H W ', B = batch_size)

        return x
    
def main():
    x = torch.rand(5, 3, 5, 224, 224)
    VAE = VideoAutoencoderKL(latent_channels=16, from_pretrained = "madebyollin/sdxl-vae-fp16-fix", micro_batch_size = 1)
    latent = VAE.encode(x)
    print(latent.shape)
    output = VAE.decode(latent)
    print(output.shape)
    return None

if __name__ == "__main__":
    main()