import torch
import torch.nn as nn
from openSD3.models.dit.dit import DiT
from openSD3.schedulers.time_scheduler import LogitNormalSampler

class FlowMatching(nn.Module):
    def __init__(self,
                model,
                loss               
    ):
        super().__init__()
        self.flow_prediction_model = model
        self.loss = loss
    
    def forward(self, x_0, epsilon, t, Caption):
        device = x_0.device
        time = torch.tensor(t).to(device)
        time = time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        
        z_t = (1-time) * x_0 + time * epsilon
        u_t = -x_0 + epsilon
        v_t = self.flow_prediction_model(z_t, t, Caption)
        loss_ = self.loss(u_t, v_t)
        
        return loss_
    
    def inference(self, z_t, epsilon, t, Caption, step):
        # There are many numerical way to inference
        with torch.no_grad():
            v_t = self.flow_prediction_model(z_t, t, Caption)
            z_t += step * v_t

        return z_t

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xx = torch.rand(2,4, 16,32,32).to(device)
    epsilon = torch.rand_like(xx).to(device)
    Caption = ["A photo of a cat", "A photo of a dog"]

    t = [2,7]
    t = torch.tensor(t)
    loss_func = nn.MSELoss()
    model = DiT(learn_sigma=False)

    match = FlowMatching(model=model, loss=loss_func)
    output = match(xx, epsilon, t, Caption)

    print(output)

    return None

if __name__ == "__main__":

    test()