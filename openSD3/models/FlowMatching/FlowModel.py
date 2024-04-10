import torch
import torch.nn as nn

class FlowBasedModel(nn.Module):
    def __init__(self,
                model,
                loss,
                batch_size
                
    ):
        super().__init__()
        self.flow_prediction_model = model
        self.loss = loss
        self.batch_size = batch_size
    
    def train(self, x_0, epsilon, t, Caption):

        z_t = (1-t) * x_0 + t * epsilon
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