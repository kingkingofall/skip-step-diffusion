import torch
import torch.nn as nn

def noise_estimation_loss(model,
                        x0: torch.Tensor,
                        t: torch.LongTensor,
                        e1: torch.Tensor,
                        b: torch.Tensor,
                        skip=9, # you can set 1(s^2-DM^2) or 49(s^2-DM^50).
                        keepdim=False):
    e2 = torch.randn_like(x0).cuda()
    
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    
    # Create index for a_s, replacing negative index with 0
    index = torch.clamp(t - skip, min=0)
    a_s = (1-b).cumprod(dim=0).index_select(0, index).view(-1, 1, 1, 1)
    
    a_skip = a / a_s
    x = x0 * a.sqrt() + e1 * (1.0 - a).sqrt()
    output = model(x, t.half())   
    mask_less_skip = t < skip

    mask_greater_equal_skip = ~mask_less_skip
    loss_less_skip = nn.MSELoss(reduction='none')(e1, output).mean(dim=(1, 2, 3))[mask_less_skip]
    loss_greater_equal_skip = 0.01 * nn.MSELoss(reduction='none')(e2, (1.0 - a_skip).sqrt() / a_skip.sqrt() * output).mean(dim=(1, 2, 3))[mask_greater_equal_skip] \
                            +  0.99 * nn.MSELoss(reduction='none')(e1, output,).mean(dim=(1, 2, 3))[mask_greater_equal_skip]

    loss = torch.cat([loss_less_skip, loss_greater_equal_skip]).float()

    return loss.mean(dim=0)

loss_registry = {
    'simple': noise_estimation_loss,
}
