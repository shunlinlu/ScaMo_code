import torch
import torch.nn as nn

class UnigramLoss(nn.Module):
    def __init__(self, look_up_prob, ignore_index):
        super(UnigramLoss, self).__init__()
        # one for the end token, 
        self.look_up_prob = look_up_prob
        self.ignore_index = ignore_index

    def forward(self, input_logits, targets):
        # import pdb; pdb.set_trace()
        probabilities = self.look_up_prob[targets].unsqueeze(2)
        normalized_logits = torch.nn.functional.softmax(input_logits, dim=-1) / probabilities + (1e-10)
        normalized_logits = normalized_logits.reshape(-1, normalized_logits.shape[-1])
        loss = torch.nn.functional.nll_loss(torch.log(normalized_logits), targets.reshape(-1), ignore_index=self.ignore_index)
        # judge loss is nan
        if torch.isnan(loss):
            import pdb; pdb.set_trace()
        return loss

    