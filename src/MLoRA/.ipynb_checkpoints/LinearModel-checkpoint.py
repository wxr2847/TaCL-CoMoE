from torch import nn
import torch

class LinearModel(nn.Module):
    """
    Linear models used for the sentiment_polarity/ metaphor_type representations
    """

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(4096, 4096)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        """
        x: ChatGLM's last_hidden_state [seq_len, batch_size, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
    
        # 先把 x 转为 [batch_size, seq_len, hidden_size]
        x = x.permute(1, 0, 2)
    
        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features_summed = torch.sum(last_state, dim=1)
        dropped = self.dropout(features_summed)
        return torch.stack((self.layer_1(features_summed), self.layer_1(dropped)), 1)