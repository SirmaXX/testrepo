import torch
import torch.nn as nn


class UstaMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim,
        output_dim,
        context_length,
        num_heads,
        dropout_rate=0,
        device="cpu",
    ):
        super().__init__()

        self.context_length = context_length

        # device parametrelerini kaldır
        self.multi_head_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout_rate, device=device
        )
        self.projection = nn.Linear(embedding_dim, output_dim, device=device)

    def forward(self, x):
        number_of_tokens = x.shape[0]
        x = x[: self.context_length]

        # Mask'ı forward içinde dinamik olarak oluştur
        mask = torch.triu(
            torch.ones(number_of_tokens, number_of_tokens, device=x.device), diagonal=1
        ).bool()

        out, _ = self.multi_head_attention(x, x, x, attn_mask=mask)
        out = self.projection(out)
        return out
