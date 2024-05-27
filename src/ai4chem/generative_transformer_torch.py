"""Naive generative transformer implementation.

Taken from Kevin Jablonka's llm-from-scratch notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Attention head."""

    def __init__(self, n_embed, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('lower_triangular', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, C  = x.shape
        key = self.key(x)
        query = self.query(x) # B, T, head
        value = self.value(x)   # B, T, head

        weight_matrix = query @ key.transpose(-2, -1) * C ** (-0.5) # shape (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        weight_matrix = weight_matrix.masked_fill(self.lower_triangular[:T, :T].logical_not(), float('-inf'))
        weight_matrix = F.softmax(weight_matrix, dim=-1)

        out = weight_matrix @ value # shape (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multihead attention module."""

    def __init__(self, num_heads, n_embed, block_size, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, block_size, head_size) for _ in range(num_heads)])

    def forward(self, x):
        # x is a tensor of shape (B, T, C)
        # we want to compute the attention for each head
        # and then concatenate the results
        # we will have a tensor of shape (B, T, num_heads * head_size)
        # in practice, we might not concatenate but add another dimension
        # to the tensors
        return torch.cat([head(x) for head in self.heads], dim=-1)


class FeedForwardLayer(nn.Module):
    """Feed-forward layer (linear->ReLU->linear)."""

    def __init__(self, n_embed, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_embed)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, block_size, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, n_embed=n_embd, block_size=block_size, head_size=head_size)
        self.ffwd = FeedForwardLayer(n_embd, n_embd*4)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

class GenerativeTransformer(nn.Module):
    """Naive generative transformer (decoder) model."""

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_blocks):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.layers = nn.Sequential(*[Block(n_embd, block_size, n_head) for _ in range(n_blocks)])
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, x):
        _, T = x.shape

        x = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))  # b,tc, batch, time - seqeuence length, embedding dimension
        x = self.layers(x)
        x = self.head(x)
        return x

    def loss(self, x, y):
        # x is a tensor of shape (B, T)
        logits = self.forward(x) # (B, T, C)
        B, _, C = logits.shape
        # Note that that the implementation below is because of how we - for educational purposes - have defined the dataset
        # A better way is to have inputs and outputs of the same length (and to not manually code the sliding window
        # but to instead use a causal mask)
        logits = logits[:, -1, :]
        logits = logits.view(B, C)
        y = y.view(B)
        loss = F.cross_entropy(logits, y)
        return loss

    def generate(self, x, max_new_tokens=100):
        # x is a tensor of shape (B, T)
        # we generate max_new_tokens new tokens
        new_tokens = []
        for _t in range(max_new_tokens):
            x_ = x[:, -self.block_size:]
            logits = self.forward(x_)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            new_tokens.append(next_token)
            x = torch.cat([x, next_token], dim=1)
        return x
