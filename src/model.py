"""
Step 3: The LSTM Model
=======================
~40 lines that learn music.

Architecture:
    Input (pitch integers)
        → Embedding (dense vectors)
        → LSTM × 2 layers (sequential memory)
        → Linear (next-note probabilities)
"""

import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    """
    Character-level LSTM for music generation.

    The model takes a sequence of encoded pitch tokens,
    embeds them into dense vectors, runs them through a
    stacked LSTM, and predicts the next token at each position.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256,
                 num_layers=2, dropout=0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Pitch integer → dense vector (like word embeddings in NLP)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Stacked LSTM — the core of the model
        # Layer 1 captures local patterns (scales, arpeggios)
        # Layer 2 captures longer structure (phrases, repetition)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Dropout before the output layer
        self.dropout = nn.Dropout(dropout)

        # Project hidden state → vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_length) tensor of token indices
            hidden: optional (h_0, c_0) tuple for stateful generation

        Returns:
            logits: (batch, seq_length, vocab_size) — raw scores
            hidden: (h_n, c_n) — final hidden state
        """
        embeds = self.embedding(x)                        # (B, S, embed_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)      # (B, S, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)                        # (B, S, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device="cpu"):
        """Create zero-initialized hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h, c)


def count_parameters(model):
    """Print a human-readable parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    return trainable
