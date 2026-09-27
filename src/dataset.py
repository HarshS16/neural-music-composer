"""
Step 2: PyTorch Dataset
========================
Converts encoded sequences into sliding-window training pairs
for the LSTM.

Input:  [C4, D4, E4, F4, G4, A4, B4, C5]
                    ↓ (seq_length=4)
X: [C4, D4, E4, F4]  →  Y: [D4, E4, F4, G4]
X: [D4, E4, F4, G4]  →  Y: [E4, F4, G4, A4]
X: [E4, F4, G4, A4]  →  Y: [F4, G4, A4, B4]
...
"""

import torch
from torch.utils.data import Dataset, DataLoader


class MusicDataset(Dataset):
    """
    Sliding window dataset for next-token prediction.

    Each sample is:
        x = encoded[i : i + seq_length]
        y = encoded[i+1 : i + seq_length + 1]

    So the model learns to predict the NEXT token at every position.
    """

    def __init__(self, sequences, token_to_idx, seq_length=64):
        """
        Args:
            sequences: list of integer sequences (from parse_midi)
            token_to_idx: vocabulary mapping
            seq_length: how many timesteps the LSTM sees at once
        """
        self.seq_length = seq_length
        self.vocab_size = len(token_to_idx)

        # Encode all sequences and concatenate with a separator
        # (REST tokens between pieces prevent learning cross-piece patterns)
        self.encoded = []
        rest_idx = token_to_idx.get(128, 0)

        for seq in sequences:
            encoded_seq = [token_to_idx[t] for t in seq]
            self.encoded.extend(encoded_seq)
            # Add a small gap between pieces
            self.encoded.extend([rest_idx] * 16)

        print(f"Dataset: {len(self)} samples | "
              f"seq_length={seq_length} | "
              f"vocab_size={self.vocab_size} | "
              f"total_tokens={len(self.encoded):,}")

    def __len__(self):
        return max(0, len(self.encoded) - self.seq_length - 1)

    def __getitem__(self, idx):
        x = torch.tensor(
            self.encoded[idx : idx + self.seq_length],
            dtype=torch.long
        )
        y = torch.tensor(
            self.encoded[idx + 1 : idx + self.seq_length + 1],
            dtype=torch.long
        )
        return x, y


def create_dataloader(sequences, token_to_idx, seq_length=64,
                      batch_size=64, shuffle=True, num_workers=0):
    """
    Convenience function to create a ready-to-use DataLoader.
    """
    dataset = MusicDataset(sequences, token_to_idx, seq_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True  # avoid partial batches
    )
    return loader, dataset.vocab_size
