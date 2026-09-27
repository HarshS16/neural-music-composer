"""
Step 5: Generate Music
=======================
Load a trained model and generate new MIDI compositions.

Usage:
    python src/generate.py --model outputs/best_model.pt --temperature 0.8

Temperature guide:
    0.2  → Conservative. Repetitive but safe.
    0.8  → Sweet spot. Creative but coherent.
    1.2  → Experimental. Surprising, sometimes wild.
    1.5+ → Chaos mode. Interesting for 10 seconds, then noise.
"""

import os
import sys
import json
import torch
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MusicLSTM
from src.parse_midi import REST_TOKEN


def load_model_and_vocab(model_path, vocab_path=None, device="cpu"):
    """Load trained model and vocabulary."""
    save_dir = os.path.dirname(model_path)
    if vocab_path is None:
        vocab_path = os.path.join(save_dir, "vocab.json")

    # Load vocab
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    token_to_idx = {int(k): v for k, v in vocab_data["token_to_idx"].items()}
    idx_to_token = {int(k): int(v) for k, v in vocab_data["idx_to_token"].items()}
    vocab_size = vocab_data["vocab_size"]

    # Recreate model and load weights
    model = MusicLSTM(vocab_size=vocab_size)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✓ Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')}, "
          f"loss {checkpoint.get('loss', '?'):.4f})")

    return model, token_to_idx, idx_to_token


def generate_sequence(model, seed_tokens, token_to_idx, idx_to_token,
                      length=500, temperature=0.8, device="cpu"):
    """
    Generate a sequence of notes autoregressively.

    Args:
        model: trained MusicLSTM
        seed_tokens: list of pitch integers to start from
        token_to_idx: vocabulary mapping
        idx_to_token: reverse mapping
        length: how many new tokens to generate
        temperature: sampling temperature (lower=safer, higher=wilder)
        device: 'cpu' or 'cuda'

    Returns:
        list of pitch integers (0-127 or 128 for rest)
    """
    model.eval()
    generated = list(seed_tokens)

    # Encode seed
    input_indices = [token_to_idx.get(t, 0) for t in seed_tokens]

    with torch.no_grad():
        hidden = None

        for i in range(length):
            # Use last 64 tokens as context (or less if we don't have 64 yet)
            context = input_indices[-64:]
            x = torch.tensor([context], dtype=torch.long).to(device)

            logits, hidden = model(x, hidden)

            # Take the prediction for the LAST position
            last_logits = logits[0, -1, :] / temperature

            # Convert to probabilities
            probs = torch.softmax(last_logits, dim=0)

            # Sample from the distribution (not argmax — that's boring)
            next_idx = torch.multinomial(probs, 1).item()
            next_token = idx_to_token[next_idx]

            generated.append(next_token)
            input_indices.append(next_idx)

            # Detach hidden state to prevent memory buildup
            hidden = tuple(h.detach() for h in hidden)

    return generated


def sequence_to_midi(sequence, output_path, fs=8, default_velocity=80):
    """
    Convert a pitch sequence back into a playable MIDI file.

    Args:
        sequence: list of pitch integers (0-127, 128=rest)
        output_path: where to save the .mid file
        fs: timesteps per second (must match training fs)
        default_velocity: note loudness (0-127)
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0, name="Piano")  # Acoustic Grand

    i = 0
    while i < len(sequence):
        pitch = sequence[i]

        # Skip rests
        if pitch == REST_TOKEN:
            i += 1
            continue

        # Find how long this note sustains (consecutive same pitch)
        start = i
        while i < len(sequence) and sequence[i] == pitch:
            i += 1
        end = i

        # Create the note
        note = pretty_midi.Note(
            velocity=default_velocity,
            pitch=int(pitch),
            start=start / fs,
            end=end / fs
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(output_path)

    duration = len(sequence) / fs
    num_notes = len(piano.notes)
    print(f"✓ Saved {output_path} ({duration:.1f}s, {num_notes} notes)")

    return midi


def plot_piano_roll(sequence, output_path=None, fs=8, title="Generated Piano Roll"):
    """
    Visualize the sequence as a piano roll (great for the article).
    """
    max_steps = min(len(sequence), 400)  # Show first ~50 seconds
    piano_roll = np.zeros((128, max_steps))

    for i in range(max_steps):
        if sequence[i] != REST_TOKEN:
            piano_roll[sequence[i], i] = 1

    # Find pitch range (don't show empty octaves)
    active_pitches = [sequence[i] for i in range(max_steps) if sequence[i] != REST_TOKEN]
    if not active_pitches:
        print("Warning: sequence is all rests!")
        return

    lo = max(0, min(active_pitches) - 5)
    hi = min(127, max(active_pitches) + 5)

    plt.figure(figsize=(16, 5))
    plt.imshow(piano_roll[lo:hi, :max_steps],
               aspect="auto", origin="lower", cmap="magma",
               interpolation="nearest")

    # Y-axis: show note names every octave
    ytick_positions = list(range(0, hi - lo, 12))
    ytick_labels = [pretty_midi.note_number_to_name(lo + p) for p in ytick_positions]
    plt.yticks(ytick_positions, ytick_labels, fontsize=9)

    # X-axis: convert steps to seconds
    xtick_positions = list(range(0, max_steps, fs * 5))  # every 5 seconds
    xtick_labels = [f"{p / fs:.0f}s" for p in xtick_positions]
    plt.xticks(xtick_positions, xtick_labels, fontsize=9)

    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Pitch", fontsize=12)
    plt.title(title, fontsize=14)
    plt.colorbar(label="Note Active", shrink=0.6)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"✓ Piano roll saved to {output_path}")
    plt.close()


def get_seed_from_data(sequences, token_to_idx, seed_length=16):
    """Grab a seed sequence from the training data."""
    if not sequences:
        # Fallback: C major scale
        return [60, 62, 64, 65, 67, 69, 71, 72,
                72, 71, 69, 67, 65, 64, 62, 60]

    # Pick a random piece, grab first N tokens
    import random
    seq = random.choice(sequences)
    start = random.randint(0, max(0, len(seq) - seed_length - 1))
    return seq[start : start + seed_length]


# ---------- CLI Entry Point ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate music with trained LSTM")
    parser.add_argument("--model", type=str, default="outputs/best_model.pt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--length", type=int, default=500,
                        help="Number of tokens to generate (~62s at fs=8)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .mid path (auto-generated if not set)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Generate multiple samples at different temperatures")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", action="store_true",
                        help="Save piano roll visualizations")
    args = parser.parse_args()

    # Load model
    model, token_to_idx, idx_to_token = load_model_and_vocab(
        args.model, device=args.device
    )

    # Load training data for seed sequences
    from src.parse_midi import load_processed_data
    try:
        sequences, _, _ = load_processed_data("data")
    except FileNotFoundError:
        sequences = []

    save_dir = os.path.dirname(args.model)

    if args.num_samples > 1:
        # Generate at multiple temperatures for comparison
        temperatures = [0.3, 0.8, 1.2]
        print(f"\nGenerating {len(temperatures)} samples at temps: {temperatures}")

        for temp in temperatures:
            seed = get_seed_from_data(sequences, token_to_idx)
            generated = generate_sequence(
                model, seed, token_to_idx, idx_to_token,
                length=args.length, temperature=temp, device=args.device
            )

            out_path = os.path.join(save_dir, f"generated_temp{temp}.mid")
            sequence_to_midi(generated, out_path)

            if args.plot:
                plot_path = os.path.join(save_dir, f"piano_roll_temp{temp}.png")
                plot_piano_roll(generated, plot_path,
                                title=f"Generated (temperature={temp})")
    else:
        seed = get_seed_from_data(sequences, token_to_idx)
        generated = generate_sequence(
            model, seed, token_to_idx, idx_to_token,
            length=args.length, temperature=args.temperature, device=args.device
        )

        out_path = args.output or os.path.join(save_dir, "generated.mid")
        sequence_to_midi(generated, out_path)

        if args.plot:
            plot_path = out_path.replace(".mid", "_piano_roll.png")
            plot_piano_roll(generated, plot_path)

    print(f"\n🎵 Done! Open the .mid files in:")
    print(f"   • MuseScore (free) — musescore.org")
    print(f"   • GarageBand (Mac)")
    print(f"   • Online: signal.vercel.app/edit")
    print(f"   • Convert to audio: timidity or fluidsynth")
