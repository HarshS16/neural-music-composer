"""
Step 1: MIDI Parsing & Encoding
================================
This module handles loading MIDI files and converting them into
integer sequences that the LSTM can learn from.

Run standalone to test:
    python src/parse_midi.py --midi_dir data/midi
"""

import os
import glob
import json
import pretty_midi
import numpy as np
from collections import Counter


# ---------- MIDI → Note List ----------

def load_midi_notes(filepath):
    """
    Load a single MIDI file and extract all non-drum notes
    sorted by start time.

    Returns:
        list of dicts with keys: pitch, start, end, velocity
    """
    try:
        midi = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"  [SKIP] {filepath}: {e}")
        return []

    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append({
                "pitch": note.pitch,       # 0-127
                "start": note.start,       # seconds
                "end": note.end,           # seconds
                "velocity": note.velocity  # 0-127
            })

    notes.sort(key=lambda x: x["start"])
    return notes


# ---------- Note List → Integer Sequence ----------

REST_TOKEN = 128  # pitch values are 0-127, so 128 = silence

def notes_to_sequence(notes, fs=8):
    """
    Quantize notes into a fixed-timestep integer sequence.

    Args:
        notes: list of note dicts from load_midi_notes()
        fs: steps per second (8 ≈ eighth notes at 120 BPM)

    Returns:
        list of ints — each int is a MIDI pitch (0-127) or 128 (rest)
    """
    if not notes:
        return []

    total_time = max(n["end"] for n in notes)
    total_steps = int(total_time * fs) + 1
    sequence = [REST_TOKEN] * total_steps

    for note in notes:
        start_step = int(note["start"] * fs)
        end_step = int(note["end"] * fs)
        for step in range(start_step, min(end_step, total_steps)):
            # If multiple notes overlap, keep the highest pitch (melody)
            if sequence[step] == REST_TOKEN or note["pitch"] > sequence[step]:
                sequence[step] = note["pitch"]

    return sequence


# ---------- Vocabulary ----------

def build_vocab(sequences):
    """
    Build token↔index mappings from all sequences.

    Returns:
        token_to_idx: dict mapping pitch int → vocabulary index
        idx_to_token: dict mapping vocabulary index → pitch int
    """
    all_tokens = set()
    for seq in sequences:
        all_tokens.update(seq)

    sorted_tokens = sorted(all_tokens)
    token_to_idx = {t: i for i, t in enumerate(sorted_tokens)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}

    return token_to_idx, idx_to_token


# ---------- Full Pipeline ----------

def process_midi_directory(midi_dir, fs=8, min_length=64):
    """
    Full pipeline: directory of .mid files → encoded sequences + vocab.

    Args:
        midi_dir: path to folder containing .mid/.midi files
        fs: timesteps per second
        min_length: skip sequences shorter than this

    Returns:
        all_sequences: list of integer sequences (one per file)
        token_to_idx: vocabulary mapping
        idx_to_token: reverse vocabulary mapping
        stats: dict with dataset statistics
    """
    midi_files = sorted(
        glob.glob(os.path.join(midi_dir, "**", "*.mid*"), recursive=True)
    )

    if not midi_files:
        raise FileNotFoundError(
            f"No .mid/.midi files found in {midi_dir}\n"
            f"Download some from: http://www.piano-midi.de/"
        )

    print(f"Found {len(midi_files)} MIDI files in {midi_dir}")
    print("-" * 50)

    all_sequences = []
    skipped = 0

    for f in midi_files:
        notes = load_midi_notes(f)
        if not notes:
            skipped += 1
            continue

        seq = notes_to_sequence(notes, fs=fs)
        if len(seq) < min_length:
            skipped += 1
            continue

        all_sequences.append(seq)
        print(f"  ✓ {os.path.basename(f):40s} → {len(seq):6d} steps, "
              f"pitches {min(seq)}-{max(t for t in seq if t != REST_TOKEN)}")

    # Build vocabulary from all sequences
    token_to_idx, idx_to_token = build_vocab(all_sequences)

    # Compute stats
    all_tokens = [t for seq in all_sequences for t in seq]
    pitch_counts = Counter(t for t in all_tokens if t != REST_TOKEN)
    most_common_pitches = pitch_counts.most_common(5)

    stats = {
        "total_files": len(midi_files),
        "loaded_files": len(all_sequences),
        "skipped_files": skipped,
        "vocab_size": len(token_to_idx),
        "total_tokens": len(all_tokens),
        "rest_percentage": round(all_tokens.count(REST_TOKEN) / len(all_tokens) * 100, 1),
        "most_common_pitches": [
            {"pitch": p, "name": pretty_midi.note_number_to_name(p), "count": c}
            for p, c in most_common_pitches
        ]
    }

    print("-" * 50)
    print(f"Loaded: {stats['loaded_files']} files | "
          f"Skipped: {stats['skipped_files']} | "
          f"Vocab: {stats['vocab_size']} tokens | "
          f"Total: {stats['total_tokens']:,} timesteps")
    print(f"Rest tokens: {stats['rest_percentage']}%")
    print(f"Most common pitches: "
          + ", ".join(f"{p['name']}({p['count']})" for p in stats["most_common_pitches"]))

    return all_sequences, token_to_idx, idx_to_token, stats


# ---------- Save / Load Helpers ----------

def save_processed_data(all_sequences, token_to_idx, output_dir="data"):
    """Save processed sequences and vocab for reuse."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "sequences.npy"),
            np.array(all_sequences, dtype=object), allow_pickle=True)

    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump({"token_to_idx": {str(k): v for k, v in token_to_idx.items()}}, f)

    print(f"Saved processed data to {output_dir}/")


def load_processed_data(data_dir="data"):
    """Load previously processed sequences and vocab."""
    sequences = np.load(
        os.path.join(data_dir, "sequences.npy"), allow_pickle=True
    ).tolist()

    with open(os.path.join(data_dir, "vocab.json")) as f:
        raw = json.load(f)
        token_to_idx = {int(k): v for k, v in raw["token_to_idx"].items()}
        idx_to_token = {v: k for k, v in token_to_idx.items()}

    return sequences, token_to_idx, idx_to_token


# ---------- CLI Entry Point ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse MIDI files into sequences")
    parser.add_argument("--midi_dir", type=str, default="data/midi",
                        help="Directory containing .mid files")
    parser.add_argument("--fs", type=int, default=8,
                        help="Timesteps per second (default: 8)")
    parser.add_argument("--save", action="store_true",
                        help="Save processed data for later")
    args = parser.parse_args()

    sequences, t2i, i2t, stats = process_midi_directory(args.midi_dir, fs=args.fs)

    if args.save:
        save_processed_data(sequences, t2i)

    # Quick sanity check: print first 20 tokens of first sequence as note names
    if sequences:
        first_20 = sequences[0][:20]
        names = [pretty_midi.note_number_to_name(p) if p != REST_TOKEN else "—"
                 for p in first_20]
        print(f"\nFirst 20 tokens: {names}")
