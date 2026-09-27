# 🎵 LSTM Music Generator

Generate classical piano compositions using a 2-layer LSTM trained on MIDI files.

Built with PyTorch. ~200 lines of core code.

## Quick Start (5 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download training data (or manually add .mid files to data/midi/)
bash download_data.sh

# 3. Train the model (~5-15 min on GPU, ~30-60 min on CPU)
python src/train.py --midi_dir data/midi --epochs 50

# 4. Generate music at 3 different temperatures
python src/generate.py --model outputs/best_model.pt --num_samples 3 --plot

# 5. Listen to outputs/generated_temp0.8.mid in any MIDI player
```

## How It Works

```
MIDI files → Parse notes → Integer sequences → Train LSTM → Generate → MIDI output
                                                    ↑
                                          Next-token prediction
                                          (same idea as GPT, but
                                           for music, with an LSTM)
```

## Project Structure

```
├── data/midi/          ← Put your .mid files here
├── src/
│   ├── parse_midi.py   ← MIDI loading + pitch encoding
│   ├── dataset.py      ← Sliding window PyTorch Dataset
│   ├── model.py        ← MusicLSTM (embed → LSTM×2 → linear)
│   ├── train.py        ← Training loop with grad clipping + LR schedule
│   └── generate.py     ← Temperature sampling + MIDI export + piano roll plots
├── outputs/            ← Trained models, generated .mid files, plots
├── requirements.txt
└── download_data.sh    ← Downloads MAESTRO dataset
```

## Temperature Sampling

| Temperature | Style | Best For |
|-------------|-------|----------|
| 0.2-0.4 | Conservative, repetitive | Studying how the model learned patterns |
| 0.7-0.9 | Balanced, musical | **Best output quality** |
| 1.0-1.3 | Experimental, surprising | Creative exploration |
| 1.5+ | Chaotic | Blog post demos showing what NOT to do |

## Training Tips

- **Start small**: 20-50 MIDI files, 50 epochs, watch the loss curve
- **Good loss range**: 1.0-1.8 means the model learned musical structure
- **Overfit first**: If loss won't drop below 3.0, increase hidden_dim to 512
- **CPU is fine**: Training 50 files for 50 epochs takes ~30 min on a modern laptop

## Listening to Output

The generated `.mid` files can be played in:
- **MuseScore** (free, cross-platform) — musescore.org
- **GarageBand** (Mac)
- **Online**: signal.vercel.app/edit
- **Convert to WAV**: `timidity generated.mid -Ow -o output.wav`
