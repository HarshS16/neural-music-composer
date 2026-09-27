# ACTION PLAN — Day by Day

## DAY 1: Setup + Get Data Running (2-3 hours)

### Do these in order:

- [ ] **Clone/copy this project** to your machine
- [ ] **Create a virtual environment**
      ```
      python -m venv venv
      source venv/bin/activate        # Mac/Linux
      .\venv\Scripts\activate          # Windows
      ```
- [ ] **Install dependencies**
      ```
      pip install -r requirements.txt
      ```
- [ ] **Get MIDI files** — pick ONE of these:
      
      **Fastest (5 min):** Go to http://www.piano-midi.de/, click "Bach" or "Chopin",
      right-click → Save each .mid file into `data/midi/`. Grab 20-30 files.
      
      **Best (10 min):** Run `bash download_data.sh` to get the full MAESTRO dataset
      (1200+ piano performances from Google). If wget fails, download manually from
      https://magenta.tensorflow.org/datasets/maestro (get the "MIDI only" zip).

- [ ] **Test the parser**
      ```
      python src/parse_midi.py --midi_dir data/midi --save
      ```
      You should see output like:
      ```
      Found 25 MIDI files in data/midi
      --------------------------------------------------
        ✓ bach_846.mid                 →   2048 steps, pitches 36-84
        ✓ chopin_nocturne_op9_1.mid    →   4520 steps, pitches 33-89
      ...
      Loaded: 25 files | Vocab: 67 tokens | Total: 85,230 timesteps
      ```
- [ ] **Celebrate** — your data pipeline works. The hardest part is done.

---

## DAY 2: Train Your First Model (2-3 hours)

- [ ] **Run training** (start with small settings to verify everything works):
      ```
      python src/train.py --midi_dir data/midi --epochs 10 --device auto
      ```
      This should finish in a few minutes. Check that loss is decreasing.

- [ ] **Run the real training:**
      ```
      python src/train.py --midi_dir data/midi --epochs 50 --device auto
      ```
      
      **What to watch for:**
      - Loss should drop from ~4.5 → ~1.5 over 50 epochs
      - If loss stalls above 3.0: add more MIDI files or increase `--hidden_dim 512`
      - If loss drops below 1.0: you might be overfitting (add `--dropout 0.4`)

- [ ] **Check outputs/**
      - `best_model.pt` — your trained model
      - `loss_curve.png` — screenshot this for your article
      - `training_config.json` — hyperparameters and final metrics

---

## DAY 3: Generate Music + Iterate (2-3 hours)

- [ ] **Generate your first compositions:**
      ```
      python src/generate.py --model outputs/best_model.pt --num_samples 3 --plot
      ```
      This creates 3 MIDI files at temperatures 0.3, 0.8, and 1.2.

- [ ] **Listen to them!**
      - Open in MuseScore (free download at musescore.org)
      - Or upload to https://signal.vercel.app/edit
      - Or convert: `timidity outputs/generated_temp0.8.mid -Ow -o demo.wav`

- [ ] **Iterate on quality:**
      
      | Problem | Fix |
      |---------|-----|
      | All one note repeating | Raise temperature to 1.0, or train more epochs |
      | Random noise | Lower temperature to 0.5, or train more epochs |
      | Too short phrases | Increase `--seq_length 128` and retrain |
      | Sounds mechanical | Add velocity variation in generate.py |
      | Want richer sound | Use more MIDI files (MAESTRO full dataset) |

- [ ] **Save your best 3-4 outputs** — you'll embed these in the article.

---

## DAY 4: Visualizations for the Article (2 hours)

- [ ] **Piano roll comparisons** — generated at temperatures 0.3 vs 0.8 vs 1.2:
      ```
      python src/generate.py --model outputs/best_model.pt --num_samples 3 --plot
      ```
      The `--plot` flag saves piano roll PNGs in outputs/.

- [ ] **Loss curve** — already saved at `outputs/loss_curve.png`

- [ ] **Screenshot of MIDI in a DAW** — open your best .mid in GarageBand or
      MuseScore, take a screenshot of the notation. This is your hero image.

- [ ] **Convert to audio for embedding** in the article:
      - MuseScore: File → Export → MP3
      - Or: `timidity outputs/generated_temp0.8.mid -Ow -o demo.wav`
      - Upload to SoundCloud (free) for embeddable player

---

## DAY 5-6: Write the Article (4-6 hours)

- [ ] **Hook** (first 3 lines — this is what shows before "see more" on LinkedIn):
      ```
      I fed 50 Bach and Chopin pieces to a neural network.
      After 15 minutes of training, it started composing on its own.
      Here's the code, the music, and what it taught me about how AI "thinks."
      ```

- [ ] **Section 1:** Why MIDI + LSTM (300 words)
      - MIDI = sheet music for computers (symbolic, not audio)
      - LSTMs have memory — they learn patterns like "resolve tension"
      - Same architecture that powered early Google Translate

- [ ] **Section 2:** Data pipeline with code snippets from parse_midi.py
      - Show the piano roll visualization
      - Explain the encoding: "C4 = 60, D4 = 62, rest = 128"

- [ ] **Section 3:** The model — paste the MusicLSTM class
      - Draw/describe the architecture diagram
      - Key insight: "pitches are like words, embedding them works the same way"

- [ ] **Section 4:** Training — show loss curve with annotations
      - "At epoch 10, it learned scales. By epoch 30, it found phrases."
      - Mention gradient clipping (connect to vanishing gradient theory)

- [ ] **Section 5:** Generation — THE MONEY SECTION
      - Explain temperature with the pianist analogy
      - **Embed audio samples** — this is what gets shared
      - Show side-by-side piano rolls at different temperatures

- [ ] **Section 6:** What I learned (200 words)
      - Why understanding RNNs matters even in the transformer era
      - Tease follow-up: "Next I'm adding attention on top of the LSTM..."

---

## DAY 7: Publish + Promote

- [ ] **Push code to GitHub** with clean README
- [ ] **Create Google Colab notebook** (copy the 4 src files into one notebook)
- [ ] **Publish article on LinkedIn** (Article format, not post)
      - Add tags: #DeepLearning #PyTorch #LSTM #MusicAI #MachineLearning
      - Post Tuesday-Wednesday, 8-10 AM your timezone
- [ ] **Cross-post to Medium** with canonical URL to LinkedIn
- [ ] **Share a teaser POST** on LinkedIn (not article):
      ```
      🎵 I trained an LSTM on classical piano music.
      
      Here's what it composed after seeing 50 Bach pieces ↓
      
      [attach audio clip or short video of MIDI playing]
      
      Full code + article in comments.
      ```
- [ ] **Reply to every comment** in the first 2 hours (algorithm boost)
- [ ] **Share in relevant subreddits:**
      - r/MachineLearning (title: "[P] LSTM Music Generator in ~200 lines of PyTorch")
      - r/deeplearning
      - r/learnmachinelearning

---

## QUICK COMMANDS CHEATSHEET

```bash
# Parse MIDI files and check data
python src/parse_midi.py --midi_dir data/midi --save

# Train (auto-detects GPU)
python src/train.py --midi_dir data/midi --epochs 50

# Generate 3 samples with piano roll plots
python src/generate.py --model outputs/best_model.pt --num_samples 3 --plot

# Generate 1 long sample at custom temperature
python src/generate.py --model outputs/best_model.pt --temperature 0.8 --length 1000

# Retrain with bigger model if output quality is poor
python src/train.py --midi_dir data/midi --epochs 80 --hidden_dim 512 --seq_length 128
```
