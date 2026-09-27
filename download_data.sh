#!/bin/bash
# ============================================================
# Download Classical Piano MIDI files for training
# ============================================================
#
# Option A: Small dataset (recommended to start)
#   ~50 files from piano-midi.de (Bach, Chopin, Mozart)
#   Download manually from: http://www.piano-midi.de/
#   Save .mid files into data/midi/
#
# Option B: MAESTRO v3 (MIDI only — ~85MB)
#   200+ hours of virtuoso piano performances
#   Best quality, but takes longer to train
#
# ============================================================

echo "=== LSTM Music Generator — Data Download ==="
echo ""

mkdir -p data/midi

echo "Downloading MAESTRO v3 MIDI-only dataset (~85MB)..."
echo "Source: Google Magenta (Creative Commons BY-NC-SA 4.0)"
echo ""

# MAESTRO v3 MIDI-only download
wget -q --show-progress \
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip" \
    -O data/maestro-midi.zip

if [ $? -eq 0 ]; then
    echo "Extracting..."
    unzip -q data/maestro-midi.zip -d data/maestro-raw
    
    # Flatten: copy all .midi files into data/midi/
    find data/maestro-raw -name "*.midi" -exec cp {} data/midi/ \;
    
    # Count
    NUM_FILES=$(ls data/midi/*.midi 2>/dev/null | wc -l)
    echo ""
    echo "✓ Downloaded $NUM_FILES MIDI files into data/midi/"
    echo ""
    
    # Cleanup
    rm -f data/maestro-midi.zip
    rm -rf data/maestro-raw
else
    echo ""
    echo "⚠ Auto-download failed. Manual options:"
    echo ""
    echo "  Option A (Quickest — ~50 files):"
    echo "    1. Go to http://www.piano-midi.de/"
    echo "    2. Click a composer (Bach, Chopin, etc.)"
    echo "    3. Download individual .mid files"
    echo "    4. Save them into data/midi/"
    echo ""
    echo "  Option B (Best — MAESTRO dataset):"
    echo "    1. Go to https://magenta.tensorflow.org/datasets/maestro"
    echo "    2. Download 'MAESTRO v3.0.0 MIDI' (~85MB zip)"
    echo "    3. Extract and copy .midi files into data/midi/"
    echo ""
fi

echo "Once you have MIDI files in data/midi/, run:"
echo "  python src/train.py --epochs 50"
