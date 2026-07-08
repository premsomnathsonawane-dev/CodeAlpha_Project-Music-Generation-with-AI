# CodeAlpha_Project-Music-Generation-with-AI
An AI-powered music generation project that creates unique melodies using machine learning techniques. The system learns musical patterns from data and generates original compositions automatically. Built with Python and AI models, showcasing creativity, deep learning, and generative AI applications in music production. 🚀🎶

Author- Prem Somnath Sonawane

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI-powered Music Generation Script
Author: Deep Learning & Audio Science Expert
Description: Reads MIDI files, preprocesses notes/chords, trains an LSTM network,
             and generates standard MIDI musical sequences using music21 & Keras/Tensorflow.
"""

import os
import glob
import pickle
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from music21 import converter, instrument, note, chord, stream

# ==========================================
# 1. DATA PREPROCESSING (music21)
# ==========================================

def get_notes_from_midi(midi_dir):
    """
    Load MIDI files from a specified directory, parse notes and chords.
    If an element is a Note, get its pitch.
    If it's a Chord, join the pitches of its notes with a dot '.' (e.g. C4.E4.G4).
    """
    notes = []
    print(f"[*] Starting note extraction from MIDI files in: {midi_dir}")
    
    midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) or glob.glob(os.path.join(midi_dir, "*.midi"))
    if not midi_files:
        print("[!] Warning: No MIDI files found in specified directory.")
        return notes

    for file_path in midi_files:
        try:
            print(f"[-] Parsing: {os.path.basename(file_path)}")
            # Load and parse MIDI file
            midi_data = converter.parse(file_path)
            
            # Check if there are instrument parts
            parts = instrument.partitionByInstrument(midi_data)
            if parts: # File has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # File is a flat structure
                notes_to_parse = midi_data.flat.notes
                
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    # Join notes of the chord with a dot separator
                    notes.append(".".join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"[!] Error parsing file {file_path}: {e}")
            
    print(f"[+] Extraction complete. Total parsed elements: {len(notes)}")
    return notes


def prepare_sequences(notes, n_vocab, sequence_length=100):
    """
    Map unique note strings to integers (tokenization) and construct
    supervised input/output training sequences of window size 'sequence_length'.
    """
    # Create vocabulary mapping
    pitches = sorted(list(set(notes)))
    note_to_int = {note_str: num for num, note_str in enumerate(pitches)}
    
    # Save mapping for generation phase
    with open("notes_mapping.pkl", "wb") as f:
        pickle.dump(note_to_int, f)
        print("[+] Notes mapping successfully pickled to 'notes_mapping.pkl'.")

    network_input = []
    network_output = []

    # Create sliding windows of sequence_length
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    if n_patterns == 0:
        raise ValueError("[!] Error: Not enough parsed notes to construct sequences of length " + str(sequence_length))

    # Reshape the input into format compatible with LSTM layers [samples, time steps, features]
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize input features between 0 and 1
    network_input = network_input / float(n_vocab)
    
    # One-hot encode output targets
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return network_input, network_output


# ==========================================
# 2. LSTM MODEL ARCHITECTURE
# ==========================================

def create_network(network_input, n_vocab):
    """
    Build a Sequential Keras model optimized for music generation.
    Includes stacked LSTM layers with Dropout to prevent overfitting.
    """
    model = Sequential()
    
    # Layer 1: LSTM with 512 units, returning sequences
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    
    # Layer 2: LSTM with 512 units, returning sequences
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    
    # Layer 3: LSTM with 512 units
    model.add(LSTM(512))
    model.add(Dropout(0.3))
    
    # Layer 4: Dense Softmax layer for categorical output
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam", # Can also use 'rmsprop'
        metrics=["accuracy"]
    )
    
    model.summary()
    return model


# ==========================================
# 3. TRAINING PIPELINE
# ==========================================

def train_network(network_input, network_output, epochs=50, batch_size=64):
    """
    Train the model and save weight checkpoints whenever loss improves.
    """
    n_vocab = network_output.shape[1]
    model = create_network(network_input, n_vocab)
    
    # Define ModelCheckpoint callback
    checkpoint_path = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min"
    )
    callbacks_list = [checkpoint]
    
    print(f"[*] Commencing model training for {epochs} epochs...")
    model.fit(
        network_input, 
        network_output, 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=callbacks_list
    )
    print("[+] Training pipeline finished.")
    return model


# ==========================================
# 4. MUSIC GENERATION PIPELINE
# ==========================================

def sample_with_temperature(prediction_probabilities, temperature=1.0):
    """
    Heuristics for choosing the next note using temperature-based sampling.
    Low temperature -> very structured/expected notes.
    High temperature -> creative/unexpected notes.
    """
    if temperature <= 0.0:
        return np.argmax(prediction_probabilities)
    
    # Log scale adjust probabilities
    preds = np.log(prediction_probabilities + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    # Draw from multinomial distribution
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_melody(weights_file, notes_mapping_file, notes, sequence_length=100, num_notes_to_gen=500, temperature=0.7):
    """
    Load model weights, pick a seed sequence from processed inputs,
    and iteratively generate note/chord index strings.
    """
    # Load tokenization mapping
    with open(notes_mapping_file, "rb") as f:
        note_to_int = pickle.load(f)
    
    # Construct reverse mapping
    int_to_note = {num: char for char, num in note_to_int.items()}
    n_vocab = len(note_to_int)
    
    # Reconstruct network input for choosing random seed
    pitches = sorted(list(set(notes)))
    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
    
    # Recreate model with the loaded weights
    model = create_network(np.reshape(network_input, (len(network_input), sequence_length, 1)) / float(n_vocab), n_vocab)
    model.load_weights(weights_file)
    print(f"[+] Loaded weights from {weights_file} successfully.")

    # Select random seed sequence to kickstart generation
    start_idx = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start_idx]
    
    generated_sequence = []
    print(f"[*] Generating {num_notes_to_gen} notes using seed sequence of length {sequence_length}...")
    
    for note_index in range(num_notes_to_gen):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        
        # Predict probability array
        prediction = model.predict(prediction_input, verbose=0)[0]
        
        # Sample with temperature
        result_idx = sample_with_temperature(prediction, temperature)
        result_note = int_to_note[result_idx]
        
        generated_sequence.append(result_note)
        
        # Append predicted index and slide window forward
        pattern.append(result_idx)
        pattern = pattern[1:]

    return generated_sequence


# ==========================================
# 5. MIDI CONVERSION & SAVING
# ==========================================

def create_midi_from_sequence(prediction_output, output_filename="generated_ai_music.mid"):
    """
    Convert sequence of string-notes/chords back into music21 Note and Chord objects.
    Sequence is offset chronologically sequentially.
    """
    offset = 0.0
    output_notes = []

    print("[*] Building music21 stream from prediction output...")
    for pattern in prediction_output:
        # If pattern is a Chord (separated by dot)
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note) if current_note.isdigit() else current_note)
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # If pattern is a single Note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increment offset by 0.5 beat so notes are played sequentially
        offset += 0.5

    # Create MIDI Stream and write to file
    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_filename)
    print(f"[+] Output MIDI successfully exported as: '{output_filename}'")


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================

def main():
    # Setup directory configuration
    MIDI_DIRECTORY = "data/midi_input"
    WEIGHTS_PATH = "best_weights.keras"
    VOCAB_MAPPING_PATH = "notes_mapping.pkl"
    OUTPUT_MIDI_PATH = "generated_output.mid"
    
    print("="*60)
    print("      LSTM DEEP LEARNING MUSIC GENERATOR PIPELINE      ")
    print("="*60)
    
    # Create mock directory if not existing
    if not os.path.exists(MIDI_DIRECTORY):
        os.makedirs(MIDI_DIRECTORY)
        print(f"[i] Created '{MIDI_DIRECTORY}' directory. Please copy input MIDI files (.mid) here.")
        # Creating a small mock midi file is omitted here. Let's return.
        return

    # STEP 1: Preprocess notes from raw MIDI files
    notes = get_notes_from_midi(MIDI_DIRECTORY)
    if len(notes) == 0:
        print("[!] Aborting: Please supply MIDI files in the input folder.")
        return

    n_vocab = len(set(notes))
    
    # STEP 2: Prepare sliding sequences for training
    X, Y = prepare_sequences(notes, n_vocab, sequence_length=100)
    
    # STEP 3: Train LSTM model
    train_network(X, Y, epochs=3, batch_size=32) # Small epochs for safety demonstration
    
    # STEP 4 & 5: Generate and Export new audio
    # Note: For demo, we assume weights-improvement was saved as 'best_weights.keras'
    if os.path.exists(WEIGHTS_PATH) and os.path.exists(VOCAB_MAPPING_PATH):
        generated_notes = generate_melody(
            WEIGHTS_PATH, 
            VOCAB_MAPPING_PATH, 
            notes, 
            sequence_length=100, 
            num_notes_to_gen=200, 
            temperature=0.7
        )
        create_midi_from_sequence(generated_notes, OUTPUT_MIDI_PATH)
    else:
        print("[i] Skipping generation step because trained weights file or vocab mapping wasn't found.")

if __name__ == "__main__":
    main()
