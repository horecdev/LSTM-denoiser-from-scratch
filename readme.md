# Pure NumPy Audio Denoiser (LSTM)

A complete speech enhancement and audio denoising pipeline built entirely from scratch using only NumPy. No high-level machine learning frameworks (like PyTorch or TensorFlow) or audio processing libraries (like Librosa) were used for the core architecture and signal processing.

## Features

- **Custom LSTM Implementation:** Forward and backward pass, cell states, backpropagation through time (BPTT), and gating mechanisms implemented manually via raw math.
- **Signal Processing from Scratch:** Custom implementation of the Short-Time Fourier Transform (STFT) and Inverse Short-Time Fourier Transform (ISTFT) to convert temporal audio signals into the frequency domain and back.
- **Time-Frequency Masking:** The network predicts masks which are applied to the noisy spectrogram to isolate clean speech and suppress background noise.
- **Visualizations:** The included Jupyter Notebook (`noise_reduction.ipynb`) contains step-by-step visualizations of the noisy spectrograms, the predicted masks, and the final cleaned audio spectrograms.

## Project Structure

- `/src/` - Contains the raw NumPy implementations of the LSTM layers, optimization logic, and the STFT/ISTFT functions.
- `noise_reduction.ipynb` - The main notebook demonstrating the data pipeline, training process, inference, and visualization of the masks/spectrograms.
- `README.md` - Project documentation.
