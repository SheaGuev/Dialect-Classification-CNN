# Arabic Dialect Classification from Speech Using CNN

An academic deep learning project that classifies Arabic dialects from spoken audio using a Convolutional Neural Network (CNN) trained on Mel‑spectrogram representations.

## Project Overview

This project explores whether a CNN can learn to distinguish Arabic dialects from short speech audio clips. It takes raw `.wav` files, converts them into Mel‑spectrograms with Per‑Channel Energy Normalization (PCEN), and trains a 3‑block convolutional classifier to predict one of eight dialect classes.

A literature review comparing CNNs, RNNs/LSTMs, and Transformers for dialect classification is included in the notebook, along with references to prior academic work.

## Dataset

**DARPA TIMIT Acoustic‑Phonetic Continuous Speech Corpus** — a standard benchmark for acoustic‑phonetic modelling containing recordings of 8 major dialect regions of American English. *(Note: despite the "Arabic" focus of the literature review, the notebook experiments use the TIMIT corpus.)*

The dataset is downloaded programmatically via `kagglehub`.

**Dialect classes (8):**
| Label | Dialect Region |
|:-----:|:---------------:|
| dr1   | New England      |
| dr2   | Northern         |
| dr3   | North Midland    |
| dr4   | South Midland    |
| dr5   | Southern         |
| dr6   | New York City    |
| dr7   | Western          |
| dr8   | Army Brat (moved) |

## Architecture

All models are built with TensorFlow / Keras (Sequential API).

### CNN Model

```text
Input:  (64, 144, 1)  ← Mel‑spectrogram (64 mel bands × 144 time steps)

┌─ Conv2D  (32 filters, 3×3, ReLU, same padding)
│   BatchNormalization
│   MaxPooling2D (2×2)
│   Dropout(0.16)
│
├─ Conv2D  (64 filters, 3×3, ReLU, same padding)
│   BatchNormalization
│   MaxPooling2D (2×2)
│   Dropout(0.24)
│
├─ Conv2D  (128 filters, 3×3, ReLU, same padding)
│   BatchNormalization
│   MaxPooling2D (2×2)
│   Dropout(0.16)
│
├─ Flatten
│   Dense (128, ReLU)
│   BatchNormalization
│   Dropout(0.08)
│
└─ Dense (8, Softmax)
```

- **Optimizer:** Adam (learning rate = 1 × 10⁻⁴)
- **Loss:** Sparse Categorical Crossentropy
- **Regularisation:** BatchNormalisation + Dropout layers to combat overfitting

## Preprocessing Pipeline

1. **Load audio** — `librosa.load()` at original sample rate
2. **Mel‑spectrogram** — 64 mel bands (via `librosa.feature.melspectrogram`)
3. **PCEN** — Per‑Channel Energy Normalization (`librosa.pcen`) to reduce sensitivity to loudness variation
4. **Length normalisation** — all spectrograms trimmed or zero‑padded to 144 time steps
5. **Reshape** — add channel dimension for Conv2D input `(64, 144, 1)`

## Experiments & Results

| Setup | Training Accuracy | Validation Accuracy | Notes |
|-------|-------------------|---------------------|-------|
| Raw audio (initial) | ~17% | — | Data issues |
| After preprocessing | ~34% | — | Improvement from PCEN + trimming |
| Simple CNN, 30 epochs | ~95% | — | **Heavy overfitting** (loss 0.12, val_loss 6) |
| CNN with regularisation, 36 time steps | 66.6% | 15.2% | Audio clipped to 150 length |
| CNN with regularisation, 64 time steps | 74.1% | 13.9% | Audio clipped to 150 length |
| **Best balanced (epoch 22)** | **~74%** | **~14%** | 64 time steps, audio trimmed/padded to 144 |

> **Key finding:** The model achieves good training accuracy but consistently overfits — validation loss trends upward after the first few epochs. The gap between training and validation performance suggests the model memorises patterns rather than learning generalisable dialect features.

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3** | Language |
| **TensorFlow / Keras** | Deep learning framework — model definition, training, evaluation |
| **Librosa** | Audio loading, Mel‑spectrogram extraction, PCEN normalization |
| **NumPy** | Numerical operations |
| **Pandas** | Data manipulation |
| **Matplotlib** | Visualisation (waveforms, spectrograms, training curves) |
| **KaggleHub** | Programmatic dataset download |
| **libsndfile** | Audio codec support (via `pip install`) |
| **Graphviz** | Model architecture visualisation |

## How to Run

1. Clone the repository and open the notebook:
   ```bash
   git clone https://github.com/SheaGuev/Dialect-Classification-CNN.git
   cd Dialect-Classification-CNN
   jupyter notebook "Applied AI Dialect.ipynb"
   ```

2. Install dependencies (run the install cells in the notebook or):
   ```bash
   pip install pandas kagglehub librosa numpy matplotlib tensorflow graphviz
   ```

3. Run all cells — the dataset will be downloaded automatically via KaggleHub on first execution.

## Literature Review

The notebook includes a survey of three deep learning paradigms for dialect/speech classification:

- **CNNs** — strong at extracting spatial patterns from spectrograms; computationally efficient
- **RNNs (LSTMs)** — excellent for temporal sequence modelling; can suffer from vanishing gradients & high compute
- **Transformers** — capture long-range dependencies via attention; pre‑trained models enable fine‑tuning on smaller datasets

### Key References

- Alansari, I.S. (2023). *Artificial Intelligence Model to Detect and Classify Arabic Dialects.* Journal of Software Engineering and Applications, 16(07), pp.287–300. [doi:10.4236/jsea.2023.167015](https://doi.org/10.4236/jsea.2023.167015)
- Song, T., Thi, L. & Ta, T.V. (2023). *MPSA‑DenseNet: A novel deep learning model for English accent classification.* [arXiv:2306.08798](https://arxiv.org/abs/2306.08798)
- Themistocleous, C. (2019). *Dialect Classification From a Single Sonorant Sound Using Deep Neural Networks.* Frontiers in Communication, 4. [doi:10.3389/fcomm.2019.00064](https://doi.org/10.3389/fcomm.2019.00064)
- Humayun, M.A. et al. (2022). *A transformer fine‑tuning strategy for text dialect identification.* Neural Computing and Applications. [doi:10.1007/s00521-022-07944-5](https://doi.org/10.1007/s00521-022-07944-5)
- Baniata, L.H. & Kang, S. (2023). *Transformer Text Classification Model for Arabic Dialects That Utilizes Inductive Transfer.* Mathematics, 11(24), p.4960. [doi:10.3390/math11244960](https://doi.org/10.3390/math11244960)

## Future Work

- Data augmentation (noise injection, time stretching, pitch shifting) to reduce overfitting
- Audio trimming to capture only voiced segments (silence removal)
- Multi‑modal approach (speech + text/transcription)
- Hyperparameter tuning (learning rate schedules, deeper architectures)
- Ensemble with RNN/LSTM branch for temporal context
- Evaluation on held‑out dialect datasets for better generalisation testing