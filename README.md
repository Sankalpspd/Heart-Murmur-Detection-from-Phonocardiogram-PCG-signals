# Heart-Murmur-Detection-from-Phonocardiogram-PCG-signals

# Overview

Heart murmurs are abnormal heart sounds that may indicate underlying cardiovascular conditions such as valve defects. Early and accurate detection of murmurs is important for timely medical intervention, but manual auscultation requires significant clinical expertise and may be subjective. This project develops an AI-based system for automated heart murmur detection using deep learning techniques applied to heart sound recordings.

The objective of this project is to design a robust model capable of classifying heart sound recordings into murmur present or murmur absent, improving reliability in identifying pathological heart sounds. The system leverages both acoustic features from heart sound recordings and positional metadata from auscultation locations to enhance classification performance.

The dataset used in this project comes from the PhysioNet CirCor DigiScope Dataset from the PhysioNet/CinC Challenge 2022, which contains heart sound recordings collected from multiple patients at different auscultation positions including aortic valve (AV), mitral valve (MV), pulmonary valve (PV), and tricuspid valve (TV). Each patient recording is labeled with the presence or absence of a murmur.

# Data Preprocessing

The raw audio recordings undergo several preprocessing steps to extract meaningful features for model training.

# Audio Segmentation

Each heart sound recording is divided into fixed-length segments to increase the number of training samples and capture localized acoustic patterns.

# Feature Extraction

For each segment, log-Mel spectrograms are generated to represent the time-frequency characteristics of heart sounds. Additionally:

Delta features capture temporal changes in the signal.

Delta-delta features capture acceleration of spectral changes.

These three components are stacked to form a 3-channel spectrogram representation, similar to an RGB image.

# Normalization

Spectrogram features are normalized per frequency bin to stabilize training and improve model convergence.

# Positional Encoding

Heart sound recordings are associated with their auscultation position (AV, MV, PV, TV). These positions are encoded and fed into the model as an additional input branch, allowing the model to learn positional patterns associated with murmurs.

# Handling Class Imbalance

The dataset contains significantly more normal recordings than murmur cases. To address this imbalance:

Oversampling is applied to the minority class.

Binary focal loss is used to focus learning on harder examples.

# Model Architecture

The model is designed as a multi-input deep neural network combining convolutional and recurrent layers.

# CNN Feature Extractor

A stack of Convolutional Neural Network (CNN) layers processes the spectrogram input to extract spatial and frequency-based acoustic features.

# Temporal Modeling with Bi-LSTM

The CNN output is reshaped and passed through Bidirectional LSTM layers, enabling the model to capture temporal dependencies within heart sound signals.

# Metadata Integration

A separate branch processes auscultation position metadata, which is then concatenated with the learned audio features.

# Classification Head

The combined features pass through fully connected layers with dropout regularization before producing a binary murmur prediction using a sigmoid output.

# Training Strategy

The model is trained using:

Adam optimizer

Binary focal loss to emphasize difficult samples

Early stopping based on validation AUC to prevent overfitting

Oversampled training data to mitigate class imbalance

# Evaluation Metrics

Model performance was evaluated using multiple metrics to capture both overall accuracy and minority-class performance:


AUC (Area Under ROC Curve) – Ability to separate murmur and non-murmur classes

Classification Report
Class	   Precision	Recall	F1-score
Absent	    0.93	   0.76	   0.84
Present	    0.44	   0.77	   0.56

Overall Accuracy: 0.76
Macro F1-score: 0.70

# Tools Used

Python

TensorFlow / Keras

Librosa (audio feature extraction)

NumPy & Pandas

Scikit-learn

# Future Improvements

Train another model to predict the patient's overall outcome ("Normal", "abnormal") using the patient's demographic data and the heart murmur prediction.
