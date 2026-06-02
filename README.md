# SpectrumX Signal Detection Challenge

This project was developed for the NSF SpectrumX Data & Algorithm Competition (SpX-DAC). The objective was to detect the presence of active wireless transmissions in raw I/Q spectrum data collected from the 7 GHz band using a supervised machine learning approach.

The system classifies one-second I/Q recordings as:
0 = no transmitter present  
1 = active transmitter present

---

## Project Overview

This work builds a binary classification model for spectrum sensing using raw complex-valued I/Q data. The approach focuses on transforming high-dimensional signal data into a compact feature representation and applying a classical machine learning model for classification.

Key components:
- Signal preprocessing and feature extraction from raw I/Q samples
- Time-domain and frequency-domain feature engineering
- Random Forest classification model
- Evaluation on unseen spectrum data

---

## Methodology

### Feature Extraction

Each I/Q sample is converted into a 5-dimensional feature vector:

- Mean magnitude: average signal energy level
- Standard deviation of magnitude: variability of signal strength
- Total spectral power: integrated power from the PSD estimate
- Maximum spectral power: strongest frequency component
- Frequency of maximum power: location of peak energy in frequency domain

These features are computed using FFT-based power spectral density estimation.

---

### Model

A Random Forest classifier is used for classification.

Key properties:
- Ensemble of decision trees
- 100 estimators
- Bootstrap aggregation
- Gini impurity split criterion

The model was chosen for its robustness to noise and ability to capture non-linear relationships in spectral features.

---

## Results

- Local validation accuracy: ~89.6%
- Competition leaderboard accuracy: ~81%
- Strong performance across varying signal-to-noise ratios
- Low false negative rate in internal evaluation

---

## Repository Structure

SpecX_Challenge/
 │
 ├── main.py                        # Training + evaluation
 ├── RandomForestModel.py           # Model training and feature pipeline
 ├── config.py                      # Dataset paths and constants
 ├── PSD_visualizer.py              # Signal visualization tool
 ├── export_model.py                # Model serialization
 │
 ├── SpecXDataCompFinalReport.pdf   # Full technical report
 └── user_reqs.txt                  # Requirements file

---

## Key Insight

Maximum spectral power was the most important predictive feature, indicating that even low signal-to-noise transmissions contain detectable spectral peaks that can be identified using ensemble learning methods.

---

## Future Work

- Improve performance in extremely low SNR conditions
- Explore deep learning approaches such as convolutional neural networks
- Implement adaptive noise floor normalization
- Extend model to multi-class spectrum occupancy classification

---

## Acknowledgements

NSF SpectrumX Data and Algorithm Competition (SpX-DAC)

Contributors:
John Wojciechowski 
Ryan Sordillo  
