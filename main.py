import numpy as np
import joblib
from pathlib import Path

#load the model
MODEL = joblib.load('model.pkl')

SAMPLE_RATE = 10_000_000
NFFT = 4096

def extract_features(iq_samples):
    """
    Extracts time-domain statistics and spectral density metrics to create a 
    5-point feature vector for binary transmitter classification.
    """
    magnitude = np.abs(iq_samples)
    mean_mag = np.mean(magnitude)
    std_mag = np.std(magnitude)

    #PSD calculation
    num_blocks = len(iq_samples) // NFFT
    psd_accum = np.zeros(NFFT)
    for i in range(num_blocks):
        block = iq_samples[i * NFFT:(i+1) * NFFT]
        if len(block) < NFFT: 
            continue 
        fft_vals = np.fft.fftshift(np.fft.fft(block))
        psd_accum += np.abs(fft_vals) ** 2
    
    psd_avg = psd_accum / num_blocks
    psd_db = 10 * np.log10(psd_avg + 1e-12)
    
    #Feature Vector Construction 
    total_power = np.sum(psd_db)
    max_power = np.max(psd_db)
    
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1/SAMPLE_RATE))
    max_freq = freqs[np.argmax(psd_db)]

    # median_power = np.median(psd_db)
    # snr_simple = max_power - median_power
    # papr = max_power - (10 * np.log10(np.mean(psd_avg) + 1e-12))

    #Return as a 2D array for scikit-learn (1 sample, 5 features)
    return np.array([[mean_mag, std_mag, total_power, max_power, max_freq]])

def evaluate(filename):
    try:
        #The server passes the full path to an .npy file
        iq_samples = np.load(filename)
        
        #Get the 5-feature vector
        features = extract_features(iq_samples)
        
        #Get the prediction (0 or 1)
        prediction = MODEL.predict(features)[0]
        
        return int(prediction)
    except Exception:
        #If a file is corrupt or empty, default to 'No Transmitter'
        return 0