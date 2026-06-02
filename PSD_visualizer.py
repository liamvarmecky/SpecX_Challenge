import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from config import TRAINING_DATA_ROOT, GROUNDTRUTH_CSV, SAMPLE_RATE, NFFT


def get_file_paths():
    iq_files = sorted(TRAINING_DATA_ROOT.glob("*.npy"))
    return iq_files


#compute averaged power spectral density(PSD) from IQ samples
def compute_psd(x, sample_rate = SAMPLE_RATE, nfft = NFFT):
    num_blocks = len(x) // nfft
    psd_accum = np.zeros(nfft)

    for i in range(num_blocks):
        block = x[i * nfft : (i + 1) * nfft]
        fft_vals = np.fft.fftshift(np.fft.fft(block))
        psd_accum += np.abs(fft_vals) ** 2

    psd_avg = psd_accum / num_blocks
    psd_db = 10 * np.log10(psd_avg + 1e-12)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d = 1 / sample_rate))

    return freqs, psd_db


#power spectral density visualization
def plot_psd(freqs, psd_db, title="PSD"):
    plt.figure(figsize=(10, 4))
    plt.plot(freqs / 1e6, psd_db)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_groundtruth():
    df = pd.read_csv(GROUNDTRUTH_CSV)
    return df


#returns a dictionary: {filename : 0 or 1}
def build_label_lookup(df):
    return dict(zip(df["filename"], df["label"]))


def plot_iq_file_with_label(file_path, label_lookup, sample_rate = SAMPLE_RATE, nfft = NFFT):
    """
    Load an IQ .npy file, compute its PSD, look up the transmitter label,
    and plot the PSD with filename and label in the title.
    
    Parameters:
        file_path (Path or str): Path to the IQ .npy file
        label_lookup (dict): filename -> 0/1 label
        sample_rate (int): sample rate in Hz
        nfft (int): FFT size for PSD computation
    """
    # Load IQ data
    x = np.load(file_path)

    # Compute PSD
    freqs, psd_db = compute_psd(x, sample_rate=sample_rate, nfft=nfft)

    # Lookup label
    filename = Path(file_path).name
    label = label_lookup.get(filename, None)
    label_str = "Transmitter Present" if label == 1 else "No Transmitter"

    # Plot PSD
    plot_psd(freqs, psd_db, title=f"{filename} | {label_str}")


def main():
    #load groundtruth and build lookup
    df = load_groundtruth()
    label_lookup = build_label_lookup(df)

    #get IQ files
    files = get_file_paths()
    currentfile = 60

    #plot PSD + label
    plot_iq_file_with_label(files[currentfile], label_lookup)


if __name__ == "__main__":
    main()
