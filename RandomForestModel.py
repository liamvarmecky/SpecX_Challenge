import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from config import TRAINING_DATA_ROOT, VLA_BRUTAL_ROOT, GROUNDTRUTH_CSV, SAMPLE_RATE, NFFT
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# extract features from IQ samples: mean magnitude and standard deviation
def extract_features(iq_samples, SAMPLE_RATE, NFFT):
    magnitude = np.abs(iq_samples)
    mean_mag = np.mean(magnitude)
    std_mag = np.std(magnitude)

    # PSD-based features
    num_blocks = len(iq_samples) // NFFT
    psd_accum = np.zeros(NFFT)
    for i in range(num_blocks):
        block = iq_samples[i * NFFT:(i+1) * NFFT]
        fft_vals = np.fft.fftshift(np.fft.fft(block))
        psd_accum += np.abs(fft_vals) ** 2

    psd_avg = psd_accum / num_blocks
    psd_db = 10 * np.log10(psd_avg + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1/SAMPLE_RATE))

    total_power = np.sum(psd_db)
    max_power = np.max(psd_db)
    max_freq = freqs[np.argmax(psd_db)]

    # median_power = np.median(psd_db)
    # snr_simple = max_power - median_power 
    # papr = max_power - (10 * np.log10(np.mean(psd_avg) + 1e-12))

    return np.array([mean_mag, std_mag, total_power, max_power, max_freq])


#load groundtruth.csv and return a label lookup dictionary
def load_groundtruth():
    df = pd.read_csv(GROUNDTRUTH_CSV)
    return dict(zip(df["filename"], df["label"]))


#returns a sorted list of all IQ .npy files in TRAINING_DATA_ROOT
def get_file_paths():
    return sorted(TRAINING_DATA_ROOT.glob("*npy"))


#builds dataset for ML
#returns: 
#   features (np.ndarray): Feature matrix of shape (num_files, num_features)
#   labels (np.ndarray): Label vector of shape (num_files,)
def build_dataset_from_csv():
    df = pd.read_csv(GROUNDTRUTH_CSV)
    features, labels = [], []

    print(f"Processing {len(df)} files listed in groundtruth.csv...")

    for _, row in df.iterrows():
        fname = row['filename']
        label = row['label']

        path_vla = VLA_BRUTAL_ROOT / fname
        path_train = TRAINING_DATA_ROOT / fname

        target_path = None
        if path_vla.exists():
            target_path = path_vla
        elif path_train.exists():
            target_path = path_train

        if target_path:
            iq_samples = np.load(target_path)
            features.append(extract_features(iq_samples, SAMPLE_RATE, NFFT))
            labels.append(label)
        else:
            # Uncomment the line below if you want to see which files are missing
            # print(f"Skipping {fname}: Not found in either directory.")
            continue

    return np.array(features), np.array(labels)


#scatter plot of features with colors for labels
def visualize_features(features, labels, x_idx = 2, y_idx = 3):
    plt.figure(figsize=(8,6))
    for lbl, color in zip([0,1], ["blue", "red"]):
        idx = labels == lbl
        plt.scatter(features[idx, x_idx], features[idx, y_idx], c=color, label=f"Label {lbl}", alpha=0.7)

    xlabel = ["Mean Magnitude", "Std Magnitude", "Total Power", "Max Power", "Max Frequency"][x_idx]
    ylabel = ["Mean Magnitude", "Std Magnitude", "Total Power", "Max Power", "Max Frequency"][y_idx]

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Feature Scatter Plot: {xlabel} vs {ylabel}")
    plt.legend()
    plt.grid(True)
    plt.show()


#split features and labels into training and testing sets
# def split_dataset(features, labels, test_size=0.2, random_state=42):
#     X_train, X_test, y_train, y_test = train_test_split(
#         features, labels, test_size=test_size, random_state=random_state, stratify=labels
#     )
#     return X_train, X_test, y_train, y_test


#train a random forest classifier on the training data
def train_random_forest(X_train, y_train, n_estimators=100, random_state=123):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


#evaluate the trained model using accuracy and confusion matrix
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return acc, cm


def main():
    #Load everything using the CSV
    X, y = build_dataset_from_csv()
    print(f"Dataset built. Total samples: {len(X)}")

    #Split into Training (80%) and Testing (20%)
    #This ensures the model tests on data it has never seen before
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100, stratify=y
    )

    #Train
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=100)
    clf.fit(X_train, y_train)

    #Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nEvaluation Results:")
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    #Check Feature Importance
    importances = clf.feature_importances_
    feature_names = ["Mean Mag", "Std Mag", "Total Power", "Max Power", "Max Freq"]
    
    print("\nFeature Importances:")
    for name, imp in zip(feature_names, importances):
        print(f"{name}: {imp:.4f}")


if __name__ == "__main__":
    main()