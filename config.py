from pathlib import Path

TRAINING_DATA_ROOT = Path(r"D:\SpectrumXDataSet\files\kdoke@hamilton.edu\trainingData")

VLA_BRUTAL_ROOT = Path(r"D:\SpectrumXDataSet\files\kdoke@hamilton.edu\VLA_brutal")

GROUNDTRUTH_CSV = VLA_BRUTAL_ROOT / "groundtruth.csv"

SAMPLE_RATE = 10_000_000
NFFT = 4096