from pathlib import Path
import numpy as np
import librosa
import librosa.feature
from tqdm import tqdm

# SETTINGS

# Path to your dataset root
DATA_ROOT = Path("data/raw/speech_commands_v2")
OUT_DIR = Path("data/processed")

# Audio
SR = 16000        # sample rate (Hz)
DURATION = 1.0    # seconds
TARGET_LEN = int(SR * DURATION)

# Log-Mel
N_MELS = 40
N_FFT = 512
HOP_LENGTH = int(0.01 * SR)   # 10 ms
WIN_LENGTH = int(0.025 * SR)  # 25 ms

# 10 Classes
KEYWORDS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go",
]

CLASS_NAMES = KEYWORDS + ["unknown", "silence"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


# FUNCTIONS

def load_split_lists():
    val_path = DATA_ROOT / "validation_list.txt"
    test_path = DATA_ROOT / "testing_list.txt"

    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("validation_list.txt or testing_list.txt not found.")

    val_list = val_path.read_text().strip().splitlines()
    test_list = test_path.read_text().strip().splitlines()

    return set(val_list), set(test_list)


def get_split(rel_path, val_set, test_set):
    #Return 'train', 'val', or 'test' for a given file.
    if rel_path in val_set:
        return "val"
    elif rel_path in test_set:
        return "test"
    else:
        return "train"

def load_and_pad(path):
    #Load audio, resample, and pad/trim to exactly 1 second.
    y, sr = librosa.load(path, sr=SR)
    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    elif len(y) < TARGET_LEN:
        pad_width = TARGET_LEN - len(y)
        y = np.pad(y, (0, pad_width))
    return y.astype(np.float32)


def compute_logmel(audio):
    #Compute log-Mel spectrogram.
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        window="hann",
        fmin=0.0,
        fmax=SR / 2,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def generate_silence_example():
    #Return one second of silence.
    return np.zeros(TARGET_LEN, dtype=np.float32)


# -----MAIN LOGIC--------

def main():
    print(f"Using dataset at: {DATA_ROOT.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_set, test_set = load_split_lists()

    # We store data for each split in lists first
    features = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}

    # 1) Process all wav files except background noise
    wav_paths = sorted(DATA_ROOT.rglob("*.wav"))

    print(f"Found {len(wav_paths)} wav files.")

    for wav_path in tqdm(wav_paths, desc="Processing audio"):
        # Skip background noise folder for now
        if wav_path.parent.name == "_background_noise_":
            continue

        # e.g. 'yes/0a7c2a8d_nohash_0.wav'
        rel_path = wav_path.relative_to(DATA_ROOT).as_posix()

        split = get_split(rel_path, val_set, test_set)

        word = wav_path.parent.name
        if word in KEYWORDS:
            label_name = word
        else:
            label_name = "unknown"

        label_id = CLASS_TO_ID[label_name]

        audio = load_and_pad(wav_path)
        logmel = compute_logmel(audio)  # shape (40, T)

        features[split].append(logmel)
        labels[split].append(label_id)

    # 2) Synthetic silence examples (1000 per split)
    silence_per_split = 1000
    print(f"Adding {silence_per_split} silence examples per split...")

    for split in ["train", "val", "test"]:
        for _ in range(silence_per_split):
            audio = generate_silence_example()
            logmel = compute_logmel(audio)
            features[split].append(logmel)
            labels[split].append(CLASS_TO_ID["silence"])

    # 3) Turn lists into arrays
    x_train = np.stack(features["train"], axis=0)
    y_train = np.array(labels["train"], dtype=np.int64)

    x_val = np.stack(features["val"], axis=0)
    y_val = np.array(labels["val"], dtype=np.int64)

    x_test = np.stack(features["test"], axis=0)
    y_test = np.array(labels["test"], dtype=np.int64)

    print("Shapes BEFORE normalisation:")
    print("  Train:", x_train.shape)
    print("  Val:  ", x_val.shape)
    print("  Test: ", x_test.shape)

    # 4) Compute mean/std from TRAIN only (for normalisation)
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-6

    # 5) Apply normalisation
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    # 6) Save everything to .npz files
    np.savez_compressed(OUT_DIR / "logmel_train.npz", x=x_train, y=y_train)
    np.savez_compressed(OUT_DIR / "logmel_val.npz", x=x_val, y=y_val)
    np.savez_compressed(OUT_DIR / "logmel_test.npz", x=x_test, y=y_test)
    np.savez_compressed(OUT_DIR / "logmel_stats.npz",
                        mean=mean, std=std, class_names=np.array(CLASS_NAMES))

    print("Saved files in:", OUT_DIR.resolve())
    print("Done!")

if __name__ == "__main__":
    main()
