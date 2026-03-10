from pathlib import Path
import json

import numpy as np
import librosa
import librosa.feature
from tqdm import tqdm

# SETTINGS
DATA_ROOT = Path("data/raw/speech_commands_v2")
OUT_DIR = Path("data/processed")
SEED = 0

# Audio
SR = 16000
DURATION = 1.0
TARGET_LEN = int(SR * DURATION)

# Log-Mel
N_MELS = 40
N_FFT = 512
HOP_LENGTH = int(0.01 * SR)
WIN_LENGTH = int(0.025 * SR)

# Split ratios
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

BASE_KEYWORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "unknown",
    "silence",
]
NEW_KEYWORDS = ["zero", "one", "two", "three"]
CLASS_NAMES = BASE_KEYWORDS + NEW_KEYWORDS
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Regular keywords used to compute median reference for unknown/silence caps.
REGULAR_KEYWORDS = BASE_KEYWORDS[:-2] + NEW_KEYWORDS  # yes..go + zero..three (14 classes)
DOMINANCE_WARN_THRESHOLD = 0.40


def load_and_pad(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR)
    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    elif len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    return y.astype(np.float32)


def compute_logmel(audio: np.ndarray) -> np.ndarray:
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
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def generate_silence_example() -> np.ndarray:
    # Keep legacy synthetic silence behavior for compatibility.
    return np.zeros(TARGET_LEN, dtype=np.float32)


def label_name_from_word(word: str) -> str:
    if word in BASE_KEYWORDS and word not in {"unknown", "silence"}:
        return word
    if word in NEW_KEYWORDS:
        return word
    return "unknown"


def split_counts(total_count: int) -> dict:
    exact = {k: total_count * SPLIT_RATIOS[k] for k in SPLIT_ORDER}
    base = {k: int(np.floor(v)) for k, v in exact.items()}
    remainder = total_count - sum(base.values())
    if remainder > 0:
        ranked = sorted(SPLIT_ORDER, key=lambda k: (exact[k] - base[k]), reverse=True)
        for i in range(remainder):
            base[ranked[i]] += 1
    return base


def sample_list(items: list, n: int, rng: np.random.Generator) -> list:
    if n <= 0:
        return []
    if len(items) == 0:
        return []
    n = min(n, len(items))
    idx = rng.permutation(len(items))[:n]
    return [items[i] for i in idx]


def class_counts(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(y, minlength=num_classes).astype(np.int64)


def print_split_stats(split_name: str, y: np.ndarray) -> dict:
    counts = class_counts(y, len(CLASS_NAMES))
    total = int(len(y))
    out = {"total": total, "classes": {}}
    print(f"\n[STATS] {split_name}: total={total}")
    for i, name in enumerate(CLASS_NAMES):
        c = int(counts[i])
        pct = float(c / total) if total > 0 else 0.0
        out["classes"][name] = {"count": c, "pct": pct}
        print(f"  {i:2d} {name:10s} count={c:6d} pct={pct*100:6.2f}%")

    unk = int(counts[CLASS_TO_ID["unknown"]])
    sil = int(counts[CLASS_TO_ID["silence"]])
    dom = (unk + sil) / float(total) if total > 0 else 0.0
    out["unknown_silence_share"] = dom
    if dom > DOMINANCE_WARN_THRESHOLD:
        print(f"[WARN] {split_name}: unknown+silence share={dom*100:.2f}% (> {DOMINANCE_WARN_THRESHOLD*100:.0f}%).")
    return out


def main() -> None:
    print(f"Using dataset at: {DATA_ROOT.resolve()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Buckets for all non-silence classes from real wav files.
    buckets = {name: [] for name in CLASS_NAMES if name != "silence"}

    wav_paths = sorted(DATA_ROOT.rglob("*.wav"))
    print(f"Found {len(wav_paths)} wav files.")
    for wav_path in tqdm(wav_paths, desc="Processing audio"):
        if wav_path.parent.name == "_background_noise_":
            continue
        label = label_name_from_word(wav_path.parent.name)
        if label == "silence":
            continue
        audio = load_and_pad(wav_path)
        buckets[label].append(compute_logmel(audio))

    # Build splits for all real classes first.
    features = {s: [] for s in SPLIT_ORDER}
    labels = {s: [] for s in SPLIT_ORDER}
    split_source_counts = {s: {name: 0 for name in CLASS_NAMES} for s in SPLIT_ORDER}

    for class_name in CLASS_NAMES:
        if class_name == "silence":
            continue
        items = buckets[class_name]
        rng.shuffle(items)
        counts = split_counts(len(items))
        start = 0
        for split in SPLIT_ORDER:
            n = counts[split]
            part = items[start : start + n]
            start += n
            features[split].extend(part)
            label_id = CLASS_TO_ID[class_name]
            labels[split].extend([label_id] * len(part))
            split_source_counts[split][class_name] += len(part)

    # Cap unknown/silence to median of regular keyword counts per split.
    for split in SPLIT_ORDER:
        regular_counts = [split_source_counts[split][k] for k in REGULAR_KEYWORDS]
        target_cap = int(np.median(np.array(regular_counts, dtype=np.int64)))

        unk_items = [x for x, y in zip(features[split], labels[split]) if y == CLASS_TO_ID["unknown"]]
        non_unk_x = [x for x, y in zip(features[split], labels[split]) if y != CLASS_TO_ID["unknown"]]
        non_unk_y = [y for y in labels[split] if y != CLASS_TO_ID["unknown"]]
        unk_kept = sample_list(unk_items, target_cap, rng)

        # Generate synthetic silence and cap to target.
        sil_items = [compute_logmel(generate_silence_example()) for _ in range(target_cap)]
        sil_labels = [CLASS_TO_ID["silence"]] * len(sil_items)

        new_x = non_unk_x + unk_kept + sil_items
        new_y = non_unk_y + [CLASS_TO_ID["unknown"]] * len(unk_kept) + sil_labels
        p = rng.permutation(len(new_y))
        features[split] = [new_x[i] for i in p]
        labels[split] = [new_y[i] for i in p]

    arrays = {}
    for split in SPLIT_ORDER:
        x = np.stack(features[split], axis=0).astype(np.float32)
        y = np.array(labels[split], dtype=np.int64)
        arrays[split] = {"x": x, "y": y}

    # Normalize using train split statistics.
    mean = arrays["train"]["x"].mean(axis=(0, 2), keepdims=True)
    std = arrays["train"]["x"].std(axis=(0, 2), keepdims=True) + 1e-6
    for split in SPLIT_ORDER:
        arrays[split]["x"] = (arrays[split]["x"] - mean) / std

    np.savez_compressed(OUT_DIR / "logmel_train.npz", x=arrays["train"]["x"], y=arrays["train"]["y"])
    np.savez_compressed(OUT_DIR / "logmel_val.npz", x=arrays["val"]["x"], y=arrays["val"]["y"])
    np.savez_compressed(OUT_DIR / "logmel_test.npz", x=arrays["test"]["x"], y=arrays["test"]["y"])
    # Keep CL file for compatibility; aligned to training split policy.
    np.savez_compressed(OUT_DIR / "logmel_cl_train.npz", x=arrays["train"]["x"], y=arrays["train"]["y"])

    split_names = np.array(SPLIT_ORDER)
    split_class_counts = np.stack(
        [class_counts(arrays[s]["y"], len(CLASS_NAMES)) for s in SPLIT_ORDER], axis=0
    )
    np.savez_compressed(
        OUT_DIR / "logmel_stats.npz",
        mean=mean,
        std=std,
        class_names=np.array(CLASS_NAMES),
        base_keywords=np.array(BASE_KEYWORDS),
        new_keywords=np.array(NEW_KEYWORDS),
        split_names=split_names,
        split_class_counts=split_class_counts,
    )

    stats_json = {
        "seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "class_names": CLASS_NAMES,
        "splits": {},
    }
    for split in SPLIT_ORDER:
        stats_json["splits"][split] = print_split_stats(split, arrays[split]["y"])

    with (OUT_DIR / "logmel_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2, ensure_ascii=True)

    print(f"\nSaved files in: {OUT_DIR.resolve()}")
    print("Done!")


if __name__ == "__main__":
    main()
