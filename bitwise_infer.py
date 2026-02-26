import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view

from bnn_layers import BinaryActivation, BinaryConv2D, BinaryDense
from train_bnn import build_bnn_model


# +1 => bit 1, -1 => bit 0
# dot = 2*popcount(xnor) - n_bits
POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

CASE_NAMES = [
    "interior",
    "top",
    "bottom",
    "left",
    "right",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]
CASE_TO_OFFSETS = {name: None for name in CASE_NAMES}


def case_offsets(case_name):
    """Return valid (ky, kx) positions for a 3x3 kernel under SAME border handling."""
    all_coords = [(ky, kx) for ky in range(3) for kx in range(3)]
    if case_name == "interior":
        return all_coords

    valid = []
    for ky, kx in all_coords:
        dy = ky - 1
        dx = kx - 1
        ok_y = True
        ok_x = True

        if "top" in case_name and dy < 0:
            ok_y = False
        if "bottom" in case_name and dy > 0:
            ok_y = False
        if "left" in case_name and dx < 0:
            ok_x = False
        if "right" in case_name and dx > 0:
            ok_x = False

        if ok_y and ok_x:
            valid.append((ky, kx))
    return valid


for _name in CASE_NAMES:
    CASE_TO_OFFSETS[_name] = case_offsets(_name)


def position_case(y, x, h, w):
    top = y == 0
    bottom = y == (h - 1)
    left = x == 0
    right = x == (w - 1)

    if top and left:
        return "top_left"
    if top and right:
        return "top_right"
    if bottom and left:
        return "bottom_left"
    if bottom and right:
        return "bottom_right"
    if top:
        return "top"
    if bottom:
        return "bottom"
    if left:
        return "left"
    if right:
        return "right"
    return "interior"


def pack_bits_1d(bits_1d):
    return np.packbits(bits_1d.astype(np.uint8), bitorder="little")


def xnor_popcount_dot_vec(packed_input, packed_weights, nbits):
    """Compute vector of binary dot products between one packed input and packed weights.

    packed_input:  (num_bytes,)
    packed_weights:(out_dim, num_bytes)
    returns dot:   (out_dim,), where dot = 2*ones - nbits
    """
    xnor = np.bitwise_not(np.bitwise_xor(packed_weights, packed_input[None, :]))

    # Mask trailing pad bits so only true nbits are counted.
    rem = int(nbits) % 8
    if rem != 0:
        mask = np.uint8((1 << rem) - 1)
        xnor[:, -1] = np.bitwise_and(xnor[:, -1], mask)

    ones = POPCOUNT_LUT[xnor].sum(axis=1, dtype=np.int32)
    return (2 * ones - int(nbits)).astype(np.float32)


def xnor_popcount_dot_mat(packed_inputs, packed_weights, nbits):
    """Batched binary dot product via XNOR+popcount.

    packed_inputs:  (n_patches, num_bytes)
    packed_weights: (out_dim,   num_bytes)
    returns:        (n_patches, out_dim)

    Mapping:
    +1 => bit 1, -1 => bit 0
    dot = 2*popcount(xnor) - n_bits
    """
    xnor = np.bitwise_not(np.bitwise_xor(packed_inputs[:, None, :], packed_weights[None, :, :]))

    # Mask trailing pad bits in the final byte so only real bits contribute.
    rem = int(nbits) % 8
    if rem != 0:
        mask = np.uint8((1 << rem) - 1)
        xnor[:, :, -1] = np.bitwise_and(xnor[:, :, -1], mask)

    ones = POPCOUNT_LUT[xnor].sum(axis=2, dtype=np.int32)
    return (2 * ones - int(nbits)).astype(np.float32)


def bn_sign_from_threshold(z, threshold, flip, gamma, beta):
    """Apply folded BN sign decision with sign(0)=>+1."""
    out = np.zeros_like(z, dtype=bool)

    pos = gamma > 0.0
    neg = gamma < 0.0
    zer = gamma == 0.0

    out[pos] = z[pos] >= threshold[pos]
    out[neg] = z[neg] <= threshold[neg]
    out[zer] = beta[zer] >= 0.0

    # flip is kept in artifacts per request; gamma sign drives the compare direction.
    _ = flip
    return out


def bn_sign_from_threshold_mat(z, threshold, gamma, beta):
    """Vectorized folded-BN sign decision for a matrix z (n_patches, out_ch)."""
    out = np.zeros_like(z, dtype=bool)

    pos = gamma > 0.0
    neg = gamma < 0.0
    zer = gamma == 0.0

    if np.any(pos):
        out[:, pos] = z[:, pos] >= threshold[pos]
    if np.any(neg):
        out[:, neg] = z[:, neg] <= threshold[neg]
    if np.any(zer):
        out[:, zer] = beta[zer] >= 0.0
    return out


def majority_pool2x2(bits_hwc):
    """2x2 majority-vote pooling for binary activations (popcount >= 2 of 4)."""
    h, w, c = bits_hwc.shape
    out_h = h // 2
    out_w = w // 2

    cropped = bits_hwc[: out_h * 2, : out_w * 2, :]
    votes = cropped.reshape(out_h, 2, out_w, 2, c).sum(axis=(1, 3))
    return votes >= 2


def conv3x3_bitwise_same(input_bits, prefix, art):
    """Bitwise 3x3 SAME binary convolution with border-case valid-position masking.

    SAME OOB handling here ignores OOB entries (equivalent to multiply by 0).
    """
    h, w, in_ch = input_bits.shape

    alpha = art[f"{prefix}_alpha"]
    out_ch = int(alpha.shape[0])

    threshold = art[f"{prefix}_bn_threshold"]
    flip = art[f"{prefix}_bn_flip"].astype(bool)
    gamma = art[f"{prefix}_bn_gamma"]
    beta = art[f"{prefix}_bn_beta"]

    packed_by_case = {name: art[f"packed_{prefix}_{name}"] for name in CASE_NAMES}
    nbits_by_case = {name: int(art[f"{prefix}_nbits_{name}"]) for name in CASE_NAMES}

    out_bits = np.zeros((h, w, out_ch), dtype=bool)

    for y in range(h):
        for x in range(w):
            case = position_case(y, x, h, w)
            coords = CASE_TO_OFFSETS[case]

            flat_parts = []
            for ky, kx in coords:
                iy = y + (ky - 1)
                ix = x + (kx - 1)
                flat_parts.append(input_bits[iy, ix, :])
            patch_bits = np.concatenate(flat_parts, axis=0)
            packed_patch = pack_bits_1d(patch_bits)

            packed_w = packed_by_case[case]
            nbits = nbits_by_case[case]

            dot = xnor_popcount_dot_vec(packed_patch, packed_w, nbits)
            z = dot * alpha
            out_bits[y, x, :] = bn_sign_from_threshold(z, threshold, flip, gamma, beta)

    return out_bits


def conv3x3_bitwise_same_fast(input_bits, prefix, art):
    """Fast bitwise 3x3 SAME conv.

    Strategy:
    - Vectorize interior pixels (1:-1, 1:-1) with sliding windows + batched XNOR+popcount.
    - Keep exact border-case handling in a small loop to preserve SAME-zero behavior.
    """
    h, w, in_ch = input_bits.shape

    alpha = art[f"{prefix}_alpha"]
    out_ch = int(alpha.shape[0])

    threshold = art[f"{prefix}_bn_threshold"]
    flip = art[f"{prefix}_bn_flip"].astype(bool)
    gamma = art[f"{prefix}_bn_gamma"]
    beta = art[f"{prefix}_bn_beta"]

    packed_by_case = {name: art[f"packed_{prefix}_{name}"] for name in CASE_NAMES}
    nbits_by_case = {name: int(art[f"{prefix}_nbits_{name}"]) for name in CASE_NAMES}

    out_bits = np.zeros((h, w, out_ch), dtype=bool)

    # Interior only exists for h,w >= 3.
    if h >= 3 and w >= 3:
        packed_w_int = packed_by_case["interior"]
        nbits_int = nbits_by_case["interior"]

        # Extract all 3x3 interior patches in one shot.
        patches = sliding_window_view(input_bits.astype(np.uint8), (3, 3), axis=(0, 1))
        # NumPy layout can be either (h-2,w-2,3,3,c) or (h-2,w-2,c,3,3) by version.
        if patches.shape[-1] == in_ch:
            patches = patches.reshape(-1, 9 * in_ch)
        elif patches.shape[2] == in_ch:
            patches = np.transpose(patches, (0, 1, 3, 4, 2)).reshape(-1, 9 * in_ch)
        else:
            raise RuntimeError(f"Unexpected sliding_window_view shape: {patches.shape}")

        packed_patches = np.packbits(patches, axis=1, bitorder="little")
        dots = xnor_popcount_dot_mat(packed_patches, packed_w_int, nbits_int)
        z = dots * alpha[None, :]
        interior_bits = bn_sign_from_threshold_mat(z, threshold, gamma, beta).reshape(h - 2, w - 2, out_ch)
        out_bits[1:-1, 1:-1, :] = interior_bits

    # Border pixels only. SAME OOB handling stays exact using per-case packed weights.
    border_coords = []
    if h > 0:
        for x in range(w):
            border_coords.append((0, x))
    if h > 1:
        for x in range(w):
            border_coords.append((h - 1, x))
    if w > 0:
        for y in range(1, max(h - 1, 1)):
            border_coords.append((y, 0))
    if w > 1:
        for y in range(1, max(h - 1, 1)):
            border_coords.append((y, w - 1))

    for y, x in border_coords:
        case = position_case(y, x, h, w)
        coords = CASE_TO_OFFSETS[case]

        flat_parts = []
        for ky, kx in coords:
            iy = y + (ky - 1)
            ix = x + (kx - 1)
            flat_parts.append(input_bits[iy, ix, :])
        patch_bits = np.concatenate(flat_parts, axis=0)
        packed_patch = pack_bits_1d(patch_bits)

        packed_w = packed_by_case[case]
        nbits = nbits_by_case[case]

        dot = xnor_popcount_dot_vec(packed_patch, packed_w, nbits)
        z = dot * alpha
        out_bits[y, x, :] = bn_sign_from_threshold(z, threshold, flip, gamma, beta)

    return out_bits


def validate_fast_conv_matches_reference(art, num_trials=2, seed=0):
    """Quick correctness check: reference conv vs fast conv on random boolean inputs."""
    rng = np.random.default_rng(seed)
    total_mismatch = 0

    for prefix in ("bconv2", "bconv3"):
        in_ch = int(art[f"{prefix}_in_ch"])
        # Small synthetic maps keep validation quick.
        for trial in range(num_trials):
            h = int(rng.integers(6, 12))
            w = int(rng.integers(7, 14))
            test_bits = rng.integers(0, 2, size=(h, w, in_ch), dtype=np.uint8).astype(bool)

            ref = conv3x3_bitwise_same(test_bits, prefix, art)
            fast = conv3x3_bitwise_same_fast(test_bits, prefix, art)
            mismatch = int(np.count_nonzero(ref != fast))
            total_mismatch += mismatch

            print(f"[CHECK] {prefix} trial {trial + 1}: mismatches={mismatch}")
            if mismatch != 0:
                raise AssertionError(f"Fast conv mismatch for {prefix} trial {trial + 1}: {mismatch}")

    print(f"[CHECK] Fast conv validation passed. Total mismatches: {total_mismatch}")


def bdense_float(flat_bits, bdense_w, bdense_b, art):
    # Replace bitwise BinaryDense runtime with plain float Dense runtime.
    # Binary activations are mapped back to {-1, +1} before matmul.
    flat_pm1 = np.where(flat_bits, 1.0, -1.0).astype(np.float32)
    x = flat_pm1 @ bdense_w + bdense_b

    # BN in float
    x = art["bn_dense_gamma"] * (x - art["bn_dense_mean"]) / art["bn_dense_std"] + art["bn_dense_beta"]
    x = np.maximum(x, 0.0)  # ReLU

    # Final softmax Dense in float
    logits = x @ art["softmax_w"] + art["softmax_b"]
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    return probs.astype(np.float32)


def hybrid_predict_one(sample_ba1, bdense_w, bdense_b, art):
    """Run bitwise path from ba1 onward.

    sample_ba1: (40, 101, 16) from TF first block (conv1 + BN + BinaryActivation).
    """
    ba1_bits = sample_ba1 >= 0.0

    # bconv2 + BN(threshold) + BinaryActivation
    x = conv3x3_bitwise_same_fast(ba1_bits, "bconv2", art)

    # AvgPool2D replacement: 2x2 majority vote
    x = majority_pool2x2(x)  # (20, 50, 32)

    # bconv3 + BN(threshold) + BinaryActivation
    x = conv3x3_bitwise_same_fast(x, "bconv3", art)

    # AvgPool2D replacement: 2x2 majority vote
    x = majority_pool2x2(x)  # (10, 25, 64)

    flat_bits = x.reshape(-1)
    return bdense_float(flat_bits, bdense_w, bdense_b, art)


def build_ba1_model(model):
    ba_layers = [l for l in model.layers if isinstance(l, BinaryActivation)]
    if len(ba_layers) < 1:
        raise RuntimeError("Could not find BinaryActivation layers")
    return tf.keras.Model(inputs=model.input, outputs=ba_layers[0].output)


def load_test_data(test_path):
    data = np.load(test_path)
    x_test = data["x"][..., np.newaxis].astype(np.float32)
    y_test = data["y"].astype(np.int32)
    return x_test, y_test


def benchmark_full_tf(model, x, warmup=20):
    n = x.shape[0]
    wu = min(warmup, n)

    for i in range(wu):
        _ = model(x[i : i + 1], training=False)

    t0 = time.perf_counter()
    for i in range(n):
        _ = model(x[i : i + 1], training=False)
    t1 = time.perf_counter()

    return ((t1 - t0) * 1000.0) / n


def benchmark_hybrid(ba1_model, x, bdense_w, bdense_b, art, warmup=20):
    n = x.shape[0]
    wu = min(warmup, n)

    for i in range(wu):
        ba1 = ba1_model(x[i : i + 1], training=False).numpy()[0]
        _ = hybrid_predict_one(ba1, bdense_w, bdense_b, art)

    t0 = time.perf_counter()
    for i in range(n):
        ba1 = ba1_model(x[i : i + 1], training=False).numpy()[0]
        _ = hybrid_predict_one(ba1, bdense_w, bdense_b, art)
    t1 = time.perf_counter()

    return ((t1 - t0) * 1000.0) / n


def benchmark_hybrid_backend_only(ba1_batch, bdense_w, bdense_b, art, warmup=20):
    """Benchmark only bitwise backend by reusing precomputed ba1 outputs."""
    n = ba1_batch.shape[0]
    wu = min(warmup, n)

    for i in range(wu):
        _ = hybrid_predict_one(ba1_batch[i], bdense_w, bdense_b, art)

    t0 = time.perf_counter()
    for i in range(n):
        _ = hybrid_predict_one(ba1_batch[i], bdense_w, bdense_b, art)
    t1 = time.perf_counter()

    return ((t1 - t0) * 1000.0) / n


def main():
    parser = argparse.ArgumentParser(description="Hybrid TF + bitwise inference runtime")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--weights_path", type=str, default="models/bnn_kws_v2_registered.weights.h5")
    parser.add_argument("--artifacts", type=str, default="models/bitwise_artifacts.npz")
    parser.add_argument("--test_npz", type=str, default="data/processed/logmel_test.npz")
    parser.add_argument("--stats_npz", type=str, default="data/processed/logmel_stats.npz")
    args = parser.parse_args()

    weights_path = Path(args.weights_path)
    art_path = Path(args.artifacts)
    test_path = Path(args.test_npz)
    stats_path = Path(args.stats_npz)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing weights file: {weights_path}. "
            "Run train_bnn.py first to generate bnn_kws_v2_registered.weights.h5."
        )
    if not art_path.exists():
        raise FileNotFoundError(f"Missing artifact file: {art_path}. Run export_bitwise.py first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test npz file: {test_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats npz file: {stats_path}")

    x_test, y_test = load_test_data(test_path)
    n = min(args.num_samples, x_test.shape[0])
    x = x_test[:n]
    y = y_test[:n]

    stats_data = np.load(stats_path, allow_pickle=True)
    num_classes = len(stats_data["class_names"])
    model = build_bnn_model(num_classes)
    model.load_weights(weights_path)
    ba1_model = build_ba1_model(model)
    bdense_layers = [l for l in model.layers if isinstance(l, BinaryDense)]
    if len(bdense_layers) != 1:
        raise RuntimeError(f"Expected exactly 1 BinaryDense layer, found {len(bdense_layers)}")
    bdense_w = bdense_layers[0].kernel.numpy().astype(np.float32)
    if bdense_layers[0].bias is None:
        bdense_b = np.zeros((bdense_w.shape[1],), dtype=np.float32)
    else:
        bdense_b = bdense_layers[0].bias.numpy().astype(np.float32)
    with np.load(art_path) as f:
        art = {k: f[k] for k in f.files}

    validate_fast_conv_matches_reference(art)

    # Prediction agreement
    tf_probs = model(x, training=False).numpy()
    tf_pred = np.argmax(tf_probs, axis=1)

    ba1_batch = ba1_model(x, training=False).numpy()
    hybrid_probs = np.zeros_like(tf_probs, dtype=np.float32)
    for i in range(n):
        hybrid_probs[i] = hybrid_predict_one(ba1_batch[i], bdense_w, bdense_b, art)
    hybrid_pred = np.argmax(hybrid_probs, axis=1)

    mismatch = np.mean(tf_pred != hybrid_pred)
    hybrid_acc = np.mean(hybrid_pred == y)
    tf_acc = np.mean(tf_pred == y)

    print(f"[INFO] Samples evaluated: {n}")
    print(f"[RESULT] TF acc on subset: {tf_acc * 100:.2f}%")
    print(f"[RESULT] Hybrid acc on subset: {hybrid_acc * 100:.2f}%")
    print(f"[RESULT] TF vs Hybrid mismatch rate: {mismatch * 100:.2f}%")

    # Runtime benchmark
    tf_ms = benchmark_full_tf(model, x)
    hybrid_ms = benchmark_hybrid(ba1_model, x, bdense_w, bdense_b, art)
    hybrid_backend_ms = benchmark_hybrid_backend_only(ba1_batch, bdense_w, bdense_b, art)

    print(f"[BENCH] Full TF inference: {tf_ms:.3f} ms/sample")
    print(f"[BENCH] Hybrid bitwise inference: {hybrid_ms:.3f} ms/sample")
    print(f"[BENCH] Hybrid bitwise backend only: {hybrid_backend_ms:.3f} ms/sample")

    # Memory footprint
    float_bytes = sum(w.nbytes for w in model.get_weights())
    packed_bytes = 0
    for key, value in art.items():
        if key.startswith("packed_"):
            packed_bytes += value.nbytes

    ratio = float_bytes / packed_bytes if packed_bytes > 0 else np.inf

    print(f"[MEM] Float weights total bytes: {float_bytes}")
    print(f"[MEM] Bitpacked weights total bytes: {packed_bytes}")
    print(f"[MEM] Compression ratio (float/bitpacked): {ratio:.3f}x")


if __name__ == "__main__":
    main()
