import numpy as np
from pathlib import Path
import tensorflow as tf
from logging import getLogger, INFO, StreamHandler, Formatter
from bnn_layers import BinaryActivation, BinaryConv2D, BinaryDense
from train_bnn import build_bnn_model


logger = getLogger(__name__)
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


def case_offsets(case_name):
    """Return valid (ky, kx) positions for a 3x3 kernel under SAME border handling.

    OOB positions are ignored (equivalent to multiplying by 0), so each case keeps only
    in-bounds positions.
    """
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


def fold_bn_to_sign_threshold(gamma, beta, moving_mean, moving_var, epsilon):
    """Fold BN into sign compare parameters.

    BN(z) = gamma*(z-mean)/sqrt(var+eps) + beta.
    sign(BN(z)) with sign(0)=>+1 reduces to:
    - gamma > 0: z >= threshold
    - gamma < 0: z <= threshold
    - gamma == 0: output is constant sign(beta)

    We store threshold + flip (gamma<0), and keep beta/gamma for the gamma==0 edge.
    """
    std = np.sqrt(moving_var + epsilon).astype(np.float32)
    gamma = gamma.astype(np.float32)
    beta = beta.astype(np.float32)
    moving_mean = moving_mean.astype(np.float32)

    threshold = np.zeros_like(gamma, dtype=np.float32)
    nz = gamma != 0.0
    threshold[nz] = moving_mean[nz] - (beta[nz] * std[nz] / gamma[nz])

    flip = (gamma < 0.0).astype(np.uint8)
    return threshold, flip, gamma, beta, moving_mean, std


def pack_binary_conv_weights(w_fp):
    """Pack BinaryConv2D weights by border case.

    Mapping:
    +1 => bit 1, -1 => bit 0
    dot = 2*popcount(xnor) - n_bits
    """
    # w_fp shape: (3, 3, in_ch, out_ch)
    if w_fp.shape[0] != 3 or w_fp.shape[1] != 3:
        raise ValueError(f"Expected 3x3 kernel, got shape {w_fp.shape}")

    in_ch = w_fp.shape[2]
    out_ch = w_fp.shape[3]
    alpha = np.mean(np.abs(w_fp), axis=(0, 1, 2)).astype(np.float32)

    packed = {}
    nbits = {}

    for case in CASE_NAMES:
        coords = case_offsets(case)

        # Gather valid kernel positions, then flatten per output channel.
        # bits shape after transpose/reshape: (out_ch, valid_k * in_ch)
        gathered = np.stack([w_fp[ky, kx, :, :] for (ky, kx) in coords], axis=0)
        bits = (gathered >= 0.0).astype(np.uint8)
        bits = np.transpose(bits, (2, 0, 1)).reshape(out_ch, -1)

        packed_case = np.packbits(bits, axis=-1, bitorder="little")
        packed[case] = packed_case
        nbits[case] = np.int32(bits.shape[1])

    return packed, nbits, alpha, np.int32(in_ch), np.int32(out_ch)


def pack_binary_dense_weights(w_fp):
    """Pack BinaryDense weights for bitwise XNOR+popcount runtime."""
    # w_fp shape: (in_dim, out_dim)
    in_dim, out_dim = w_fp.shape
    alpha = np.mean(np.abs(w_fp), axis=0).astype(np.float32)

    bits = (w_fp >= 0.0).astype(np.uint8).T  # (out_dim, in_dim)
    packed = np.packbits(bits, axis=-1, bitorder="little")
    nbits = np.int32(in_dim)

    return packed, nbits, alpha, np.int32(in_dim), np.int32(out_dim)


def collect_layers(model):
    convs = [l for l in model.layers if isinstance(l, BinaryConv2D)]
    bns = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    bdenses = [l for l in model.layers if isinstance(l, BinaryDense)]
    denses = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

    if len(convs) != 2:
        raise RuntimeError(f"Expected 2 BinaryConv2D layers, found {len(convs)}")
    if len(bdenses) != 1:
        raise RuntimeError(f"Expected 1 BinaryDense layer, found {len(bdenses)}")
    if len(denses) < 1:
        raise RuntimeError("Expected final softmax Dense layer")
    if len(bns) < 4:
        raise RuntimeError(f"Expected at least 4 BatchNorm layers, found {len(bns)}")

    # BN order in this model:
    # bn0 after conv1, bn1 after bconv2, bn2 after bconv3, bn3 after bdense
    return {
        "bconv2": convs[0],
        "bconv3": convs[1],
        "bn_bconv2": bns[1],
        "bn_bconv3": bns[2],
        "bdense": bdenses[0],
        "bn_bdense": bns[3],
        "softmax": denses[-1],
    }


def main():
    weights_path = Path("models/bnn_kws_v2_registered.weights.h5")
    out_path = Path("models/bitwise_artifacts.npz")
    stats_path = Path("data/processed/logmel_stats.npz")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. "
            "Run train_bnn.py first to generate bnn_kws_v2_registered.weights.h5."
        )
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found at {stats_path}")

    stats_data = np.load(stats_path, allow_pickle=True)
    num_classes = len(stats_data["class_names"])
    model = build_bnn_model(num_classes)
    model.load_weights(weights_path)
    layers_map = collect_layers(model)

    artifacts = {}

    # Binary conv 2
    bconv2_w = layers_map["bconv2"].w_fp.numpy().astype(np.float32)
    packed2, nbits2, alpha2, in2, out2 = pack_binary_conv_weights(bconv2_w)
    artifacts["bconv2_alpha"] = alpha2
    artifacts["bconv2_in_ch"] = in2
    artifacts["bconv2_out_ch"] = out2
    for case in CASE_NAMES:
        artifacts[f"packed_bconv2_{case}"] = packed2[case]
        artifacts[f"bconv2_nbits_{case}"] = nbits2[case]

    g, b, m, v = [x.astype(np.float32) for x in layers_map["bn_bconv2"].get_weights()]
    t, f, gg, bb, mm, ss = fold_bn_to_sign_threshold(g, b, m, v, layers_map["bn_bconv2"].epsilon)
    artifacts["bconv2_bn_threshold"] = t
    artifacts["bconv2_bn_flip"] = f
    artifacts["bconv2_bn_gamma"] = gg
    artifacts["bconv2_bn_beta"] = bb
    artifacts["bconv2_bn_mean"] = mm
    artifacts["bconv2_bn_std"] = ss

    # Binary conv 3
    bconv3_w = layers_map["bconv3"].w_fp.numpy().astype(np.float32)
    packed3, nbits3, alpha3, in3, out3 = pack_binary_conv_weights(bconv3_w)
    artifacts["bconv3_alpha"] = alpha3
    artifacts["bconv3_in_ch"] = in3
    artifacts["bconv3_out_ch"] = out3
    for case in CASE_NAMES:
        artifacts[f"packed_bconv3_{case}"] = packed3[case]
        artifacts[f"bconv3_nbits_{case}"] = nbits3[case]

    g, b, m, v = [x.astype(np.float32) for x in layers_map["bn_bconv3"].get_weights()]
    t, f, gg, bb, mm, ss = fold_bn_to_sign_threshold(g, b, m, v, layers_map["bn_bconv3"].epsilon)
    artifacts["bconv3_bn_threshold"] = t
    artifacts["bconv3_bn_flip"] = f
    artifacts["bconv3_bn_gamma"] = gg
    artifacts["bconv3_bn_beta"] = bb
    artifacts["bconv3_bn_mean"] = mm
    artifacts["bconv3_bn_std"] = ss

    # Binary dense
    bdense_w = layers_map["bdense"].kernel.numpy().astype(np.float32)
    bdense_b = (
        layers_map["bdense"].bias.numpy().astype(np.float32)
        if layers_map["bdense"].bias is not None
        else np.zeros((bdense_w.shape[1],), dtype=np.float32)
    )
    packed_d, nbits_d, alpha_d, in_d, out_d = pack_binary_dense_weights(bdense_w)
    artifacts["packed_bdense"] = packed_d
    artifacts["bdense_nbits"] = nbits_d
    artifacts["bdense_alpha"] = alpha_d
    artifacts["bdense_bias"] = bdense_b
    artifacts["bdense_in_dim"] = in_d
    artifacts["bdense_out_dim"] = out_d

    g, b, m, v = [x.astype(np.float32) for x in layers_map["bn_bdense"].get_weights()]
    std = np.sqrt(v + layers_map["bn_bdense"].epsilon).astype(np.float32)
    artifacts["bn_dense_gamma"] = g
    artifacts["bn_dense_beta"] = b
    artifacts["bn_dense_mean"] = m
    artifacts["bn_dense_std"] = std

    # Final softmax dense kept in float
    softmax_w, softmax_b = layers_map["softmax"].get_weights()
    artifacts["softmax_w"] = softmax_w.astype(np.float32)
    artifacts["softmax_b"] = softmax_b.astype(np.float32)

    artifacts["case_names"] = np.array(CASE_NAMES)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **artifacts)
    logger.info("[INFO] Saved bitwise artifacts to: %s", out_path.resolve())


if __name__ == "__main__":
    main()
