import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from cl.algorithms import TinyOLv2, build_algorithm
from cl.data_stream import build_stream_from_dataset
from cl.eval import evaluate_cl_accuracy, evaluate_cl_with_diagnostics, forgetting_metric, mismatch_rate_vs_model
from cl.model_utils import (
    keyword_names_to_indices,
    load_frozen_backbone_and_expand_head,
    parse_keyword_list,
    sanity_check_only_last_layer_trainable,
    set_global_seed,
)


def _load_npz_xy(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    x = d["x"][..., np.newaxis].astype(np.float32)
    y = d["y"].astype(np.int32)
    return x, y


def _ensure_no_overlap(a: np.ndarray, b: np.ndarray) -> None:
    overlap = np.intersect1d(a, b)
    if overlap.size > 0:
        raise ValueError(f"base_keywords and new_keywords overlap on class indices: {overlap.tolist()}")


def _csv_writer(path: Path) -> tuple[csv.DictWriter, object]:
    f = path.open("w", newline="", encoding="utf-8")
    cols = ["step", "loss", "acc_base", "acc_new", "acc_all", "forgetting_base", "mismatch_rate_vs_tf"]
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    return writer, f


def _per_class_csv_writer(path: Path) -> Tuple[csv.DictWriter, object]:
    f = path.open("w", newline="", encoding="utf-8")
    cols = ["label_id", "label_name", "count", "pct", "acc"]
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    return writer, f


def _top_confusions_csv_writer(path: Path) -> Tuple[csv.DictWriter, object]:
    f = path.open("w", newline="", encoding="utf-8")
    cols = ["true_id", "pred_id", "count", "true_name", "pred_name"]
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    return writer, f


def _apply_paper16_preset(class_names: np.ndarray) -> Tuple[List[str], List[str], bool]:
    base = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
    new = ["one", "two", "three", "four"]
    names = set(map(str, class_names.tolist()))
    needed = set(base + new)
    missing = sorted(needed - names)
    if missing:
        print(f"[PAPER_16] Missing labels in stats: {missing}. Preset skipped.")
        return [], [], False
    print("[PAPER_16] Using base/new labels for paper-comparable 16-class setup.")
    return base, new, True


def _print_mapping_table(class_names: np.ndarray, keywords: List[str], y_test: np.ndarray) -> List[Dict[str, object]]:
    rows = []
    print("[MAP] keyword -> label_id -> test_count")
    for kw in keywords:
        idx = int(np.where(class_names == kw)[0][0])
        cnt = int(np.sum(y_test == idx))
        row = {"keyword": kw, "label_id": idx, "test_count": cnt}
        rows.append(row)
        print(f"  {kw:>10s} -> {idx:2d} -> {cnt}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual learning for BNN KWS (head-only CL).")
    parser.add_argument("--base_keywords", type=str, required=True)
    parser.add_argument("--new_keywords", type=str, required=True)
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["tinyol", "tinyol_b", "tinyolv2", "tinyolv2_b", "lwf", "lwf_b", "cwr"],
    )
    parser.add_argument("--mix_old_ratio", type=float, default=0.75)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda_kd", type=float, default=1.0)

    parser.add_argument("--base_weights_path", type=str, default="models/bnn_kws_v2_registered.weights.h5")
    parser.add_argument("--train_npz", type=str, default="data/processed/logmel_cl_train.npz")
    parser.add_argument("--test_npz", type=str, default="data/processed/logmel_test.npz")
    parser.add_argument("--stats_npz", type=str, default="data/processed/logmel_stats.npz")
    parser.add_argument("--balance_stream_per_class", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--out_dir", type=str, default="cl_runs")
    parser.add_argument("--mismatch_vs_tf", action="store_true")
    parser.add_argument("--debug_eval", action="store_true")
    parser.add_argument("--balanced_eval_per_class", type=int, default=0)
    parser.add_argument("--save_eval_report", action="store_true")
    parser.add_argument("--paper_16", action="store_true")
    args = parser.parse_args()

    # Batch variants default to the paper-style batch size if caller leaves size at 1.
    if args.algo.endswith("_b") and args.batch_size == 1:
        args.batch_size = 32

    set_global_seed(args.seed)

    stats = np.load(args.stats_npz, allow_pickle=True)
    class_names = stats["class_names"]
    total_num_classes = int(len(class_names))

    if args.paper_16:
        paper_base, paper_new, ok = _apply_paper16_preset(class_names)
        if not ok:
            raise ValueError("paper_16 preset requested but required labels are not available.")
        base_keywords = paper_base
        new_keywords = paper_new
    else:
        base_keywords = parse_keyword_list(args.base_keywords)
        new_keywords = parse_keyword_list(args.new_keywords)

    # HARD ASSERT: every keyword must exist in class_names.
    class_name_set = set(map(str, class_names.tolist()))
    for kw in base_keywords + new_keywords:
        if kw not in class_name_set:
            raise ValueError(f"Keyword '{kw}' not found in class_names.")

    base_idx = keyword_names_to_indices(class_names, base_keywords)
    new_idx = keyword_names_to_indices(class_names, new_keywords)
    _ensure_no_overlap(base_idx, new_idx)

    x_train, y_train = _load_npz_xy(Path(args.train_npz))
    x_test, y_test = _load_npz_xy(Path(args.test_npz))
    mapping_rows = _print_mapping_table(class_names, base_keywords + new_keywords, y_test)

    setup = load_frozen_backbone_and_expand_head(
        base_weights_path=args.base_weights_path,
        base_class_indices=base_idx,
        total_num_classes=total_num_classes,
    )
    feature_extractor = setup["feature_extractor"]
    head_w = setup["head_w"]
    head_b = setup["head_b"]
    sanity_check_only_last_layer_trainable(feature_extractor)

    algo = build_algorithm(
        algo_name=args.algo,
        feature_extractor=feature_extractor,
        head_w=head_w,
        head_b=head_b,
        old_class_indices=base_idx,
        new_class_indices=new_idx,
        lr=args.lr,
        temperature=args.temperature,
        lambda_kd=args.lambda_kd,
        batch_size=args.batch_size,
    )
    algo.sanity_check_train_vars()

    stream = build_stream_from_dataset(
        x=x_train,
        y=y_train,
        old_class_indices=base_idx,
        new_class_indices=new_idx,
        mix_old_ratio=args.mix_old_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
        balance_per_class=args.balance_stream_per_class,
    )
    # Stream integrity checks: new pool should contain only declared new labels.
    if not np.all(np.isin(stream.y_new, new_idx)):
        bad = sorted(set(np.unique(stream.y_new).tolist()) - set(new_idx.tolist()))
        raise AssertionError(f"Stream new pool contains unexpected labels: {bad}")
    if not np.all(np.isin(stream.y_old, base_idx)):
        bad = sorted(set(np.unique(stream.y_old).tolist()) - set(base_idx.tolist()))
        raise AssertionError(f"Stream old pool contains unexpected labels: {bad}")

    # Baseline before CL for forgetting metric.
    w0, b0 = algo.get_eval_head()
    base_metrics_before = evaluate_cl_accuracy(
        feature_extractor=feature_extractor,
        head_w=w0,
        head_b=b0,
        x=x_test,
        y=y_test,
        old_class_indices=base_idx,
        new_class_indices=new_idx,
    )
    base_acc_before = base_metrics_before["acc_base"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.algo}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_writer, csv_file = _csv_writer(out_dir / "cl_log.csv")

    reference_model = None
    if args.mismatch_vs_tf:
        # Optional reference-only hook. Kept None by default unless you wire an external model here.
        reference_model = None

    latest_diag = None

    for step in range(1, args.num_steps + 1):
        xb, yb = stream.sample_batch()
        xb_t = tf.convert_to_tensor(xb, dtype=tf.float32)
        yb_t = tf.convert_to_tensor(yb, dtype=tf.int32)
        train_out = algo.train_step(xb_t, yb_t)

        if step == 1 or step % args.log_every == 0 or step == args.num_steps:
            w_eval, b_eval = algo.get_eval_head()
            m = evaluate_cl_accuracy(
                feature_extractor=feature_extractor,
                head_w=w_eval,
                head_b=b_eval,
                x=x_test,
                y=y_test,
                old_class_indices=base_idx,
                new_class_indices=new_idx,
            )
            latest_diag = evaluate_cl_with_diagnostics(
                feature_extractor=feature_extractor,
                head_w=w_eval,
                head_b=b_eval,
                x=x_test,
                y=y_test,
                old_class_indices=base_idx,
                new_class_indices=new_idx,
                class_names=class_names,
                debug=args.debug_eval,
                balanced_eval_per_class=(args.balanced_eval_per_class if args.balanced_eval_per_class > 0 else None),
                balanced_seed=args.seed,
            )
            forget = forgetting_metric(base_acc_before, m["acc_base"])
            mismatch = mismatch_rate_vs_model(
                feature_extractor=feature_extractor,
                head_w=w_eval,
                head_b=b_eval,
                x=x_test,
                reference_model=reference_model,
            )
            row = {
                "step": step,
                "loss": train_out["loss"],
                "acc_base": m["acc_base"],
                "acc_new": m["acc_new"],
                "acc_all": m["acc_all"],
                "forgetting_base": forget,
                "mismatch_rate_vs_tf": mismatch,
            }
            csv_writer.writerow(row)
            stream_stats = stream.get_stream_stats()
            print(
                f"[STEP {step:5d}] "
                f"loss={row['loss']:.4f} "
                f"acc_base={row['acc_base']:.4f} "
                f"acc_new={row['acc_new']:.4f} "
                f"acc_all={row['acc_all']:.4f} "
                f"forgetting_base={row['forgetting_base']:.4f} "
                f"mismatch_rate_vs_tf={row['mismatch_rate_vs_tf']}"
            )
            if args.debug_eval:
                print(
                    "[STREAM] "
                    f"old_drawn={stream_stats['old_drawn']} "
                    f"new_drawn={stream_stats['new_drawn']} "
                    f"ratio_old={stream_stats['ratio_old']:.4f} "
                    f"ratio_new={stream_stats['ratio_new']:.4f}"
                )

    csv_file.close()

    # Required TinyOLv2 guarantee: old-class params unchanged.
    if isinstance(algo, TinyOLv2):
        algo.post_train_sanity()

    w_final, b_final = algo.get_eval_head()
    np.savez(
        out_dir / "final_head.npz",
        head_w=w_final.astype(np.float32),
        head_b=b_final.astype(np.float32),
        base_class_indices=base_idx.astype(np.int32),
        new_class_indices=new_idx.astype(np.int32),
    )
    print(f"[INFO] Saved CL run outputs to: {out_dir.resolve()}")

    # Final summary line requested.
    final_eval = evaluate_cl_accuracy(
        feature_extractor=feature_extractor,
        head_w=w_final,
        head_b=b_final,
        x=x_test,
        y=y_test,
        old_class_indices=base_idx,
        new_class_indices=new_idx,
    )
    final_diag = evaluate_cl_with_diagnostics(
        feature_extractor=feature_extractor,
        head_w=w_final,
        head_b=b_final,
        x=x_test,
        y=y_test,
        old_class_indices=base_idx,
        new_class_indices=new_idx,
        class_names=class_names,
        debug=args.debug_eval,
        balanced_eval_per_class=(args.balanced_eval_per_class if args.balanced_eval_per_class > 0 else None),
        balanced_seed=args.seed,
    )
    stream_stats = stream.get_stream_stats()
    print(
        "[STREAM FINAL] "
        f"old_drawn={stream_stats['old_drawn']} "
        f"new_drawn={stream_stats['new_drawn']} "
        f"ratio_old={stream_stats['ratio_old']:.4f} "
        f"ratio_new={stream_stats['ratio_new']:.4f} "
        f"(target_old={args.mix_old_ratio:.4f})"
    )
    if args.debug_eval:
        print(f"[STREAM FINAL] per_class_draw_counts={stream_stats['per_class_draw_counts']}")
    print(f"[METRIC] micro_acc_all={final_diag['metrics']['acc_all_micro']:.4f}")
    print(f"[METRIC] macro_acc_all={final_diag['metrics']['acc_all_macro']:.4f}")
    print(f"[METRIC] balanced_acc_all={final_diag['metrics']['acc_all_balanced']:.4f}")
    print(f"[METRIC] expected_acc_all_weighted={final_diag['metrics']['expected_acc_all_weighted']:.4f}")
    print(f"[METRIC] expected_acc_all_macro={final_diag['metrics']['expected_acc_all_macro']:.4f}")
    if final_diag["balanced_eval"] is not None:
        print(
            "[METRIC] paper_comparable_balanced "
            f"k={final_diag['balanced_eval']['k_per_class']} "
            f"micro={final_diag['balanced_eval']['acc_micro_balanced_subset']:.4f} "
            f"macro={final_diag['balanced_eval']['acc_macro_balanced_subset']:.4f}"
        )

    # Save diagnostics artifacts.
    if args.save_eval_report:
        report = {
            "mapping_table": mapping_rows,
            "stream_stats": stream_stats,
            "diagnostics": final_diag,
            "latest_diag": latest_diag,
        }
        with (out_dir / "eval_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)

        per_writer, per_file = _per_class_csv_writer(out_dir / "per_class_accuracy.csv")
        for row in final_diag["per_class_all"]:
            per_writer.writerow(row)
        per_file.close()

        conf_writer, conf_file = _top_confusions_csv_writer(out_dir / "top_confusions.csv")
        for row in final_diag["top_confusions_all"]:
            row = dict(row)
            row["true_name"] = str(class_names[row["true_id"]])
            row["pred_name"] = str(class_names[row["pred_id"]])
            conf_writer.writerow(row)
        conf_file.close()
        print(f"[INFO] Saved eval diagnostics to: {(out_dir / 'eval_report.json').resolve()}")

    print(f"acc_base={final_eval['acc_base']:.4f}")
    print(f"acc_new={final_eval['acc_new']:.4f}")
    print(f"acc_all={final_eval['acc_all']:.4f}")
    print(f"forgetting_base={forgetting_metric(base_acc_before, final_eval['acc_base']):.4f}")
    print("mismatch_rate_vs_tf=nan")


if __name__ == "__main__":
    main()
