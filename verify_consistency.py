"""
deliverables/verify_consistency.py
Run from deliverables/ as cwd:
    cd /home/zzz0054/GoldenF/deliverables
    python verify_consistency.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE      = Path(__file__).parent
SRC       = BASE.parent  # GoldenF/

RUNS_OLD  = SRC  / "work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515"
RUNS_REF  = SRC  / "work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835"
RUNS_DEL  = BASE / "outputs/runs/single_085_090_human_only_20260302_235146"
PLOTS_DEL = BASE / "outputs/plots/single_085_090_human_only_20260302_235146"
ALL_ANALY = SRC  / "outputs/human_only_combined_clean/all_analyses_combined.csv"

SUBSETS   = ["085", "090"]
EVAL_MODELS = ["gnn_structlite", "fusion", "gnn_old_holdout"]
LINKPRED_METRIC_COLS = ["test_auc", "test_ap"]
EVAL_METRIC_COLS = ["silhouette_euclidean", "silhouette_cosine", "davies_bouldin", "calinski_harabasz"]
INFORMATIONAL_MODELS = {"gnn_old_holdout"}
TOLERANCE_PCT = 2.0  # percent


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pct(ref, del_):
    """Relative percent difference, safe for near-zero values."""
    base = abs(ref) if abs(ref) > 1e-12 else 1e-12
    return abs(del_ - ref) / base * 100.0


def _status(pct, warn=1.0, fail=5.0, ref_val=None, abs_diff=None):
    """
    Relative-% status.  When the reference value is near zero (<0.02) the
    relative percentage is misleading, so fall back to absolute tolerance:
      abs_diff < 0.005  → OK
      abs_diff < 0.015  → WARN
      otherwise         → FAIL
    """
    if ref_val is not None and abs(ref_val) < 0.02 and abs_diff is not None:
        if abs_diff < 0.005:
            return "OK"
        if abs_diff < 0.015:
            return "WARN"
        return "FAIL"
    if pct > fail:
        return "FAIL"
    if pct > warn:
        return "WARN"
    return "OK"


def _warn_or_ok(pct, tol=TOLERANCE_PCT):
    """For 2% tolerance sections: OK / WARN / FAIL."""
    if pct > tol * 2.5:  # >5%
        return "FAIL"
    if pct > tol:
        return "WARN"
    return "OK"


# ─── section 1: NPZ exact-copy checks ─────────────────────────────────────────

def check_npz_pair(label, src_path, del_path):
    """Compare two NPZ files; return (label, match_str)."""
    src_path, del_path = Path(src_path), Path(del_path)
    if not src_path.exists():
        return label, "SKIP_SRC_MISSING"
    if not del_path.exists():
        return label, "SKIP_DEL_MISSING"
    try:
        src = np.load(src_path, allow_pickle=True)
        dst = np.load(del_path, allow_pickle=True)
        # compare all shared keys
        src_keys = set(src.files)
        dst_keys = set(dst.files)
        if src_keys != dst_keys:
            return label, f"KEY_MISMATCH src={src_keys} del={dst_keys}"
        for key in sorted(src_keys):
            sa, da = src[key], dst[key]
            if sa.dtype.kind in ("U", "S", "O"):  # string/object arrays
                match = np.array_equal(sa, da)
            else:
                match = np.allclose(sa, da, atol=1e-6)
            if not match:
                return label, f"MISMATCH_key={key}"
        return label, "MATCH"
    except Exception as exc:
        return label, f"ERROR({exc})"


def section1():
    print("\n[1/5] NPZ exact-copy checks")
    results = []

    # 8 data NPZ files: source = RUNS_REF inputs/ (GoldenF/data/human_only_085_090/ doesn't exist)
    npz_names = {
        "085": [
            "fusion_subset_085_human.npz",
            "gnn_old_holdout_subset_085_human.npz",
            "go_embeddings_subset_085_human_l2.npz",
            "pager_embeddings_subset_085_human_l2.npz",
        ],
        "090": [
            "fusion_subset_090_human.npz",
            "gnn_old_holdout_subset_090_human.npz",
            "go_embeddings_subset_090_human_l2.npz",
            "pager_embeddings_subset_090_human_l2.npz",
        ],
    }
    for ss in SUBSETS:
        src_base = RUNS_REF / f"subset{ss}/inputs"
        del_base = BASE / f"data/human_only_085_090/subset{ss}"
        for fname in npz_names[ss]:
            label = f"data/human_only_085_090/subset{ss}/{fname}"
            lbl, status = check_npz_pair(label, src_base / fname, del_base / fname)
            results.append((lbl, status))
            print(f"  {status:<25} {lbl}")

    # gnn_holdout_embeddings_l2.npz
    src_gnn = SRC / "work_alpha_gnn_20260212/embeddings/gnn_holdout_embeddings_l2.npz"
    del_gnn = BASE / "stages/02_gnn_text/result/work_alpha_gnn_20260212/embeddings/gnn_holdout_embeddings_l2.npz"
    lbl, status = check_npz_pair(
        "stages/02_gnn_text/result/.../gnn_holdout_embeddings_l2.npz",
        src_gnn,
        del_gnn,
    )
    results.append((lbl, status))
    print(f"  {status:<25} {lbl}")

    n_match = sum(1 for _, s in results if s == "MATCH")
    n_total = len(results)
    print(f"  → {n_match}/{n_total} exact matches")
    return results


# ─── section 2: eval CSVs ref vs latest ──────────────────────────────────────

def load_eval_csv(run_base, subset, model):
    path = run_base / f"subset{subset}/csv/{model}_eval_subset{subset}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def section2():
    print("\n[2/5] Eval CSVs  ref vs latest  (6 file pairs)")
    results = []  # list of (label, status)

    for ss in SUBSETS:
        for model in EVAL_MODELS:
            if model in INFORMATIONAL_MODELS:
                print(f"  subset{ss}  {model:<20} [informational, not from-scratch human-only]")
                results.append((f"subset{ss}/{model}", "INFO"))
                continue

            ref_df = load_eval_csv(RUNS_REF, ss, model)
            del_df = load_eval_csv(RUNS_DEL, ss, model)
            if ref_df is None:
                print(f"  subset{ss}  {model:<20} SKIP_REF_MISSING")
                results.append((f"subset{ss}/{model}", "SKIP"))
                continue
            if del_df is None:
                print(f"  subset{ss}  {model:<20} SKIP_DEL_MISSING")
                results.append((f"subset{ss}/{model}", "SKIP"))
                continue

            for col in EVAL_METRIC_COLS:
                if col not in ref_df.columns or col not in del_df.columns:
                    continue
                ref_vals = ref_df[col].dropna()
                del_vals = del_df[col].dropna()
                if len(ref_vals) == 0 or len(del_vals) == 0:
                    continue
                ref_mean = ref_vals.mean()
                del_mean = del_vals.mean()
                pct = _pct(ref_mean, del_mean)
                st = _warn_or_ok(pct)
                # short col name for display
                short = col.replace("silhouette_", "sil_").replace("davies_bouldin", "db").replace("calinski_harabasz", "ch")
                print(f"  subset{ss}  {model:<20} {short:<12} ref={ref_mean:.4f}  del={del_mean:.4f}  diff={pct:.2f}%  {st}")
                results.append((f"subset{ss}/{model}/{col}", st))

    n_ok   = sum(1 for _, s in results if s == "OK")
    n_warn = sum(1 for _, s in results if s == "WARN")
    n_fail = sum(1 for _, s in results if s == "FAIL")
    print(f"  → OK={n_ok}  WARN={n_warn}  FAIL={n_fail}")
    return results


# ─── section 3: linkpred final epoch ref vs latest ───────────────────────────

def load_linkpred_csv(run_base, subset):
    path = run_base / f"subset{subset}/csv/text_structlite_linkpred_subset{subset}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def section3():
    print("\n[3/5] Linkpred final epoch  ref vs latest  (4 file pairs)")
    results = []

    for ss in SUBSETS:
        ref_df = load_linkpred_csv(RUNS_REF, ss)
        del_df = load_linkpred_csv(RUNS_DEL, ss)
        if ref_df is None:
            print(f"  subset{ss}  SKIP_REF_MISSING")
            continue
        if del_df is None:
            print(f"  subset{ss}  SKIP_DEL_MISSING")
            continue

        ref_last = ref_df.iloc[-1]
        del_last = del_df.iloc[-1]

        for col in LINKPRED_METRIC_COLS:
            if col not in ref_df.columns:
                continue
            ref_val = ref_last[col]
            del_val = del_last[col]
            pct = _pct(ref_val, del_val)
            st = _warn_or_ok(pct)
            print(f"  subset{ss}  {col:<12} ref={ref_val:.4f}  del={del_val:.4f}  diff={pct:.2f}%  {st}")
            results.append((f"subset{ss}/{col}", st))

    n_ok   = sum(1 for _, s in results if s == "OK")
    n_warn = sum(1 for _, s in results if s == "WARN")
    n_fail = sum(1 for _, s in results if s == "FAIL")
    print(f"  → OK={n_ok}  WARN={n_warn}  FAIL={n_fail}")
    return results


# ─── section 4: historical trend old → ref ───────────────────────────────────

def section4():
    print("\n[4/5] Historical trend  old (Feb 20) → ref (Mar 2 21:08)  (informational)")
    for ss in SUBSETS:
        for model in EVAL_MODELS:
            if model in INFORMATIONAL_MODELS:
                continue
            old_df = load_eval_csv(RUNS_OLD, ss, model)
            ref_df = load_eval_csv(RUNS_REF, ss, model)
            if old_df is None or ref_df is None:
                print(f"  subset{ss}  {model:<20} SKIP (missing file)")
                continue
            for col in EVAL_METRIC_COLS:
                if col not in old_df.columns or col not in ref_df.columns:
                    continue
                old_val = old_df[col].dropna().mean()
                ref_val = ref_df[col].dropna().mean()
                pct = (ref_val - old_val) / (abs(old_val) if abs(old_val) > 1e-12 else 1e-12) * 100.0
                sign = "+" if pct >= 0 else ""
                short = col.replace("silhouette_", "sil_").replace("davies_bouldin", "db").replace("calinski_harabasz", "ch")
                print(f"  subset{ss}  {model:<20} {short:<12}  {old_val:.4f} → {ref_val:.4f}  ({sign}{pct:.1f}%)")


# ─── section 5: additional metrics vs all_analyses_combined ──────────────────

# Mapping: (category, metric) in additional_metrics → column in all_analyses_combined
ADD_METRIC_MAP = [
    ("text_coherence_meta", "text_coherence_clean_mean", "text_coherence_clean_mean"),
    ("stability",           "pairwise_ari_mean",         "pairwise_ari_mean"),
    ("confound",            "primary_gse_ari",            "primary_gse_ari"),
    ("confound",            "primary_platform_nmi",       "primary_platform_nmi"),
]
# Model mapping: all_analyses_combined → additional_metrics_comparison column
MODEL_MAP = {
    "fusion_a0.5": "fusion",
    "gnn_structlite": "gnn_structlite",
}
# Informational in all_analyses_combined (not from-scratch human)
INFORMATIONAL_SRC = {"gnn", "golden_fusion"}


def section5():
    print("\n[5/5] Additional metrics vs all_analyses_combined.csv")
    results = []

    if not ALL_ANALY.exists():
        print(f"  SKIP: {ALL_ANALY} not found")
        return results

    src_df = pd.read_csv(ALL_ANALY)

    for ss in SUBSETS:
        add_path = PLOTS_DEL / f"additional_metrics_comparison_subset{ss}.csv"
        if not add_path.exists():
            print(f"  subset{ss}  SKIP (deliverables file missing: {add_path})")
            continue
        add_df = pd.read_csv(add_path)

        subset_int = int(ss)  # 85 or 90
        src_sub = src_df[src_df["subset"] == subset_int]

        for src_model, del_col in MODEL_MAP.items():
            src_row = src_sub[src_sub["model"] == src_model]
            if src_row.empty:
                print(f"  subset{ss}  {src_model:<20} SKIP (not in all_analyses_combined)")
                continue
            sr = src_row.iloc[0]

            for cat, met, src_col in ADD_METRIC_MAP:
                if src_col not in sr.index:
                    continue
                ref_val = sr[src_col]

                # look up in additional_metrics_comparison
                mask = (add_df["category"] == cat) & (add_df["metric"] == met)
                rows = add_df[mask]
                if rows.empty or del_col not in rows.columns:
                    print(f"  subset{ss}  {src_model:<16} {met:<28} SKIP (not in deliverables)")
                    continue
                del_val = rows.iloc[0][del_col]

                pct = _pct(ref_val, del_val)
                abs_diff = abs(del_val - ref_val)
                st = _status(pct, warn=1.0, fail=5.0, ref_val=ref_val, abs_diff=abs_diff)
                print(f"  subset{ss}  {src_model:<16} {met:<28} ref={ref_val:.5f}  del={del_val:.5f}  {st} {pct:.1f}%")
                results.append((f"subset{ss}/{src_model}/{met}", st))

        # informational models
        for src_model in INFORMATIONAL_SRC:
            sr = src_sub[src_sub["model"] == src_model]
            if not sr.empty:
                print(f"  subset{ss}  {src_model:<20} [informational, not from-scratch human-only]")

    n_ok   = sum(1 for _, s in results if s == "OK")
    n_warn = sum(1 for _, s in results if s == "WARN")
    n_fail = sum(1 for _, s in results if s == "FAIL")
    print(f"  → OK={n_ok}  WARN={n_warn}  FAIL={n_fail}")
    return results


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(" GoldenF deliverables/ consistency report")
    print(f" REF  (Mar 2 21:08): {RUNS_REF.relative_to(SRC)}")
    print(f" DEL  (Mar 2 23:51): {RUNS_DEL.relative_to(BASE)}")
    print("=" * 65)

    npz_results  = section1()
    eval_results = section2()
    lp_results   = section3()
    section4()
    add_results  = section5()

    # ── summary ──
    print("\n" + "=" * 65)
    n_npz_match = sum(1 for _, s in npz_results if s == "MATCH")
    n_npz_total = len(npz_results)

    def _counts(res):
        ok   = sum(1 for _, s in res if s == "OK")
        warn = sum(1 for _, s in res if s == "WARN")
        fail = sum(1 for _, s in res if s == "FAIL")
        tot  = sum(1 for _, s in res if s not in ("INFO", "SKIP"))
        return ok, warn, fail, tot

    e_ok, e_warn, e_fail, e_tot = _counts(eval_results)
    l_ok, l_warn, l_fail, l_tot = _counts(lp_results)
    a_ok, a_warn, a_fail, a_tot = _counts(add_results)

    total_fail = e_fail + l_fail + a_fail
    print(f"SUMMARY:")
    print(f"  NPZ exact-copy:   {n_npz_match}/{n_npz_total} MATCH")
    print(f"  Eval CSVs:        {e_ok}/{e_tot} OK  {e_warn} WARN  {e_fail} FAIL")
    print(f"  Linkpred final:   {l_ok}/{l_tot} OK  {l_warn} WARN  {l_fail} FAIL")
    print(f"  Addl metrics:     {a_ok}/{a_tot} OK  {a_warn} WARN  {a_fail} FAIL")
    print(f"  Total FAILures:   {total_fail}")
    print("=" * 65)

    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
