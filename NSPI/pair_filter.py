"""
pair_filter.py

Read a USGS M2M Landsat-8 metadata CSV and produce pairs.csv with one
(target, reference) acquisition pair per WRS Path/Row. Both rows in a pair
share the same Path/Row (so identical geographic footprint), but have
different acquisition dates separated by min_days_apart..max_days_apart.

The "target" is the lower-cloud-cover acquisition (we'll synthesize SLC-off
stripes on it). The "reference" is the other acquisition in the pair, used
as the temporal-reference input for NSPI / LaMa-with-reference.

Output schema (one row per scene, two rows per pair):
    pair_id, role, entity_id, product_id, path, row, date, cloud_cover

Usage:
    python pair_filter.py \
        --csv landsat_ot_c2_l1_69d9a2d9e460c018.csv \
        --out pairs.csv \
        --n_pairs 24 \
        --max_cloud 10 \
        --min_days_apart 90 \
        --max_days_apart 365 \
        --seed 0
"""
import argparse
import pandas as pd
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="pairs.csv")
    p.add_argument("--n_pairs", type=int, default=24)
    p.add_argument("--max_cloud", type=float, default=2.0,
                   help="Max Land Cloud Cover percent (default: 2)")
    p.add_argument("--min_days_apart", type=int, default=16,
                   help="Minimum days between target and reference (default: 16)")
    p.add_argument("--max_days_apart", type=int, default=90,
                   help="Max days between target and reference (default: 90)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.csv, encoding="latin-1")
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Defensive filters
    df = df[df["Satellite"] == 8]
    df = df[df["Collection Category"] == "T1"]
    df = df[df["Data Type L1"] == "OLI_TIRS_L1TP"]
    df = df[df["Nadir/Off Nadir"] == "NADIR"]
    df = df[df["Land Cloud Cover"] < args.max_cloud]
    print(f"After cloud-cover and quality filters: {len(df)} rows")

    df["Date Acquired"] = pd.to_datetime(df["Date Acquired"])

    rng = np.random.default_rng(args.seed)
    group_keys = list(df.groupby(["WRS Path", "WRS Row"]).groups.keys())
    rng.shuffle(group_keys)

    candidate_pairs = []
    for path, row in group_keys:
        sub = df[(df["WRS Path"] == path) & (df["WRS Row"] == row)].sort_values("Date Acquired")
        if len(sub) < 2:
            continue
        dates = sub["Date Acquired"].values
        pair_idx = None
        for i in range(len(dates)):
            for j in range(i + 1, len(dates)):
                delta_days = (dates[j] - dates[i]) / np.timedelta64(1, "D")
                if args.min_days_apart <= delta_days <= args.max_days_apart:
                    pair_idx = (i, j)
                    break
            if pair_idx:
                break
        if pair_idx is None:
            continue
        i, j = pair_idx
        rows = sub.iloc[[i, j]].sort_values("Land Cloud Cover").reset_index(drop=True)
        target_row = rows.iloc[0]
        ref_row = rows.iloc[1]
        candidate_pairs.append((target_row, ref_row))
        if len(candidate_pairs) >= args.n_pairs:
            break

    if len(candidate_pairs) < args.n_pairs:
        print(f"WARNING: only {len(candidate_pairs)} pairs found, requested {args.n_pairs}")

    out_rows = []
    for pair_id, (tgt, ref) in enumerate(candidate_pairs):
        for role, r in [("target", tgt), ("reference", ref)]:
            out_rows.append({
                "pair_id": pair_id,
                "role": role,
                "entity_id": r["Entity ID"],
                "product_id": r["Landsat Product Identifier L1"],
                "path": int(r["WRS Path"]),
                "row": int(r["WRS Row"]),
                "date": r["Date Acquired"].strftime("%Y-%m-%d"),
                "cloud_cover": float(r["Land Cloud Cover"]),
            })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(candidate_pairs)} pairs ({len(out_df)} scene rows) to {args.out}")
    print("\nFirst 6 rows:")
    print(out_df.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
