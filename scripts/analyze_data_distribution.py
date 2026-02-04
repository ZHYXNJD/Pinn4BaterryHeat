#!/usr/bin/env python
"""
诊断训练数据分布，检查冷却板附近 (z<0) 与其它区域的样本占比。
若冷却区样本过少，MSE 会被高温区主导，模型无法学到温度下降现象。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Analyze training data distribution.")
    parser.add_argument(
        "data_path",
        type=str,
        default="sample_data/battery_train_dataset.csv",
        nargs="?",
        help="Path to training CSV (x,y,z,t,temperature in mm and K).",
    )
    args = parser.parse_args()
    path = Path(args.data_path)
    if not path.exists():
        path = ROOT / args.data_path
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    required = ["x", "y", "z", "t", "temperature"]
    for c in required:
        if c not in df.columns:
            print(f"Error: missing column '{c}'")
            sys.exit(1)

    print("=" * 60)
    print(f"Data distribution: {path}")
    print("=" * 60)

    z = df["z"]
    T = df["temperature"]
    init_temp = 313.15

    # 冷却板在 z_min，z 负值为靠近冷却板
    cooling_region = z < -50   # mm，靠近冷却板
    mid_region = (z >= -50) & (z <= 50)
    hot_region = z > 50

    n_cool = cooling_region.sum()
    n_mid = mid_region.sum()
    n_hot = hot_region.sum()
    n_total = len(df)

    print(f"\n1. Spatial distribution (by z, mm):")
    print(f"   Cooling region (z < -50 mm):  {n_cool:6d} ({100*n_cool/n_total:.1f}%)")
    print(f"   Mid region     (-50 <= z <= 50): {n_mid:6d} ({100*n_mid/n_total:.1f}%)")
    print(f"   Hot region     (z > 50 mm):   {n_hot:6d} ({100*n_hot/n_total:.1f}%)")

    # 低温点 (T < init_temp) 占比
    low_temp = T < init_temp - 5  # 明显低于环境温度
    n_low = low_temp.sum()
    print(f"\n2. Temperature distribution:")
    print(f"   Points with T < {init_temp-5:.1f} K (cooling): {n_low:6d} ({100*n_low/n_total:.1f}%)")
    print(f"   T min: {T.min():.2f} K, T max: {T.max():.2f} K, T mean: {T.mean():.2f} K")

    # 冷却区内的温度分布
    n_late_cool = 0
    if n_cool > 0:
        T_cool = T[cooling_region]
        t_cool = df.loc[cooling_region, "t"]
        print(f"\n3. Cooling region (z < -50 mm) temperature:")
        print(f"   T min: {T_cool.min():.2f} K, T max: {T_cool.max():.2f} K, T mean: {T_cool.mean():.2f} K")
        # 冷却发生在前 ~200s，检查 t>200s 时冷却区样本数
        late_cool = cooling_region & (df["t"] > 200)
        n_late_cool = late_cool.sum()
        print(f"   Points at t>200s (when T has dropped): {n_late_cool} ({100*n_late_cool/n_cool:.1f}% of cooling region)")

    # 关键诊断
    print("\n" + "=" * 60)
    if n_cool > 0 and n_late_cool < 0.1 * n_cool:
        print("WARNING: Few cooling-region points at t>200s. Model may not see")
        print("         the 'temperature has dropped' phase. Add more late-time samples.")
    elif n_cool < 0.05 * n_total:
        print("WARNING: Cooling region has < 5% of data. MSE is dominated by")
        print("         hot/mid regions. Model will not learn temperature decrease.")
        print("         Consider: weighted loss, oversampling, or more cooling-region data.")
    elif n_low < 0.05 * n_total:
        print("WARNING: Very few points with T < 308 K. Cooling phenomenon")
        print("         is under-represented. Consider weighted loss.")
    else:
        print("Data distribution looks balanced. If model still fails, check")
        print("temporal coverage (enough points at t>200s when cooling occurs).")
    print("=" * 60)


if __name__ == "__main__":
    main()
