"""Interval stats for 4 same-position matches in 7xc history."""
from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import Path


def parse_draws(path: Path) -> list[tuple[str, str, tuple[int, ...]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\d+) \| ([\d-]+) \| (.+?) \|", line.strip())
        if m:
            rows.append((m.group(1), m.group(2), tuple(int(x) for x in m.group(3).split())))
    return rows


def match_count(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(x == y for x, y in zip(a, b))


def gap_stats(gaps: list[int]) -> dict[str, float | int]:
    if not gaps:
        return {"count": 0}
    c = Counter(gaps)
    return {
        "events": len(gaps) + 1,
        "gaps": len(gaps),
        "mean": statistics.mean(gaps),
        "median": statistics.median(gaps),
        "min": min(gaps),
        "max": max(gaps),
        "most_common": c.most_common(5),
    }


def main() -> None:
    draws = parse_draws(Path("docs/7xc.md"))
    n = len(draws)

    # A: each issue's BEST match vs all prior; when best==4, gap since last best==4
    best_counts = []
    gaps_best_eq4: list[int] = []
    last_idx_eq4 = None
    for i in range(1, n):
        cur = draws[i][2]
        best = max(match_count(cur, draws[j][2]) for j in range(i))
        best_counts.append(best)
        if best == 4:
            if last_idx_eq4 is not None:
                gaps_best_eq4.append(i - last_idx_eq4)
            last_idx_eq4 = i

    # B: each issue has some prior with exactly 4/7 (not necessarily best)
    has_any4 = []
    gaps_any4: list[int] = []
    last_any4 = None
    for i in range(1, n):
        cur = draws[i][2]
        any4 = any(match_count(cur, draws[j][2]) == 4 for j in range(i))
        has_any4.append(any4)
        if any4:
            if last_any4 is not None:
                gaps_any4.append(i - last_any4)
            last_any4 = i

    # C: vs previous issue only, exactly 4/7
    gaps_prev_eq4: list[int] = []
    last_prev4 = None
    for i in range(1, n):
        mc = match_count(draws[i][2], draws[i - 1][2])
        if mc == 4:
            if last_prev4 is not None:
                gaps_prev_eq4.append(i - last_prev4)
            last_prev4 = i

    # D: most recent prior issue within all history with >=4 match (recency gap)
    recency_gaps_ge4: list[int] = []
    for i in range(1, n):
        cur = draws[i][2]
        recent = None
        for j in range(i - 1, -1, -1):
            if match_count(cur, draws[j][2]) >= 4:
                recent = i - j
                break
        if recent is not None:
            recency_gaps_ge4.append(recent)

    # E: front6 only, best prior match == 4
    gaps_front6_eq4: list[int] = []
    last_f6 = None
    for i in range(1, n):
        cur = draws[i][2][:6]
        best = max(match_count(cur, draws[j][2][:6]) for j in range(i))
        if best == 4:
            if last_f6 is not None:
                gaps_front6_eq4.append(i - last_f6)
            last_f6 = i

    bc = Counter(best_counts)
    print("=== 七星彩：同位重复 4 码 间隔统计 ===")
    print(f"总期数: {n}\n")

    print("【1】每期 vs 之前全部历史：最高同位重复恰为 4/7")
    print(f"  出现期数: {sum(1 for x in best_counts if x == 4)} / {len(best_counts)} ({100*sum(1 for x in best_counts if x==4)/len(best_counts):.1f}%)")
    s = gap_stats(gaps_best_eq4)
    if s.get("gaps"):
        print(f"  连续两次「最高=4」之间间隔: 平均 {s['mean']:.1f} 期, 中位 {s['median']:.0f} 期, 最短 {s['min']}, 最长 {s['max']}")
        print(f"  最常见间隔: {s['most_common']}")

    print("\n【2】每期 vs 之前全部历史：存在至少一期 4/7（不要求最高=4）")
    print(f"  出现期数: {sum(has_any4)} / {len(has_any4)} ({100*sum(has_any4)/len(has_any4):.1f}%)")
    s2 = gap_stats(gaps_any4)
    if s2.get("gaps"):
        print(f"  连续两次出现间隔: 平均 {s2['mean']:.1f} 期, 中位 {s2['median']:.0f} 期 (几乎每期都有)")

    print("\n【3】仅 vs 上一期：同位重复恰 4/7")
    prev4_count = len(gaps_prev_eq4) + (1 if last_prev4 else 0)
    print(f"  出现期数: {prev4_count} / {n-1}")
    s3 = gap_stats(gaps_prev_eq4)
    if s3.get("gaps"):
        print(f"  两次间隔: 平均 {s3['mean']:.1f} 期, 中位 {s3['median']:.0f} 期, 最短 {s3['min']}, 最长 {s3['max']}")
        print(f"  最常见间隔: {s3['most_common']}")

    print("\n【4】距「最近一期」至少 4/7 相同的间隔（往前找最近命中）")
    s4 = gap_stats(recency_gaps_ge4)
    print(f"  平均 {s4['mean']:.1f} 期, 中位 {s4['median']:.0f} 期, 最短 {s4['min']}, 最长 {s4['max']}")
    print(f"  最常见间隔: {s4['most_common'][:8]}")

    print("\n【5】前6位 vs 全部历史：最高同位重复恰 4/6")
    s5 = gap_stats(gaps_front6_eq4)
    if s5.get("gaps"):
        print(f"  两次间隔: 平均 {s5['mean']:.1f} 期, 中位 {s5['median']:.0f} 期, 最短 {s5['min']}, 最长 {s5['max']}")
        print(f"  最常见间隔: {s5['most_common']}")

    print("\n【参考】每期最高同位重复分布 (vs 全部历史):")
    for k in sorted(bc.keys(), reverse=True):
        print(f"  {k}/7: {bc[k]} 期 ({100*bc[k]/len(best_counts):.1f}%)")

    # recent example for 260300+11 style: 5/7 best - when did last 5/7 best occur
    print("\n【实例】260300+11 最高 5/7；历史上「新开奖最高=5/7」间隔:")
    gaps5 = []
    last5 = None
    for i in range(1, n):
        cur = draws[i][2]
        best = max(match_count(cur, draws[j][2]) for j in range(i))
        if best == 5:
            if last5 is not None:
                gaps5.append(i - last5)
            last5 = i
    s5b = gap_stats(gaps5)
    if s5b.get("gaps"):
        print(f"  平均 {s5b['mean']:.1f} 期, 中位 {s5b['median']:.0f} 期, 最常见 {s5b['most_common'][:5]}")


if __name__ == "__main__":
    main()
