# -*- coding: utf-8 -*-
"""Top-N 3-number combinations (same-draw), SSQ red / DLT front."""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_ssq_red(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    rows: list[tuple[str, tuple[int, ...]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "期号" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 3:
            continue
        issue = parts[0]
        if not re.fullmatch(r"\d{7}", issue):
            continue
        red = tuple(sorted(int(x) for x in parts[2].split()))
        if len(red) == 6:
            rows.append((issue, red))
    return rows


def parse_dlt_front(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    rows: list[tuple[str, tuple[int, ...]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "期号" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 3:
            continue
        issue = parts[0]
        if not issue.isdigit():
            continue
        front = tuple(sorted(int(x) for x in parts[2].split()))
        if len(front) == 5:
            rows.append((issue, front))
    return rows


def theory_pick_k_from_n(pick: int, total: int, combo_size: int) -> tuple[float, float]:
    """P(specific combo_size subset appears in pick) and expected interval."""
    total_ways = math.comb(total, pick)
    ways_with_combo = math.comb(total - combo_size, pick - combo_size)
    prob = ways_with_combo / total_ways
    interval = 1.0 / prob if prob > 0 else float("inf")
    return prob * 100, interval


def analyze(
    name: str,
    draws: list[tuple[str, tuple[int, ...]]],
    pick: int,
    total: int,
    top_n: int = 15,
) -> str:
    n = len(draws)
    if n == 0:
        return f"{name}: no data\n"

    counter: Counter[tuple[int, int, int]] = Counter()
    last_issue: dict[tuple[int, int, int], str] = {}
    for issue, nums in draws:
        for tri in combinations(nums, 3):
            counter[tri] += 1
            last_issue[tri] = issue

    theo_prob_pct, theo_interval = theory_pick_k_from_n(pick, total, 3)
    latest_issue = draws[-1][0]

    ranked = counter.most_common(top_n)
    lines = [
        f"## {name}",
        f"- 样本期数：**{n}**",
        f"- 每期开出 **C({pick},3)={math.comb(pick,3)}** 组三码；候选共 **C({total},3)={math.comb(total,3)}** 组",
        f"- 理论概率（任一固定三码同期开出）：**{theo_prob_pct:.4f}%**",
        f"- 理论间隔：**{theo_interval:.1f} 期**",
        "",
        "| 排名 | 组合 | 出现期数 | 历史概率 | 平均间隔 | 理论概率 | 理论间隔 | 最近出现 | 迄今遗漏 | 相对理论 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    for rank, (tri, cnt) in enumerate(ranked, 1):
        hist_prob = cnt / n * 100
        avg_interval = n / cnt if cnt else float("inf")
        last = last_issue[tri]
        # omission: draws since last appearance
        omission = 0
        for issue, _ in reversed(draws):
            if issue == last:
                break
            omission += 1
        rel = (hist_prob / theo_prob_pct) if theo_prob_pct > 0 else 0
        combo_s = " ".join(f"{x:02d}" for x in tri)
        lines.append(
            f"| {rank} | {combo_s} | {cnt} | {hist_prob:.2f}% | {avg_interval:.1f}期 | "
            f"{theo_prob_pct:.4f}% | {theo_interval:.1f}期 | {last} | {omission}期 | {rel:.2f}x |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ssq_path = ROOT / "docs" / "双色球历史开奖号码.md"
    dlt_path = ROOT / "docs" / "大乐透历史开奖号码.md"

    out: list[str] = ["# 三码组合 TOP15（同期开出）\n"]
    if ssq_path.exists():
        ssq = parse_ssq_red(ssq_path)
        out.append(analyze("双色球 · 红球三码 TOP15", ssq, pick=6, total=33, top_n=15))
    if dlt_path.exists():
        dlt = parse_dlt_front(dlt_path)
        out.append(analyze("大乐透 · 前区三码 TOP15", dlt, pick=5, total=35, top_n=15))

    text = "\n".join(out)
    print(text)
    out_path = ROOT / "docs" / "_tmp_triple_combo_top15.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"\n已写入: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
