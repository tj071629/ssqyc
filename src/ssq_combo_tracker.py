"""Build / update SSQ 2-code and 3-code combo tracking markdown for post-draw review."""
from __future__ import annotations

import math
from collections import Counter
from datetime import date
from itertools import combinations
from pathlib import Path

from ssq_rolling_backtest import Draw, parse_history

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "docs" / "双色球历史开奖号码.md"
OUTPUT_PATH = ROOT / "docs" / "双色球2码3码组合复盘追踪.md"
TOP_N = 15

# P(specific k-combo appears in one draw)
THEO_PAIR = math.comb(31, 4) / math.comb(33, 6)
THEO_TRIP = math.comb(30, 3) / math.comb(33, 6)
THEO_PAIR_INTERVAL = 1 / THEO_PAIR
THEO_TRIP_INTERVAL = 1 / THEO_TRIP


def count_combos(draws: list[Draw], k: int) -> Counter[tuple[int, ...]]:
    counter: Counter[tuple[int, ...]] = Counter()
    for draw in draws:
        for combo in combinations(sorted(draw.red), k):
            counter[combo] += 1
    return counter


def last_hit_issue(draws: list[Draw], combo: tuple[int, ...]) -> str | None:
    for draw in reversed(draws):
        if all(n in draw.red for n in combo):
            return draw.issue
    return None


def omission_since(draws: list[Draw], combo: tuple[int, ...]) -> int:
    for idx in range(len(draws) - 1, -1, -1):
        if all(n in draws[idx].red for n in combo):
            return len(draws) - 1 - idx
    return len(draws)


def fmt_combo(combo: tuple[int, ...]) -> str:
    return " ".join(f"{n:02d}" for n in combo)


def build_top_table(
    draws: list[Draw],
    counter: Counter[tuple[int, ...]],
    k: int,
    theo_p: float,
    theo_interval: float,
) -> list[str]:
    n_draws = len(draws)
    lines: list[str] = []
    lines.append(
        f"| 排名 | 组合 | 出现期数 | 历史概率 | 平均间隔 | 理论概率 | 理论间隔 | 最近出现 | 迄今遗漏 | 相对理论 |"
    )
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |")
    for rank, (combo, hits) in enumerate(counter.most_common(TOP_N), start=1):
        hist_p = hits / n_draws * 100
        avg_gap = n_draws / hits if hits else float("inf")
        last_issue = last_hit_issue(draws, combo) or "—"
        omit = omission_since(draws, combo)
        relative = hist_p / 100 / theo_p if theo_p else 0
        lines.append(
            f"| {rank} | {fmt_combo(combo)} | {hits} | {hist_p:.2f}% | {avg_gap:.1f} 期 | "
            f"{theo_p * 100:.4f}% | {theo_interval:.1f} 期 | {last_issue} | {omit} 期 | {relative:.2f}× |"
        )
    return lines


def combos_in_draw(red: tuple[int, ...], k: int) -> set[tuple[int, ...]]:
    return set(combinations(sorted(red), k))


def review_section(draws: list[Draw], latest: Draw, pair_top: list[tuple], trip_top: list[tuple]) -> list[str]:
    pair_set = combos_in_draw(latest.red, 2)
    trip_set = combos_in_draw(latest.red, 3)
    hit_pair = [c for c in pair_top if c in pair_set]
    hit_trip = [c for c in trip_top if c in trip_set]
    near_pair = [
        c for c in pair_top if c not in pair_set and len(set(c) & set(latest.red)) == 1
    ][:5]
    near_trip = [
        c for c in trip_top if c not in trip_set and len(set(c) & set(latest.red)) == 2
    ][:5]

    lines = [
        f"### {latest.issue}（{latest.date}）",
        "",
        f"- **开奖红球**：{fmt_combo(latest.red)}　**蓝球**：{latest.blue:02d}",
        f"- **三区**：低区 {sum(1 for n in latest.red if n <= 11)} · "
        f"中区 {sum(1 for n in latest.red if 12 <= n <= 22)} · "
        f"高区 {sum(1 for n in latest.red if n >= 23)}",
        "",
        "**TOP15 中本期命中的 2 码组合**："
        + ("无" if not hit_pair else "、".join(fmt_combo(c) for c in hit_pair)),
        "",
        "**TOP15 中本期命中的 3 码组合**："
        + ("无" if not hit_trip else "、".join(fmt_combo(c) for c in hit_trip)),
        "",
    ]
    if near_pair:
        lines.append("**2 码：本期中 2 个号（差 1 码成对）**：" + "、".join(fmt_combo(c) for c in near_pair))
        lines.append("")
    if near_trip:
        lines.append("**3 码：本期中 2 个号（差 1 码成三元）**：" + "、".join(fmt_combo(c) for c in near_trip))
        lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def build_document(draws: list[Draw]) -> str:
    pair_ctr = count_combos(draws, 2)
    trip_ctr = count_combos(draws, 3)
    latest = draws[-1]
    pair_top_keys = [c for c, _ in pair_ctr.most_common(TOP_N)]
    trip_top_keys = [c for c, _ in trip_ctr.most_common(TOP_N)]

    today = date.today().isoformat()
    parts = [
        "# 双色球 2 码 / 3 码组合复盘追踪",
        "",
        "> **维护说明**：每期开奖后运行  ",
        "> `D:\\anaconda3\\python.exe src\\ssq_combo_tracker.py`  ",
        "> 先更新 `docs/双色球历史开奖号码.md`，再执行脚本刷新本表。",
        "",
        f"- **统计范围**：{draws[0].issue} ~ {latest.issue}（共 {len(draws)} 期）",
        f"- **文档刷新**：{today}",
        f"- **理论概率（单组合）**：2 码 {THEO_PAIR * 100:.4f}%（约 {THEO_PAIR_INTERVAL:.1f} 期/次）；"
        f"3 码 {THEO_TRIP * 100:.4f}%（约 {THEO_TRIP_INTERVAL:.1f} 期/次）",
        "",
        "---",
        "",
        "## 一、2 码组合 TOP 15（全历史）",
        "",
        *build_top_table(draws, pair_ctr, 2, THEO_PAIR, THEO_PAIR_INTERVAL),
        "",
        "---",
        "",
        "## 二、3 码组合 TOP 15（全历史）",
        "",
        *build_top_table(draws, trip_ctr, 3, THEO_TRIP, THEO_TRIP_INTERVAL),
        "",
        "---",
        "",
        "## 三、逐期复盘（最新在上）",
        "",
    ]

    # Last 8 issues review blocks (newest first)
    review_draws = list(reversed(draws[-8:]))
    for draw in review_draws:
        parts.extend(review_section(draws, draw, pair_top_keys, trip_top_keys))

    parts.extend(
        [
            "## 四、使用提示",
            "",
            "1. **出现期数 / 迄今遗漏**：基于全历史重算；新开奖后遗漏=0 的组会前移。",
            "2. **相对理论 > 1**：历史比随机期望更密，作骨架参考，非必出信号。",
            "3. **选号**：优先结合当期走势与规则文档，勿机械追 TOP 榜。",
            "4. **与投注规则**：全历史最高重合 4 红约束见 `ssq_rolling_backtest.py`。",
            "",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    draws = parse_history(HISTORY_PATH)
    if not draws:
        raise SystemExit(f"历史数据为空：{HISTORY_PATH}")
    text = build_document(draws)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"已更新：{OUTPUT_PATH}")
    print(f"统计至：{draws[-1].issue}（共 {len(draws)} 期）")


if __name__ == "__main__":
    main()
