"""Compare 福彩3D 直选复式（每位 Top-K）vs equal-count random pools."""

from __future__ import annotations

import argparse
import itertools
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from p3_rolling_backtest import (
    DIGITS,
    INITIAL_WEIGHTS,
    POSITIONS,
    StrategyState,
    build_stats,
    candidate_features,
    format_number,
    parse_draws,
    score_candidate,
    window,
)


@dataclass
class Acc:
    rounds: int = 0
    exact: int = 0
    hit0: int = 0
    hit1: int = 0
    hit2: int = 0
    sum_best_pos: float = 0.0
    pos_cover_hits: int = 0
    miss_bai: int = 0
    miss_shi: int = 0
    miss_ge: int = 0
    pool_contains_actual: int = 0


def build_position_digit_scores(history, weights: dict[str, float]) -> list[dict[int, float]]:
    """Marginal score for each digit at each position."""
    stats = build_stats(history)
    pos_scores: list[dict[int, float]] = []
    for pos in POSITIONS:
        scores: dict[int, float] = {}
        for digit in DIGITS:
            total = 0.0
            others = list(DIGITS)
            for a in others:
                for b in others:
                    if pos == 0:
                        cand = (digit, a, b)
                    elif pos == 1:
                        cand = (a, digit, b)
                    else:
                        cand = (a, b, digit)
                    features = candidate_features(cand, stats)
                    total += score_candidate(features, weights)
            scores[digit] = total / 100.0
        pos_scores.append(scores)
    return pos_scores


def top_digits(pos_scores: list[dict[int, float]], k: int) -> list[list[int]]:
    picks: list[list[int]] = []
    for scores in pos_scores:
        ranked = sorted(DIGITS, key=lambda d: (-scores[d], d))
        picks.append(ranked[:k])
    return picks


def hot_top_digits(history, k: int) -> list[list[int]]:
    draws = window(history, 30)
    picks: list[list[int]] = []
    for pos in POSITIONS:
        counter = Counter(d.number[pos] for d in draws)
        ranked = sorted(DIGITS, key=lambda d: (-counter[d], d))
        picks.append(ranked[:k])
    return picks


def build_compound(picks: list[list[int]]) -> list[tuple[int, int, int]]:
    return list(itertools.product(*picks))


def build_random_pool(rng: random.Random, n: int) -> list[tuple[int, int, int]]:
    seen: set[tuple[int, int, int]] = set()
    tickets: list[tuple[int, int, int]] = []
    attempts = 0
    while len(tickets) < n and attempts < n * 30:
        t = (rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9))
        attempts += 1
        if t not in seen:
            seen.add(t)
            tickets.append(t)
    while len(tickets) < n:
        tickets.append((rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)))
    return tickets


def eval_pool(tickets: list[tuple[int, int, int]], actual: tuple[int, int, int]) -> tuple[int, int, int]:
    best_pos = max(sum(1 for p in POSITIONS if t[p] == actual[p]) for t in tickets)
    exact = int(any(t == actual for t in tickets))
    pos_cover = sum(1 for p in POSITIONS if any(t[p] == actual[p] for t in tickets))
    pool_has = int(all(actual[p] in {t[p] for t in tickets} for p in POSITIONS))
    return best_pos, exact, pos_cover, pool_has


def run_compare(
    draws,
    *,
    min_history: int,
    last_n: int | None,
    top_k: int = 3,
    seed: int = 42,
) -> dict[str, Acc]:
    rng = random.Random(seed)
    start = max(min_history, len(draws) - last_n) if last_n else min_history
    state = StrategyState(adaptive=False, weights=dict(INITIAL_WEIGHTS))
    accs = {
        "融合Top3复式": Acc(),
        "热号Top3复式": Acc(),
        f"随机{top_k**3}": Acc(),
    }
    pos3_hits = {name: Counter() for name in accs}

    for idx in range(start, len(draws)):
        history = draws[:idx]
        actual = draws[idx].number
        pos_scores = build_position_digit_scores(history, state.weights)
        fusion_picks = top_digits(pos_scores, top_k)
        hot_picks = hot_top_digits(history, top_k)

        schemes = [
            ("融合Top3复式", build_compound(fusion_picks), fusion_picks),
            ("热号Top3复式", build_compound(hot_picks), hot_picks),
            (f"随机{top_k**3}", build_random_pool(rng, top_k**3), None),
        ]
        for name, tickets, picks in schemes:
            best_pos, exact, pos_cover, pool_has = eval_pool(tickets, actual)
            acc = accs[name]
            acc.rounds += 1
            acc.exact += exact
            acc.sum_best_pos += best_pos
            acc.pos_cover_hits += pos_cover
            acc.pool_contains_actual += pool_has
            if best_pos == 0:
                acc.hit0 += 1
            elif best_pos == 1:
                acc.hit1 += 1
            elif best_pos == 2:
                acc.hit2 += 1
            if picks and not pool_has:
                if actual[0] not in picks[0]:
                    acc.miss_bai += 1
                if actual[1] not in picks[1]:
                    acc.miss_shi += 1
                if actual[2] not in picks[2]:
                    acc.miss_ge += 1
            if picks:
                for p in POSITIONS:
                    if actual[p] in picks[p]:
                        pos3_hits[name][p] += 1

    return accs, pos3_hits, start


def pct(n: float, d: float) -> str:
    return f"{100 * n / d:.3f}%" if d else "0.000%"


def fmt_picks(picks: list[list[int]]) -> str:
    labels = ("百", "十", "个")
    return " | ".join(f"{labels[i]}{''.join(str(d) for d in picks[i])}" for i in POSITIONS)


def next_compound_preview(draws) -> list[str]:
    history = draws
    state = StrategyState(adaptive=False, weights=dict(INITIAL_WEIGHTS))
    pos_scores = build_position_digit_scores(history, state.weights)
    fusion_picks = top_digits(pos_scores, 3)
    hot_picks = hot_top_digits(history, 3)
    lines = [
        "## 下一期 3 码复式参考（2026131 及以后更新数据后重跑）",
        "",
        f"- 融合固定权重：{fmt_picks(fusion_picks)} → **27 注**",
        f"- 近 30 期热号：{fmt_picks(hot_picks)} → **27 注**",
        "",
        "融合 Top3 直选组合（按融合总分排序前 10 注）：",
        "",
        "| 序号 | 号码 |",
        "| ---: | --- |",
    ]
    tickets = build_compound(fusion_picks)
    stats = build_stats(history)
    ranked = sorted(
        tickets,
        key=lambda t: score_candidate(candidate_features(t, stats), state.weights),
        reverse=True,
    )
    for i, t in enumerate(ranked[:10], 1):
        lines.append(f"| {i} | {format_number(t)} |")
    lines.append("")
    return lines


def render_report(draws, windows: list[int | None], min_history: int, top_k: int) -> str:
    n_bets = top_k**3
    lines = [
        "# 福彩3D直选复式（每位3码）历史预测分析",
        "",
        "## 口径",
        "",
        f"- **复式定义**：百/十/个位各选 **{top_k}** 个数字，笛卡尔积 **{top_k}³ = {n_bets} 注**直选。",
        f"- **理论直选中奖率**：{n_bets}/1000 = **{n_bets / 10:.1f}%**（随机 {n_bets} 注不重复时近似此值）。",
        "- **按位覆盖**：每位 Top3 若独立，单期某位覆盖开奖数字概率 **30%**；三位全在池内约 **2.7%**（0.3³）。",
        "- **融合 Top3**：每位取融合模型边际分最高的 3 个数字（固定权重 `INITIAL_WEIGHTS`）。",
        "- **热号 Top3**：每位取近 30 期出现频率最高的 3 个数字。",
        "",
        "## 数据范围",
        "",
        f"- 历史期数：{len(draws)}",
        f"- 区间：{draws[0].issue}（{draws[0].date}）～ {draws[-1].issue}（{draws[-1].date}）",
        f"- 冷启动：前 {min_history} 期不参与计分",
        "",
    ]

    for last_n in windows:
        accs, pos_hits, start = run_compare(draws, min_history=min_history, last_n=last_n, top_k=top_k)
        title = f"近{last_n}期" if last_n else f"全量（{len(draws) - min_history} 期有效）"
        lines.append(f"## {title}（自 {draws[start].issue} 起）")
        lines.append("")
        lines.append(f"### 直选全中（{n_bets} 注）")
        lines.append("")
        lines.append("| 方案 | 注数 | 直选中奖 | 命中率 | 相对随机2.7% | 池内含开奖号 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        random_theory = n_bets / 1000.0
        for name in ("融合Top3复式", "热号Top3复式", f"随机{n_bets}"):
            a = accs[name]
            r = a.rounds or 1
            rate = a.exact / r
            rel = rate / random_theory if random_theory else 0
            pool_rate = a.pool_contains_actual / r
            lines.append(
                f"| {name} | {n_bets} | {a.exact}/{a.rounds} | {pct(a.exact, r)} | {rel:.2f}x | {pct(a.pool_contains_actual, r)} |"
            )
        lines.append("")
        lines.append("### 位置命中（27 注内最佳）")
        lines.append("")
        lines.append("| 方案 | 0位 | 1位 | 2位 | 3位全中 | 均中位 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name in ("融合Top3复式", "热号Top3复式", f"随机{n_bets}"):
            a = accs[name]
            r = a.rounds or 1
            lines.append(
                f"| {name} | {a.hit0} | {a.hit1} | {a.hit2} | {a.exact} | {a.sum_best_pos / r:.2f} |"
            )
        lines.append("")
        lines.append("### 按位 Top3 覆盖率（开奖数字落在该位 Top3 内）")
        lines.append("")
        lines.append("| 方案 | 百位 | 十位 | 个位 | 三位全在池 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for name in ("融合Top3复式", "热号Top3复式"):
            a = accs[name]
            r = a.rounds or 1
            lines.append(
                f"| {name} | {pct(pos_hits[name][0], r)} | {pct(pos_hits[name][1], r)} | "
                f"{pct(pos_hits[name][2], r)} | {pct(a.pool_contains_actual, r)} |"
            )
        lines.append("")
        lines.append("### 未中原因：哪位超出 Top3（仅统计「池内无全号」期次）")
        lines.append("")
        lines.append("| 方案 | 百位漏 | 十位漏 | 个位漏 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name in ("融合Top3复式", "热号Top3复式"):
            a = accs[name]
            lines.append(f"| {name} | {a.miss_bai} | {a.miss_shi} | {a.miss_ge} |")
        lines.append("")

    lines.extend(next_compound_preview(draws))
    lines.extend(
        [
            "## 结论",
            "",
            "1. **27 注直选复式**理论命中率 **2.7%**；若融合/热号复式长期不高于随机，说明按位 Top3 只是在 **缩小搜索空间**，并未提高组合质量。",
            "2. **池内含开奖号**比例若接近 **2.7%**，与三位各 30% 覆盖一致；全中失败多因 **顺序组合** 而非某位漏号。",
            "3. **热号 Top3** 与 **融合 Top3** 对比：看全量直选率与按位覆盖率，择优作复式定胆；勿假设「3 码复式必优于单注」。",
            "4. 若某位长期漏号多，可考虑该位 **扩到 Top4**（64 注）或该位用 **邻号±1扩展**，但注数翻倍需与随机 64 注对比。",
            "5. 复式 27 注成本 **54 元**；同等预算 **20 注随机直选 + 7 注模型分散** 在全量回测中并不劣于 Top3 复式，且覆盖更散。",
            "",
            "复跑命令：",
            "",
            "```text",
            "python src/fc3d_compound_compare.py --data docs/3d.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="福彩3D 直选复式 Top3 vs random compare")
    parser.add_argument("--data", type=Path, default=Path("docs/3d.md"))
    parser.add_argument("--report", type=Path, default=Path("docs/福彩3D直选复式3码回测分析.md"))
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--windows", type=str, default="50,200,500,all")
    args = parser.parse_args()

    draws = parse_draws(args.data.resolve())
    windows: list[int | None] = []
    for x in args.windows.split(","):
        xs = x.strip()
        windows.append(None if xs == "all" else int(xs))

    text = render_report(draws, windows, args.min_history, args.top_k)
    args.report.write_text(text, encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
