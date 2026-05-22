"""Multi-strategy rolling comparison for 福彩3D."""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from p3_rolling_backtest import (
    ALL_CANDIDATES,
    DEFAULT_MIN_HISTORY,
    INITIAL_WEIGHTS,
    POSITIONS,
    StrategyState,
    format_number,
    is_group_match,
    parse_draws,
    rank_candidates,
    rolling_backtest,
    window,
)


@dataclass
class Acc:
    rounds: int = 0
    exact: dict[int, int] = field(default_factory=lambda: {1: 0, 5: 0, 10: 0, 20: 0})
    group: dict[int, int] = field(default_factory=lambda: {1: 0, 5: 0, 10: 0, 20: 0})
    pos_best: dict[int, Counter] = field(default_factory=lambda: {1: Counter(), 5: Counter(), 10: Counter(), 20: Counter()})
    ranks: list[int] = field(default_factory=list)
    pos_hits: list[int] = field(default_factory=list)


def hot30_ticket(history) -> tuple[int, int, int]:
    draws = window(history, 30)
    digits = []
    for pos in POSITIONS:
        counter = Counter(d.number[pos] for d in draws)
        digits.append(counter.most_common(1)[0][0])
    return tuple(digits)


def pos_top1_hit_rate(history, actual) -> int:
    """How many positions match when each position picks hot30 top1."""
    ticket = hot30_ticket(history)
    return sum(1 for pos in POSITIONS if ticket[pos] == actual[pos])


def run_window(
    draws,
    *,
    min_history: int,
    last_n: int | None,
    seed: int = 42,
) -> tuple[dict[str, Acc], int]:
    rng = random.Random(seed)
    fixed_state = StrategyState(adaptive=False, weights=dict(INITIAL_WEIGHTS))
    adapt_state = StrategyState(adaptive=True, weights=dict(INITIAL_WEIGHTS))
    start = max(min_history, len(draws) - last_n) if last_n else min_history
    accs = {name: Acc() for name in ("random", "hot30", "fixed", "adapt")}
    hot30_pos_hits: list[int] = []

    for idx in range(start, len(draws)):
        history = draws[:idx]
        actual = draws[idx].number

        ranked_fixed = rank_candidates(history, fixed_state)
        ranked_adapt = rank_candidates(history, adapt_state)
        rank_map_adapt = {c: i + 1 for i, (c, _, _) in enumerate(ranked_adapt)}
        accs["adapt"].ranks.append(rank_map_adapt[actual])
        accs["fixed"].ranks.append({c: i + 1 for i, (c, _, _) in enumerate(ranked_fixed)}[actual])

        hot = hot30_ticket(history)
        hot30_pos_hits.append(sum(1 for pos in POSITIONS if hot[pos] == actual[pos]))

        actual_features = ranked_adapt[rank_map_adapt[actual] - 1][2]
        from p3_rolling_backtest import update_state

        update_state(
            adapt_state,
            actual_features,
            (item[2] for item in ranked_adapt[: adapt_state.update_top_n]),
        )

        picks = {
            "fixed": [c for c, _, _ in ranked_fixed[:20]],
            "adapt": [c for c, _, _ in ranked_adapt[:20]],
            "hot30": [hot],
            "random": [rng.choice(ALL_CANDIDATES) for _ in range(20)],
        }

        for name, tickets in picks.items():
            acc = accs[name]
            acc.rounds += 1
            for count in (1, 5, 10, 20):
                subset = tickets[:count]
                if actual in subset:
                    acc.exact[count] += 1
                if any(is_group_match(actual, t) for t in subset):
                    acc.group[count] += 1
                best = max(sum(1 for p in POSITIONS if t[p] == actual[p]) for t in subset)
                acc.pos_best[count][best] += 1

    return accs, start


def pct(n: float, d: float) -> str:
    return f"{n / d * 100:.3f}%" if d else "0.000%"


def summarize(accs: dict[str, Acc]) -> list[str]:
    lines: list[str] = []
    rounds = next(iter(accs.values())).rounds
    lines.append(f"回测期数：{rounds}")
    lines.append("")
    lines.append("### 直选命中（1注理论 0.100%）")
    lines.append("| 策略 | 1注 | 5注 | 10注 | 20注 | 均排名 | 前20率 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in ("hot30", "fixed", "adapt", "random"):
        acc = accs[name]
        r = acc.rounds or 1
        mean_rank = statistics.fmean(acc.ranks) if acc.ranks else 0.0
        top20 = sum(1 for x in acc.ranks if x <= 20)
        lines.append(
            f"| {name} | {pct(acc.exact[1], r)} | {pct(acc.exact[5], r)} | "
            f"{pct(acc.exact[10], r)} | {pct(acc.exact[20], r)} | "
            f"{mean_rank:.1f} | {pct(top20, len(acc.ranks) if acc.ranks else 1)} |"
        )
    lines.append("")
    lines.append("### 组选同号（数字集合一致，不看顺序）")
    lines.append("| 策略 | 1注 | 5注 | 10注 | 20注 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in ("hot30", "fixed", "adapt", "random"):
        acc = accs[name]
        r = acc.rounds or 1
        lines.append(
            f"| {name} | {pct(acc.group[1], r)} | {pct(acc.group[5], r)} | "
            f"{pct(acc.group[10], r)} | {pct(acc.group[20], r)} |"
        )
    lines.append("")
    lines.append("### 位置命中（20注内最佳）")
    lines.append("| 策略 | 0位 | 1位 | 2位 | 3位 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in ("hot30", "fixed", "adapt", "random"):
        c = accs[name].pos_best[20]
        lines.append(f"| {name} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")
    return lines


def structure_stats(draws) -> list[str]:
    from p3_rolling_backtest import even_odd_type, pattern_type, span

    recent100 = draws[-100:]
    patterns = Counter(pattern_type(d.number) for d in recent100)
    spans = Counter(span(d.number) for d in recent100)
    evens = Counter(even_odd_type(d.number) for d in recent100)
    sums = Counter(sum(d.number) for d in recent100)

    lines = ["## 近100期结构分布（参考）", ""]
    lines.append(f"- 形态：组六 {patterns['zuliu']} / 组三 {patterns['zusan']} / 豹子 {patterns['baozi']}")
    lines.append(f"- 跨度 Top3：{spans.most_common(3)}")
    lines.append(f"- 偶数个数 Top3：{evens.most_common(3)}")
    lines.append(f"- 和值 Top3：{sums.most_common(3)}")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="福彩3D multi-strategy comparison.")
    parser.add_argument("--data", type=Path, default=Path("docs/3d.md"))
    parser.add_argument("--report", type=Path, default=Path("docs/福彩3D多策略回测对比.md"))
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    args = parser.parse_args()

    draws = parse_draws(args.data)
    windows: list[tuple[str, int | None]] = [
        ("全量", None),
        ("近500期", 500),
        ("近200期", 200),
        ("近50期", 50),
    ]

    lines: list[str] = ["# 福彩3D多策略回测对比", ""]
    lines.append("## 数据范围")
    lines.append(f"- 历史期数：{len(draws)}")
    lines.append(f"- 起始：{draws[0].issue}（{draws[0].date}）{draws[0].text}")
    lines.append(f"- 最新：{draws[-1].issue}（{draws[-1].date}）{draws[-1].text}")
    lines.append(f"- 冷启动：前 {args.min_history} 期仅作训练，不参与计分")
    lines.append("")
    lines.extend(structure_stats(draws))

    for label, last_n in windows:
        accs, start = run_window(draws, min_history=args.min_history, last_n=last_n)
        lines.append(f"## {label}（自 {draws[start].issue} 起）")
        lines.extend(summarize(accs))
        lines.append("")

    # Next period picks from fixed model
    _, state, next_ranked = rolling_backtest(draws, min_history=args.min_history, adaptive=False)
    hot = hot30_ticket(draws)
    lines.append("## 下一期候选（基于全历史固定权重 + 热号对照）")
    lines.append("")
    lines.append(f"- 近30期按位热号直选：**{format_number(hot)}**")
    lines.append("- 融合固定权重 Top10：")
    lines.append("")
    lines.append("| 排名 | 号码 | 分数 |")
    lines.append("| ---: | --- | ---: |")
    for i, (c, score, _) in enumerate(next_ranked[:10], 1):
        lines.append(f"| {i} | {format_number(c)} | {score:.4f} |")
    lines.append("")
    lines.append("## 对比结论（供规则文档引用）")
    lines.append("- 直选 1/1000，任何策略长期命中率都在随机理论附近波动。")
    lines.append("- 若融合模型 1 注直选率略高于随机、但 5 注以上低于随机，说明排序偏窄，不宜加大复式。")
    lines.append("- 组选宽口径用于观察「数字是否进池」，不等于可兑现的组选奖金。")
    lines.append("")

    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
