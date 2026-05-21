from __future__ import annotations

import argparse
import math
import statistics
from itertools import permutations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DIGITS = range(10)
POSITIONS = range(3)
DEFAULT_TICKET_COUNTS = (1, 5, 10, 20)
DEFAULT_MIN_HISTORY = 100


@dataclass(frozen=True)
class Draw:
    issue: str
    date: str
    number: tuple[int, int, int]

    @property
    def text(self) -> str:
        return "".join(str(digit) for digit in self.number)


@dataclass
class StrategyState:
    adaptive: bool
    weights: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.035
    decay: float = 0.997
    update_top_n: int = 20
    min_weight: float = -2.5
    max_weight: float = 2.5


INITIAL_WEIGHTS = {
    "pos_hot10": 0.55,
    "pos_hot30": 0.75,
    "pos_hot100": 0.45,
    "all_hot30": 0.18,
    "repeat_last": 0.15,
    "neighbor_last": 0.35,
    "neighbor_prev": 0.15,
    "transition": 0.45,
    "omission_mid": 0.55,
    "omission_deep": 0.05,
    "sum_hot100": 0.28,
    "span_hot100": 0.22,
    "pattern_hot100": 0.18,
    "even_odd_fit": 0.12,
}


ALL_CANDIDATES = tuple((a, b, c) for a in DIGITS for b in DIGITS for c in DIGITS)


def parse_draws(path: Path) -> list[Draw]:
    draws: list[Draw] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        issue, date, number_text = cells[:3]
        number_text = number_text.replace(" ", "")
        if len(number_text) != 3 or not number_text.isdigit():
            continue
        draws.append(Draw(issue=issue, date=date, number=tuple(int(ch) for ch in number_text)))  # type: ignore[arg-type]
    return draws


def pattern_type(number: tuple[int, int, int]) -> str:
    unique = len(set(number))
    if unique == 1:
        return "baozi"
    if unique == 2:
        return "zusan"
    return "zuliu"


def span(number: tuple[int, int, int]) -> int:
    return max(number) - min(number)


def even_odd_type(number: tuple[int, int, int]) -> int:
    return sum(1 for digit in number if digit % 2 == 0)


def window(history: list[Draw], size: int) -> list[Draw]:
    if len(history) <= size:
        return history
    return history[-size:]


def normalized_counter_value(counter: Counter, key: object, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return counter[key] / denominator


def build_stats(history: list[Draw]) -> dict[str, object]:
    last = history[-1].number
    prev = history[-2].number if len(history) >= 2 else last

    pos_counts: dict[int, list[Counter]] = {}
    all_counts: dict[int, Counter] = {}
    for size in (10, 30, 100):
        draws = window(history, size)
        pos_counts[size] = [Counter(draw.number[pos] for draw in draws) for pos in POSITIONS]
        all_counts[size] = Counter(digit for draw in draws for digit in draw.number)

    omissions: list[dict[int, int]] = []
    for pos in POSITIONS:
        pos_omissions: dict[int, int] = {}
        for digit in DIGITS:
            miss = 0
            for draw in reversed(history):
                if draw.number[pos] == digit:
                    break
                miss += 1
            else:
                miss = len(history) + 1
            pos_omissions[digit] = miss
        omissions.append(pos_omissions)

    recent100 = window(history, 100)
    sum_counts = Counter(sum(draw.number) for draw in recent100)
    span_counts = Counter(span(draw.number) for draw in recent100)
    pattern_counts = Counter(pattern_type(draw.number) for draw in recent100)
    even_counts = Counter(even_odd_type(draw.number) for draw in recent100)

    transitions: list[dict[int, Counter]] = [defaultdict(Counter) for _ in POSITIONS]
    transition_history = window(history, 250)
    for left, right in zip(transition_history, transition_history[1:]):
        for pos in POSITIONS:
            transitions[pos][left.number[pos]][right.number[pos]] += 1

    transition_max: list[dict[int, int]] = []
    for pos in POSITIONS:
        pos_max: dict[int, int] = {}
        for digit in DIGITS:
            pos_max[digit] = max(transitions[pos][digit].values(), default=1)
        transition_max.append(pos_max)

    return {
        "last": last,
        "prev": prev,
        "pos_counts": pos_counts,
        "all_counts": all_counts,
        "omissions": omissions,
        "sum_counts": sum_counts,
        "span_counts": span_counts,
        "pattern_counts": pattern_counts,
        "even_counts": even_counts,
        "transitions": transitions,
        "transition_max": transition_max,
        "history_len": len(history),
        "recent100_len": len(recent100),
    }


def candidate_features(candidate: tuple[int, int, int], stats: dict[str, object]) -> dict[str, float]:
    last: tuple[int, int, int] = stats["last"]  # type: ignore[assignment]
    prev: tuple[int, int, int] = stats["prev"]  # type: ignore[assignment]
    pos_counts: dict[int, list[Counter]] = stats["pos_counts"]  # type: ignore[assignment]
    all_counts: dict[int, Counter] = stats["all_counts"]  # type: ignore[assignment]
    omissions: list[dict[int, int]] = stats["omissions"]  # type: ignore[assignment]
    transitions: list[dict[int, Counter]] = stats["transitions"]  # type: ignore[assignment]
    transition_max: list[dict[int, int]] = stats["transition_max"]  # type: ignore[assignment]
    recent100_len: int = stats["recent100_len"]  # type: ignore[assignment]
    sum_counts: Counter = stats["sum_counts"]  # type: ignore[assignment]
    span_counts: Counter = stats["span_counts"]  # type: ignore[assignment]
    pattern_counts: Counter = stats["pattern_counts"]  # type: ignore[assignment]
    even_counts: Counter = stats["even_counts"]  # type: ignore[assignment]

    features: dict[str, float] = {}
    for size, name in ((10, "pos_hot10"), (30, "pos_hot30"), (100, "pos_hot100")):
        denom = min(size, stats["history_len"])  # type: ignore[arg-type]
        features[name] = sum(pos_counts[size][pos][candidate[pos]] / denom for pos in POSITIONS) / 3.0

    features["all_hot30"] = sum(all_counts[30][digit] / (min(30, stats["history_len"]) * 3) for digit in candidate) / 3.0  # type: ignore[arg-type]
    features["repeat_last"] = sum(1 for pos in POSITIONS if candidate[pos] == last[pos]) / 3.0
    features["neighbor_last"] = sum(1 for pos in POSITIONS if abs(candidate[pos] - last[pos]) == 1) / 3.0
    features["neighbor_prev"] = sum(1 for pos in POSITIONS if abs(candidate[pos] - prev[pos]) == 1) / 3.0

    transition_values = []
    for pos in POSITIONS:
        from_digit = last[pos]
        to_digit = candidate[pos]
        transition_values.append(transitions[pos][from_digit][to_digit] / transition_max[pos][from_digit])
    features["transition"] = sum(transition_values) / 3.0

    mid_scores = []
    deep_scores = []
    for pos in POSITIONS:
        miss = omissions[pos][candidate[pos]]
        mid_scores.append(math.exp(-((miss - 12) ** 2) / (2 * 8**2)))
        deep_scores.append(min(miss / 80.0, 1.0))
    features["omission_mid"] = sum(mid_scores) / 3.0
    features["omission_deep"] = sum(deep_scores) / 3.0

    features["sum_hot100"] = normalized_counter_value(sum_counts, sum(candidate), recent100_len)
    features["span_hot100"] = normalized_counter_value(span_counts, span(candidate), recent100_len)
    features["pattern_hot100"] = normalized_counter_value(pattern_counts, pattern_type(candidate), recent100_len)
    features["even_odd_fit"] = normalized_counter_value(even_counts, even_odd_type(candidate), recent100_len)
    return features


def score_candidate(features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def rank_candidates(
    history: list[Draw],
    state: StrategyState,
) -> list[tuple[tuple[int, int, int], float, dict[str, float]]]:
    stats = build_stats(history)
    ranked = []
    for candidate in ALL_CANDIDATES:
        features = candidate_features(candidate, stats)
        score = score_candidate(features, state.weights)
        ranked.append((candidate, score, features))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def update_state(
    state: StrategyState,
    actual_features: dict[str, float],
    selected_features: Iterable[dict[str, float]],
) -> None:
    if not state.adaptive:
        return
    selected = list(selected_features)
    if not selected:
        return
    for name in INITIAL_WEIGHTS:
        expected = sum(features.get(name, 0.0) for features in selected) / len(selected)
        current = state.weights.get(name, 0.0)
        target = INITIAL_WEIGHTS[name]
        updated = current * state.decay + target * (1.0 - state.decay)
        updated += state.learning_rate * (actual_features.get(name, 0.0) - expected)
        state.weights[name] = max(state.min_weight, min(state.max_weight, updated))


def format_number(number: tuple[int, int, int]) -> str:
    return "".join(str(digit) for digit in number)


def is_group_match(left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    return sorted(left) == sorted(right)


def group_coverage(numbers: Iterable[tuple[int, int, int]], *, include_baozi: bool = True) -> set[tuple[int, int, int]]:
    covered: set[tuple[int, int, int]] = set()
    for number in numbers:
        if not include_baozi and len(set(number)) == 1:
            continue
        covered.update(set(permutations(number, 3)))
    return covered


def rolling_backtest(
    draws: list[Draw],
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    ticket_counts: tuple[int, ...] = DEFAULT_TICKET_COUNTS,
    adaptive: bool = True,
    learning_rate: float = 0.035,
) -> tuple[dict[str, object], StrategyState, list[tuple[tuple[int, int, int], float, dict[str, float]]]]:
    if len(draws) <= min_history:
        raise ValueError(f"Need more than {min_history} draws, got {len(draws)}.")

    state = StrategyState(
        adaptive=adaptive,
        weights=dict(INITIAL_WEIGHTS),
        learning_rate=learning_rate if adaptive else 0.0,
    )
    max_tickets = max(ticket_counts)
    exact_hits = {count: 0 for count in ticket_counts}
    group_hits = {count: 0 for count in ticket_counts}
    best_position_hits = {count: Counter() for count in ticket_counts}
    actual_ranks: list[int] = []
    top1_examples: list[dict[str, str]] = []
    round_results: list[dict[str, object]] = []

    for idx in range(min_history, len(draws)):
        history = draws[:idx]
        actual = draws[idx]
        ranked = rank_candidates(history, state)
        selected = ranked[:max_tickets]
        selected_numbers = [item[0] for item in selected]

        rank_map = {candidate: pos + 1 for pos, (candidate, _, _) in enumerate(ranked)}
        actual_ranks.append(rank_map[actual.number])

        for count in ticket_counts:
            tickets = selected_numbers[:count]
            if actual.number in tickets:
                exact_hits[count] += 1
            if any(is_group_match(actual.number, ticket) for ticket in tickets):
                group_hits[count] += 1
            best = max(sum(1 for pos in POSITIONS if ticket[pos] == actual.number[pos]) for ticket in tickets)
            best_position_hits[count][best] += 1

        round_results.append(
            {
                "issue": actual.issue,
                "date": actual.date,
                "actual": actual.number,
                "selected": selected_numbers,
            }
        )

        if len(top1_examples) < 8:
            top1_examples.append(
                {
                    "issue": actual.issue,
                    "date": actual.date,
                    "pred": format_number(selected_numbers[0]),
                    "actual": actual.text,
                    "rank": str(rank_map[actual.number]),
                }
            )

        actual_features = ranked[rank_map[actual.number] - 1][2]
        update_state(state, actual_features, (item[2] for item in selected[: state.update_top_n]))

    final_ranked = rank_candidates(draws, state)
    rounds = len(draws) - min_history
    metrics: dict[str, object] = {
        "rounds": rounds,
        "min_history": min_history,
        "adaptive": adaptive,
        "learning_rate": learning_rate if adaptive else 0.0,
        "exact_hits": exact_hits,
        "group_hits": group_hits,
        "best_position_hits": best_position_hits,
        "actual_ranks": actual_ranks,
        "mean_rank": statistics.fmean(actual_ranks),
        "median_rank": statistics.median(actual_ranks),
        "rank_top20": sum(1 for rank in actual_ranks if rank <= 20),
        "rank_top100": sum(1 for rank in actual_ranks if rank <= 100),
        "examples": top1_examples,
        "round_results": round_results,
    }
    return metrics, state, final_ranked


def pct(numerator: int | float, denominator: int | float) -> str:
    if denominator == 0:
        return "0.000%"
    return f"{numerator / denominator * 100:.3f}%"


def report(
    draws: list[Draw],
    metrics: dict[str, object],
    state: StrategyState,
    next_ranked: list[tuple[tuple[int, int, int], float, dict[str, float]]],
    *,
    ticket_counts: tuple[int, ...] = DEFAULT_TICKET_COUNTS,
) -> str:
    rounds: int = metrics["rounds"]  # type: ignore[assignment]
    exact_hits: dict[int, int] = metrics["exact_hits"]  # type: ignore[assignment]
    group_hits: dict[int, int] = metrics["group_hits"]  # type: ignore[assignment]
    best_position_hits: dict[int, Counter] = metrics["best_position_hits"]  # type: ignore[assignment]
    is_adaptive: bool = metrics["adaptive"]  # type: ignore[assignment]
    mode_name = "逐期自适应" if is_adaptive else "固定权重"
    method_text = (
        f"前 {metrics['min_history']} 期只作冷启动训练，之后每期只用过去数据预测下一期，开奖后再更新权重。"
        if is_adaptive
        else f"前 {metrics['min_history']} 期只作冷启动训练，之后每期只用过去数据预测下一期，权重全程固定不更新。"
    )

    lines: list[str] = []
    lines.append(f"# 排列3{mode_name}滚动回测报告")
    lines.append("")
    lines.append("## 数据范围")
    lines.append(f"- 历史期数：{len(draws)}")
    lines.append(f"- 起始期：{draws[0].issue}（{draws[0].date}），号码 {draws[0].text}")
    lines.append(f"- 最新期：{draws[-1].issue}（{draws[-1].date}），号码 {draws[-1].text}")
    lines.append(f"- 回测方式：{method_text}")
    lines.append("")
    lines.append("## 规则信号")
    lines.append("- 按位冷热：最近 10/30/100 期每个位置数字频率。")
    lines.append("- 承接关系：上一期同位重号、同位 ±1 邻号、上两期邻号。")
    lines.append("- 转移关系：同位置上一位数字到下一位数字的近 250 期转移频率。")
    lines.append("- 遗漏回补：同位置数字的中等遗漏窗口和深遗漏窗口。")
    lines.append("- 结构约束：和值、跨度、组三/组六/豹子形态、奇偶结构的近 100 期分布。")
    lines.append("")
    lines.append("## 直选命中率")
    lines.append("| 每期票数 | 直选命中次数 | 回测命中率 | 随机理论值 | 相对随机 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for count in ticket_counts:
        hit = exact_hits[count]
        random_rate = count / 1000
        relative = (hit / rounds) / random_rate if random_rate else 0.0
        lines.append(f"| {count} | {hit} / {rounds} | {pct(hit, rounds)} | {random_rate * 100:.3f}% | {relative:.2f}x |")
    lines.append("")
    lines.append("## 组选同号命中率")
    lines.append("这里不等同于一等奖，只表示三位数字集合相同、顺序可不同，用于观察候选池是否抓到数字。")
    lines.append("")
    lines.append("| 每期票数 | 组选同号次数 | 命中率 |")
    lines.append("| --- | ---: | ---: |")
    for count in ticket_counts:
        hit = group_hits[count]
        lines.append(f"| {count} | {hit} / {rounds} | {pct(hit, rounds)} |")
    lines.append("")
    lines.append("## 位置命中分布")
    lines.append("每期在 N 注候选中，取与开奖号同位置相同最多的那一注。")
    lines.append("")
    lines.append("| 每期票数 | 0位 | 1位 | 2位 | 3位 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for count in ticket_counts:
        counter = best_position_hits[count]
        lines.append(
            f"| {count} | {counter[0]} | {counter[1]} | {counter[2]} | {counter[3]} |"
        )
    lines.append("")
    lines.append("## 实际号码得分排名")
    lines.append(f"- 平均排名：{metrics['mean_rank']:.1f} / 1000")
    lines.append(f"- 中位排名：{metrics['median_rank']:.1f} / 1000")
    lines.append(f"- 实际号码落入模型前20：{metrics['rank_top20']} 次，{pct(metrics['rank_top20'], rounds)}")
    lines.append(f"- 实际号码落入模型前100：{metrics['rank_top100']} 次，{pct(metrics['rank_top100'], rounds)}")
    lines.append("")
    lines.append("## 当前权重")
    lines.append("| 信号 | 初始权重 | 当前权重 |")
    lines.append("| --- | ---: | ---: |")
    for name in sorted(INITIAL_WEIGHTS):
        lines.append(f"| {name} | {INITIAL_WEIGHTS[name]:.3f} | {state.weights[name]:.3f} |")
    lines.append("")
    lines.append("## 下一期候选")
    lines.append("以下候选只代表模型当前规则下的排序结果，不代表确定性提升。")
    lines.append("")
    lines.append("| 排名 | 号码 | 分数 |")
    lines.append("| ---: | --- | ---: |")
    for idx, (candidate, score, _) in enumerate(next_ranked[:20], start=1):
        lines.append(f"| {idx} | {format_number(candidate)} | {score:.4f} |")
    lines.append("")
    lines.append("## 结论")
    lines.append("排列3直选理论中奖率是每注 1/1000。若回测命中率只是在随机理论值附近波动，说明规则更多是在做候选排序，而不是形成稳定可迁移优势。后续若继续优化，应使用滚动回测或留出最近期验证，避免按全历史命中结果反向调参。")
    lines.append("")
    return "\n".join(lines)


def group_report(
    draws: list[Draw],
    metrics: dict[str, object],
    *,
    ticket_counts: tuple[int, ...] = DEFAULT_TICKET_COUNTS,
    recent_n: int = 50,
) -> str:
    rounds: int = metrics["rounds"]  # type: ignore[assignment]
    round_results: list[dict[str, object]] = metrics["round_results"]  # type: ignore[assignment]
    recent = round_results[-recent_n:]

    def summarize(results: list[dict[str, object]], count: int, *, include_baozi: bool) -> tuple[int, float]:
        hits = 0
        expected = 0.0
        for item in results:
            actual: tuple[int, int, int] = item["actual"]  # type: ignore[assignment]
            selected: list[tuple[int, int, int]] = item["selected"]  # type: ignore[assignment]
            coverage = group_coverage(selected[:count], include_baozi=include_baozi)
            expected += len(coverage) / 1000.0
            if actual in coverage:
                hits += 1
        return hits, expected

    lines: list[str] = []
    lines.append("# 排列3组选滚动回测报告")
    lines.append("")
    lines.append("## 口径")
    lines.append("- 组选宽口径：候选号与开奖号三位数字的多重集合一致即算命中，不看顺序。")
    lines.append("- 可投注组选口径：排除豹子候选；组三按 3 个直选排列覆盖，组六按 6 个直选排列覆盖。")
    lines.append("- 理论概率不是按票数直接除以 1000，而是按每期候选组选实际覆盖的直选号码数量计算。")
    lines.append("")
    lines.append("## 全历史滚动回测")
    lines.append(f"- 回测期数：{rounds}")
    lines.append("")
    lines.append("| 每期候选数 | 宽口径命中 | 宽口径命中率 | 可投注组选命中 | 可投注命中率 | 可投注理论覆盖率 | 相对理论 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for count in ticket_counts:
        wide_hits, _ = summarize(round_results, count, include_baozi=True)
        bet_hits, expected = summarize(round_results, count, include_baozi=False)
        expected_rate = expected / len(round_results)
        actual_rate = bet_hits / len(round_results)
        relative = actual_rate / expected_rate if expected_rate else 0.0
        lines.append(
            f"| {count} | {wide_hits}/{len(round_results)} | {pct(wide_hits, len(round_results))} | "
            f"{bet_hits}/{len(round_results)} | {pct(bet_hits, len(round_results))} | "
            f"{expected_rate * 100:.3f}% | {relative:.2f}x |"
        )
    lines.append("")
    lines.append(f"## 最近{len(recent)}期")
    lines.append("| 每期候选数 | 宽口径命中 | 宽口径命中率 | 可投注组选命中 | 可投注命中率 | 可投注理论覆盖率 | 相对理论 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for count in ticket_counts:
        wide_hits, _ = summarize(recent, count, include_baozi=True)
        bet_hits, expected = summarize(recent, count, include_baozi=False)
        expected_rate = expected / len(recent)
        actual_rate = bet_hits / len(recent)
        relative = actual_rate / expected_rate if expected_rate else 0.0
        lines.append(
            f"| {count} | {wide_hits}/{len(recent)} | {pct(wide_hits, len(recent))} | "
            f"{bet_hits}/{len(recent)} | {pct(bet_hits, len(recent))} | "
            f"{expected_rate * 100:.3f}% | {relative:.2f}x |"
        )
    lines.append("")
    lines.append("## 最近50期明细")
    lines.append("只列出 20 候选内命中的期次，便于复盘候选池是否抓到数字。")
    lines.append("")
    lines.append("| 期号 | 日期 | 开奖号 | 命中候选数 | 命中候选 |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for item in recent:
        actual: tuple[int, int, int] = item["actual"]  # type: ignore[assignment]
        selected: list[tuple[int, int, int]] = item["selected"]  # type: ignore[assignment]
        matched = [number for number in selected[: max(ticket_counts)] if is_group_match(actual, number)]
        if not matched:
            continue
        lines.append(
            f"| {item['issue']} | {item['date']} | {format_number(actual)} | "
            f"{len(matched)} | {', '.join(format_number(number) for number in matched)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 rolling adaptive backtest.")
    parser.add_argument("--data", type=Path, default=Path("docs/p3.md"))
    parser.add_argument("--report", type=Path, default=Path("docs/排列3逐期自适应回测报告.md"))
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument("--fixed", action="store_true", help="Disable online adaptive weight updates.")
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--group-report", type=Path, default=Path("docs/排列3组选滚动回测报告.md"))
    args = parser.parse_args()

    draws = parse_draws(args.data)
    metrics, state, next_ranked = rolling_backtest(
        draws,
        min_history=args.min_history,
        adaptive=not args.fixed,
        learning_rate=args.learning_rate,
    )
    text = report(draws, metrics, state, next_ranked)
    args.report.write_text(text, encoding="utf-8")
    args.group_report.write_text(group_report(draws, metrics), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
