from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

DIGITS = range(10)
POSITIONS = range(5)
DEFAULT_TICKET_COUNTS = (1, 5, 10, 20, 50)
DEFAULT_MIN_HISTORY = 100
_NUM_TICKETS = 10**5

_TICKET_IDS = np.arange(_NUM_TICKETS, dtype=np.int32)
_DIGIT_MATRIX = np.stack(
    [(_TICKET_IDS // (10 ** (4 - pos))) % 10 for pos in POSITIONS],
    axis=1,
).astype(np.int8)


@dataclass(frozen=True)
class Draw:
    issue: str
    date: str
    number: tuple[int, int, int, int, int]

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
        if len(number_text) != 5 or not number_text.isdigit():
            continue
        draws.append(
            Draw(
                issue=issue,
                date=date,
                number=tuple(int(ch) for ch in number_text),  # type: ignore[arg-type]
            )
        )
    return draws


def pattern_type(number: tuple[int, ...]) -> str:
    uniq = len(set(number))
    return f"u{uniq}"


def span(number: tuple[int, ...]) -> int:
    return max(number) - min(number)


def even_odd_type(number: tuple[int, ...]) -> int:
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


def _batch_features(stats: dict[str, object]) -> dict[str, np.ndarray]:
    last: tuple[int, ...] = stats["last"]  # type: ignore[assignment]
    prev: tuple[int, ...] = stats["prev"]  # type: ignore[assignment]
    pos_counts: dict[int, list[Counter]] = stats["pos_counts"]  # type: ignore[assignment]
    all_counts: dict[int, Counter] = stats["all_counts"]  # type: ignore[assignment]
    omissions: list[dict[int, int]] = stats["omissions"]  # type: ignore[assignment]
    transitions: list[dict[int, Counter]] = stats["transitions"]  # type: ignore[assignment]
    transition_max: list[dict[int, int]] = stats["transition_max"]  # type: ignore[assignment]
    history_len: int = stats["history_len"]  # type: ignore[assignment]
    recent100_len: int = stats["recent100_len"]  # type: ignore[assignment]
    sum_counts: Counter = stats["sum_counts"]  # type: ignore[assignment]
    span_counts: Counter = stats["span_counts"]  # type: ignore[assignment]
    pattern_counts: Counter = stats["pattern_counts"]  # type: ignore[assignment]
    even_counts: Counter = stats["even_counts"]  # type: ignore[assignment]

    dm = _DIGIT_MATRIX
    batch: dict[str, np.ndarray] = {}

    for size, name in ((10, "pos_hot10"), (30, "pos_hot30"), (100, "pos_hot100")):
        denom = float(min(size, history_len))
        parts = []
        for pos in POSITIONS:
            cnt = pos_counts[size][pos]
            lut = np.array([float(cnt[d]) for d in DIGITS], dtype=np.float64)
            parts.append(lut[dm[:, pos].astype(np.int64)] / denom)
        batch[name] = np.mean(parts, axis=0)

    denom30 = float(min(30, history_len) * 5)
    ac30 = all_counts[30]
    lut_ac = np.array([float(ac30[d]) for d in DIGITS], dtype=np.float64)
    batch["all_hot30"] = lut_ac[dm.astype(np.int64)].mean(axis=1) / denom30

    last_np = np.array(last, dtype=np.int8)
    prev_np = np.array(prev, dtype=np.int8)
    batch["repeat_last"] = np.mean(dm == last_np, axis=1)
    batch["neighbor_last"] = np.mean(np.abs(dm - last_np) == 1, axis=1)
    batch["neighbor_prev"] = np.mean(np.abs(dm - prev_np) == 1, axis=1)

    trans_vals = []
    for pos in POSITIONS:
        from_digit = int(last[pos])
        tmap = transitions[pos][from_digit]
        tmax = float(transition_max[pos][from_digit])
        lut = np.array([float(tmap[d]) for d in DIGITS], dtype=np.float64)
        trans_vals.append(lut[dm[:, pos].astype(np.int64)] / tmax)
    batch["transition"] = np.mean(trans_vals, axis=0)

    mid_scores = []
    deep_scores = []
    for pos in POSITIONS:
        om = omissions[pos]
        lut = np.array([float(om[d]) for d in DIGITS], dtype=np.float64)
        miss = lut[dm[:, pos].astype(np.int64)]
        mid_scores.append(np.exp(-((miss - 12.0) ** 2) / (2.0 * 8.0**2)))
        deep_scores.append(np.minimum(miss / 80.0, 1.0))
    batch["omission_mid"] = np.mean(mid_scores, axis=0)
    batch["omission_deep"] = np.mean(deep_scores, axis=0)

    sums = dm.sum(axis=1).astype(np.int16)
    spans = dm.max(axis=1) - dm.min(axis=1)
    ds = np.sort(dm, axis=1)
    uniq = 1 + (ds[:, 1:] != ds[:, :-1]).sum(axis=1)
    evens = (dm % 2 == 0).sum(axis=1)

    sum_lut = np.array(
        [normalized_counter_value(sum_counts, s, recent100_len) for s in range(46)],
        dtype=np.float64,
    )
    span_lut = np.array(
        [normalized_counter_value(span_counts, s, recent100_len) for s in range(10)],
        dtype=np.float64,
    )
    pattern_lut = np.array(
        [normalized_counter_value(pattern_counts, f"u{k}", recent100_len) for k in range(1, 6)],
        dtype=np.float64,
    )
    even_lut = np.array(
        [normalized_counter_value(even_counts, k, recent100_len) for k in range(6)],
        dtype=np.float64,
    )

    batch["sum_hot100"] = sum_lut[sums.astype(np.int64)]
    batch["span_hot100"] = span_lut[spans.astype(np.int64)]
    batch["pattern_hot100"] = pattern_lut[uniq.astype(np.int64) - 1]
    batch["even_odd_fit"] = even_lut[evens.astype(np.int64)]
    return batch


def _ticket_index(number: tuple[int, int, int, int, int]) -> int:
    return (
        number[0] * 10000
        + number[1] * 1000
        + number[2] * 100
        + number[3] * 10
        + number[4]
    )


def _features_dict_from_batch(batch: dict[str, np.ndarray], idx: int) -> dict[str, float]:
    return {name: float(arr[idx]) for name, arr in batch.items()}


def score_all(batch: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    score = np.zeros(_NUM_TICKETS, dtype=np.float64)
    for name, arr in batch.items():
        score += weights.get(name, 0.0) * arr
    return score


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


def one_roll_step(
    draws: list[Draw],
    idx: int,
    state: StrategyState,
    ticket_counts: tuple[int, ...],
) -> tuple[int, tuple[tuple[int, int, int, int, int], ...]]:
    """Predict draws[idx] from draws[:idx]; update state. Returns actual's rank and top tickets."""
    max_tickets = max(ticket_counts)
    history = draws[:idx]
    actual = draws[idx]
    order, batch, _scores = rank_for_round(history, state)
    actual_i = _ticket_index(actual.number)
    rank = int(np.where(order == actual_i)[0][0]) + 1

    learn_cap = min(state.update_top_n, _NUM_TICKETS)
    learn_indices = order[:learn_cap].astype(np.int64)
    selected_indices = order[:max_tickets].astype(np.int64)
    selected_numbers = tuple(
        tuple(int(_DIGIT_MATRIX[i, p]) for p in POSITIONS) for i in selected_indices
    )

    actual_features = _features_dict_from_batch(batch, actual_i)
    selected_feats = (_features_dict_from_batch(batch, int(i)) for i in learn_indices)
    update_state(state, actual_features, selected_feats)

    return rank, selected_numbers


def format_number(number: tuple[int, ...]) -> str:
    return "".join(str(digit) for digit in number)


def rank_for_round(
    history: list[Draw],
    state: StrategyState,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    stats = build_stats(history)
    batch = _batch_features(stats)
    scores = score_all(batch, state.weights)
    order = np.lexsort((np.arange(_NUM_TICKETS, dtype=np.int32), -scores))
    return order, batch, scores


def next_issue_predictions(
    draws: list[Draw],
    state: StrategyState,
    *,
    count: int,
    never_drawn_only: bool,
) -> list[tuple[str, float]]:
    """After rolling_backtest, rank all tickets on full history and take top `count` lines."""
    if count <= 0:
        return []
    seen = {_ticket_index(d.number) for d in draws}
    order, _, scores = rank_for_round(draws, state)
    out: list[tuple[str, float]] = []
    for idx in order:
        i = int(idx)
        if never_drawn_only and i in seen:
            continue
        num = tuple(int(_DIGIT_MATRIX[i, p]) for p in POSITIONS)
        out.append((format_number(num), float(scores[i])))
        if len(out) >= count:
            break
    return out


def stratified_three_entries(
    order: np.ndarray,
    scores: np.ndarray,
    *,
    pool: int = 100,
) -> list[dict[str, object]]:
    """First ticket in each third of top-`pool` by `order` (best first). pool=100 -> ranks 1,34,67."""
    if pool < 3:
        raise ValueError("pool must be at least 3.")
    if len(order) < pool:
        raise ValueError("order shorter than pool.")
    step = pool // 3
    positions = (0, step, 2 * step)
    rows: list[dict[str, object]] = []
    for seg_idx, pos in enumerate(positions, start=1):
        ti = int(order[pos])
        num = tuple(int(_DIGIT_MATRIX[ti, p]) for p in POSITIONS)
        rows.append(
            {
                "segment": seg_idx,
                "rank_in_top100": pos + 1,
                "number": format_number(num),
                "score": float(scores[ti]),
            }
        )
    return rows


def rolling_backtest(
    draws: list[Draw],
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    ticket_counts: tuple[int, ...] = DEFAULT_TICKET_COUNTS,
    adaptive: bool = True,
    learning_rate: float = 0.035,
    decay: float = 0.997,
    stratified_pool: int = 100,
) -> tuple[dict[str, object], StrategyState, list[tuple[tuple[int, ...], float, dict[str, float]]]]:
    if len(draws) <= min_history:
        raise ValueError(f"Need more than {min_history} draws, got {len(draws)}.")

    state = StrategyState(
        adaptive=adaptive,
        weights=dict(INITIAL_WEIGHTS),
        learning_rate=learning_rate if adaptive else 0.0,
        decay=decay,
    )
    max_tickets = max(ticket_counts)
    exact_hits = {count: 0 for count in ticket_counts}
    best_position_hits = {count: Counter() for count in ticket_counts}
    actual_ranks: list[int] = []
    top1_examples: list[dict[str, str]] = []
    round_results: list[dict[str, object]] = []

    for idx in range(min_history, len(draws)):
        rank, selected_numbers = one_roll_step(draws, idx, state, ticket_counts)
        actual = draws[idx]
        actual_ranks.append(rank)

        for count in ticket_counts:
            tickets = selected_numbers[:count]
            if actual.number in tickets:
                exact_hits[count] += 1
            best = max(
                sum(1 for pos in POSITIONS if ticket[pos] == actual.number[pos]) for ticket in tickets
            )
            best_position_hits[count][best] += 1

        round_results.append(
            {
                "issue": actual.issue,
                "date": actual.date,
                "actual": actual.number,
                "selected": list(selected_numbers),
            }
        )

        if len(top1_examples) < 8:
            top_num = selected_numbers[0]
            top1_examples.append(
                {
                    "issue": actual.issue,
                    "date": actual.date,
                    "pred": format_number(top_num),
                    "actual": actual.text,
                    "rank": str(rank),
                }
            )

    final_order, final_batch, final_scores = rank_for_round(draws, state)
    next_ranked: list[tuple[tuple[int, ...], float, dict[str, float]]] = []
    for i in range(min(20, len(final_order))):
        idx = int(final_order[i])
        num = tuple(int(_DIGIT_MATRIX[idx, p]) for p in POSITIONS)
        next_ranked.append((num, float(final_scores[idx]), _features_dict_from_batch(final_batch, idx)))

    strat_pool = max(3, stratified_pool)
    stratified_three = stratified_three_entries(final_order, final_scores, pool=strat_pool)

    rounds = len(draws) - min_history
    metrics: dict[str, object] = {
        "rounds": rounds,
        "min_history": min_history,
        "adaptive": adaptive,
        "learning_rate": learning_rate if adaptive else 0.0,
        "decay": decay,
        "exact_hits": exact_hits,
        "best_position_hits": best_position_hits,
        "actual_ranks": actual_ranks,
        "mean_rank": statistics.fmean(actual_ranks),
        "median_rank": statistics.median(actual_ranks),
        "rank_top20": sum(1 for r in actual_ranks if r <= 20),
        "rank_top100": sum(1 for r in actual_ranks if r <= 100),
        "rank_top1000": sum(1 for r in actual_ranks if r <= 1000),
        "examples": top1_examples,
        "round_results": round_results,
        "stratified_three": stratified_three,
        "stratified_pool": strat_pool,
    }
    return metrics, state, next_ranked


def pct(numerator: int | float, denominator: int | float) -> str:
    if denominator == 0:
        return "0.000%"
    return f"{numerator / denominator * 100:.3f}%"


def report(
    draws: list[Draw],
    metrics: dict[str, object],
    state: StrategyState,
    next_ranked: list[tuple[tuple[int, ...], float, dict[str, float]]],
    *,
    ticket_counts: tuple[int, ...] = DEFAULT_TICKET_COUNTS,
) -> str:
    rounds: int = metrics["rounds"]  # type: ignore[assignment]
    exact_hits: dict[int, int] = metrics["exact_hits"]  # type: ignore[assignment]
    best_position_hits: dict[int, Counter] = metrics["best_position_hits"]  # type: ignore[assignment]
    is_adaptive: bool = metrics["adaptive"]  # type: ignore[assignment]
    mode_name = "逐期自适应" if is_adaptive else "固定权重"
    method_text = (
        f"前 {metrics['min_history']} 期只作冷启动训练，之后每期只用过去数据预测下一期，开奖后再更新权重。"
        if is_adaptive
        else f"前 {metrics['min_history']} 期只作冷启动训练，之后每期只用过去数据预测下一期，权重全程固定不更新。"
    )

    lines: list[str] = []
    lines.append(f"# 排列5{mode_name}滚动回测报告")
    lines.append("")
    lines.append("## 数据范围")
    lines.append(f"- 历史期数：{len(draws)}")
    lines.append(f"- 起始期：{draws[0].issue}（{draws[0].date}），号码 {draws[0].text}")
    lines.append(f"- 最新期：{draws[-1].issue}（{draws[-1].date}），号码 {draws[-1].text}")
    lines.append(f"- 回测方式：{method_text}")
    lines.append("")
    lines.append("## 规则信号")
    lines.append("- 按位冷热：最近 10/30/100 期每个位置数字频率。")
    lines.append("- 全局冷热：最近 30 期全体位置上的数字频率（按位平均）。")
    lines.append("- 承接关系：上一期同位重号、同位 ±1 邻号、上两期邻号。")
    lines.append("- 转移关系：同位置上一位数字到下一位数字的近 250 期转移频率。")
    lines.append("- 遗漏回补：同位置数字的中等遗漏窗口和深遗漏窗口。")
    lines.append("- 结构约束：和值、跨度、不同数字个数（形态 u1–u5）、偶数个数的近 100 期分布。")
    lines.append("")
    lines.append("## 直选命中率")
    lines.append("| 每期票数 | 直选命中次数 | 回测命中率 | 随机理论值 | 相对随机 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for count in ticket_counts:
        hit = exact_hits[count]
        random_rate = count / _NUM_TICKETS
        relative = (hit / rounds) / random_rate if random_rate else 0.0
        lines.append(f"| {count} | {hit} / {rounds} | {pct(hit, rounds)} | {random_rate * 100:.3f}% | {relative:.2f}x |")
    lines.append("")
    lines.append("## 位置命中分布")
    lines.append("每期在 N 注候选中，取与开奖号同位置相同最多的那一注。")
    lines.append("")
    lines.append("| 每期票数 | 0位 | 1位 | 2位 | 3位 | 4位 | 5位 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for count in ticket_counts:
        counter = best_position_hits[count]
        lines.append(
            f"| {count} | {counter[0]} | {counter[1]} | {counter[2]} | {counter[3]} | {counter[4]} | {counter[5]} |"
        )
    lines.append("")
    lines.append("## 实际号码得分排名")
    lines.append(f"- 平均排名：{metrics['mean_rank']:.1f} / {_NUM_TICKETS}")
    lines.append(f"- 中位排名：{metrics['median_rank']:.1f} / {_NUM_TICKETS}")
    lines.append(f"- 实际号码落入模型前20：{metrics['rank_top20']} 次，{pct(metrics['rank_top20'], rounds)}")
    lines.append(f"- 实际号码落入模型前100：{metrics['rank_top100']} 次，{pct(metrics['rank_top100'], rounds)}")
    lines.append(f"- 实际号码落入模型前1000：{metrics['rank_top1000']} 次，{pct(metrics['rank_top1000'], rounds)}")
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
    st_list: list[dict[str, object]] = metrics["stratified_three"]  # type: ignore[assignment]
    spool: int = metrics["stratified_pool"]  # type: ignore[assignment]
    step = spool // 3
    lines.append(f"## 下期三期分段各一注（前{spool}名）")
    lines.append(
        f"- 将模型排名前 **{spool}** 注按名次分为三段：**1–{step}**、**{step + 1}–{2 * step}**、**{2 * step + 1}–{spool}**；"
        f"每段取该段**名次最靠前**的一注，共 3 注（即全表排序约第 **1**、**{step + 1}**、**{2 * step + 1}** 名）。"
    )
    lines.append("- 该取法用于分散注单、避免三注全挤在最前几名；不表示提高中奖概率。")
    lines.append("")
    lines.append(f"| 段 | 前{spool}内名次 | 号码 | 分数 |")
    lines.append("| ---: | ---: | --- | ---: |")
    for row in st_list:
        lines.append(
            f"| {row['segment']} | {row['rank_in_top100']} | {row['number']} | {row['score']:.4f} |"
        )
    lines.append("")
    lines.append("## 结论")
    lines.append(
        f"排列5直选理论中奖率是每注 1/{_NUM_TICKETS}。若回测命中率只是在随机理论值附近波动，"
        "说明规则更多是在做候选排序，而不是形成稳定可迁移优势。逐期自适应仅根据历史预测误差微调权重，"
        "仍需用滚动回测或留出最近期验证，避免按全历史命中结果反向调参。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="PL5 rolling adaptive backtest.")
    parser.add_argument("--data", type=Path, default=Path("docs/p5.md"))
    parser.add_argument("--report", type=Path, default=Path("docs/排列5逐期自适应回测报告.md"))
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument("--fixed", action="store_true", help="Disable online adaptive weight updates.")
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--decay", type=float, default=0.997, help="Weight anchoring decay in adaptive updates.")
    parser.add_argument(
        "--ticket-counts",
        type=str,
        default=",".join(str(n) for n in DEFAULT_TICKET_COUNTS),
        help="Comma-separated counts of top-ranked tickets to treat as bets each round (e.g. 3 or 1,3,10).",
    )
    parser.add_argument(
        "--predict-top",
        type=int,
        default=0,
        metavar="N",
        help="After backtest, print N next-issue picks (full-history score order). 0 = skip.",
    )
    parser.add_argument(
        "--never-drawn-only",
        action="store_true",
        help="With --predict-top, skip numbers that already appeared in --data history.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not print the full markdown report to stdout (still writes --report).",
    )
    parser.add_argument(
        "--stratified-pool",
        type=int,
        default=100,
        metavar="N",
        help="Pool size for 'three-segment one pick each' in report (default 100).",
    )
    args = parser.parse_args()

    ticket_parts = [part.strip() for part in args.ticket_counts.split(",") if part.strip()]
    if not ticket_parts:
        raise SystemExit("ticket-counts must list at least one positive integer.")
    ticket_counts = tuple(int(part) for part in ticket_parts)
    if any(count <= 0 for count in ticket_counts):
        raise SystemExit("ticket-counts must be positive integers.")

    draws = parse_draws(args.data)
    metrics, state, next_ranked = rolling_backtest(
        draws,
        min_history=args.min_history,
        ticket_counts=ticket_counts,
        adaptive=not args.fixed,
        learning_rate=args.learning_rate,
        decay=args.decay,
        stratified_pool=args.stratified_pool,
    )
    text = report(draws, metrics, state, next_ranked, ticket_counts=ticket_counts)
    args.report.write_text(text, encoding="utf-8")
    if args.summary_only:
        last = draws[-1]
        print(
            f"[PL5] rounds={metrics['rounds']} last_issue={last.issue} last_draw={last.text} "
            f"report={args.report}"
        )
        print("")
        print("=== 下期：前100三期分段各1注 ===")
        for row in metrics["stratified_three"]:  # type: ignore[index]
            print(
                f"段{row['segment']}  前100内第{row['rank_in_top100']}名  {row['number']}  "
                f"score={row['score']:.4f}"
            )
    else:
        print(text)

    if args.predict_top > 0:
        preds = next_issue_predictions(
            draws,
            state,
            count=args.predict_top,
            never_drawn_only=args.never_drawn_only,
        )
        tag = "（仅历史上未开出过的号码）" if args.never_drawn_only else ""
        print("")
        print(f"=== 下期预测：模型 Top {args.predict_top} {tag} ===")
        for i, (num, sc) in enumerate(preds, start=1):
            print(f"{i}. {num}  score={sc:.4f}")


if __name__ == "__main__":
    main()
