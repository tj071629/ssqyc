"""Compare Top-K per-position compound bets vs random equal ticket counts."""
from __future__ import annotations

import argparse
import itertools
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from qxc_rolling_backtest import (
    FRONT_POS,
    INITIAL_WEIGHTS,
    build_position_scores,
    parse_draws,
    prize_tier,
    random_ticket,
    ticket_score,
    window,
)


@dataclass
class Acc:
    rounds: int = 0
    tickets: int = 0
    exact7: int = 0
    front6_exact: int = 0
    tier2_plus: int = 0
    tier3_plus: int = 0
    tier6_plus: int = 0
    sum_best_pos: float = 0.0
    sum_best_front6: float = 0.0
    pos_cover_hits: int = 0
    pos_cover_total: int = 0
    back_cover_hits: int = 0


def fmt_ticket(t: tuple[int, ...]) -> str:
    return "".join(str(x) for x in t[:6]) + f"+{t[6]:02d}"


def eval_pool(tickets: list[tuple[int, ...]], actual: tuple[int, ...]) -> tuple[int, int, int, int, int, int]:
    best_pos = 0
    best_front6 = 0
    best_tier = 0
    for t in tickets:
        pos = sum(a == b for a, b in zip(t, actual))
        front6 = sum(t[i] == actual[i] for i in range(6))
        tier = prize_tier(t, actual)
        best_pos = max(best_pos, pos)
        best_front6 = max(best_front6, front6)
        best_tier = max(best_tier, tier)
    exact7 = int(any(t == actual for t in tickets))
    front6_exact = int(any(t[:6] == actual[:6] for t in tickets))
    return best_pos, best_front6, best_tier, exact7, front6_exact, len(tickets)


def position_coverage(tickets: list[tuple[int, ...]], actual: tuple[int, ...]) -> tuple[int, int, int]:
    """Count how many positions have actual digit covered by at least one ticket."""
    pos_hit = 0
    for p in range(6):
        if any(t[p] == actual[p] for t in tickets):
            pos_hit += 1
    back_hit = int(any(t[6] == actual[6] for t in tickets))
    return pos_hit, back_hit, 6


def build_compound(
    history,
    *,
    top_front: int,
    top_back: int,
    weights: dict[str, float],
    cap: int | None = None,
) -> list[tuple[int, ...]]:
    ps, bs = build_position_scores(history, weights)
    front_picks = [np.argsort(-s)[:top_front].tolist() for s in ps]
    back_picks = np.argsort(-bs)[:top_back].tolist()
    tickets: list[tuple[int, ...]] = []
    for combo in itertools.product(*front_picks):
        front = tuple(combo)  # type: ignore[assignment]
        for b in back_picks:
            tickets.append(front + (b,))
    tickets.sort(key=lambda t: ticket_score(t[:6], t[6], ps, bs), reverse=True)
    if cap is not None:
        tickets = tickets[:cap]
    return tickets


def build_random_pool(rng: random.Random, n: int) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    tickets: list[tuple[int, ...]] = []
    attempts = 0
    while len(tickets) < n and attempts < n * 20:
        t = random_ticket(rng)
        attempts += 1
        if t not in seen:
            seen.add(t)
            tickets.append(t)
    while len(tickets) < n:
        tickets.append(random_ticket(rng))
    return tickets


def run_compare(
    draws,
    *,
    min_history: int,
    last_n: int | None,
    schemes: list[tuple[str, int, int, int | None]],
    seed: int = 42,
) -> dict[str, Acc]:
    rng = random.Random(seed)
    start = max(min_history, len(draws) - last_n) if last_n else min_history
    accs = {name: Acc() for name, _, _, _ in schemes}
    weights = dict(INITIAL_WEIGHTS)

    for idx in range(start, len(draws)):
        history = draws[:idx]
        actual = draws[idx].number
        for name, top_f, top_b, cap in schemes:
            n = cap if cap else (top_f**6) * top_b
            if name.startswith("复式"):
                tickets = build_compound(history, top_front=top_f, top_back=top_b, weights=weights, cap=cap)
            else:
                tickets = build_random_pool(rng, n)
            bp, bf, bt, e7, f6, n_used = eval_pool(tickets, actual)
            pc, bc, _ = position_coverage(tickets, actual)
            a = accs[name]
            a.rounds += 1
            a.tickets = n_used
            a.exact7 += e7
            a.front6_exact += f6
            a.tier2_plus += int(bt >= 2)
            a.tier3_plus += int(bt >= 3 and bt < 99)
            a.tier6_plus += int(bt >= 6)
            a.sum_best_pos += bp
            a.sum_best_front6 += bf
            a.pos_cover_hits += pc
            a.pos_cover_total += 6
            a.back_cover_hits += bc
    return accs


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.2f}%" if d else "0%"


def print_table(accs: dict[str, Acc], schemes: list[tuple[str, int, int, int | None]]) -> None:
    meta = {s[0]: s for s in schemes}
    print(f"{'方案':<22} {'注数':>6} {'7全中':>10} {'前6中':>10} {'三等+':>10} {'二等+':>10} {'六等+':>10} {'均中位':>7} {'均前6':>7} {'前6覆盖':>8} {'末位覆盖':>8}")
    print("-" * 118)
    for name, top_f, top_b, cap in schemes:
        a = accs[name]
        n = cap if cap else (top_f**6) * top_b
        if not a.rounds:
            continue
        print(
            f"{name:<22} {n:>6} "
            f"{a.exact7:>4}/{a.rounds:<5} "
            f"{a.front6_exact:>4}/{a.rounds:<5} "
            f"{a.tier3_plus:>4}/{a.rounds:<5} "
            f"{a.tier2_plus:>4}/{a.rounds:<5} "
            f"{a.tier6_plus:>4}/{a.rounds:<5} "
            f"{a.sum_best_pos/a.rounds:>7.2f} "
            f"{a.sum_best_front6/a.rounds:>7.2f} "
            f"{pct(a.pos_cover_hits, a.pos_cover_total):>8} "
            f"{pct(a.back_cover_hits, a.rounds):>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="7xc compound vs random coverage compare")
    parser.add_argument("--data", type=Path, default=Path("docs/7xc.md"))
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--windows", type=str, default="200,all")
    args = parser.parse_args()

    draws = parse_draws(args.data.resolve())
    win_map = {"all": None}
    windows = [win_map.get(x.strip(), int(x.strip()) if x.strip().isdigit() else None) for x in args.windows.split(",")]

    # (name, top_front, top_back, cap)
    base_schemes = [
        ("复式Top2x2", 2, 2, None),       # 64*2=128
        ("随机128", 2, 2, 128),
        ("复式Top3x1", 3, 1, None),         # 729
        ("随机729", 3, 1, 729),
        ("复式Top3x3", 3, 3, None),         # 2187
        ("随机2187", 3, 3, 2187),
        ("复式Top2x3", 2, 3, None),         # 128*3=384
        ("随机384", 2, 3, 384),
    ]

    print(f"数据 {len(draws)} 期 ({draws[0].issue}~{draws[-1].issue})")
    print("对比：按位取Top-K复式 vs 同等注数随机\n")

    for win in windows:
        title = f"近{win}期" if win else f"全量({len(draws)-args.min_history}期)"
        print("=" * 118)
        print(title)
        print("=" * 118)
        accs = run_compare(draws, min_history=args.min_history, last_n=win, schemes=base_schemes)
        print_table(accs, base_schemes)
        print()

    print("=" * 118)
    print("解读")
    print("- 前6覆盖：6个位置里，实际开奖数字被注单池覆盖到的比例（复式Top3理论 3/10=30%/位）")
    print("- 若复式在同等注数下「前6中/三等+」仍不高于随机，说明按位热号并未改善组合质量")
    print("- 六等+ 看末位覆盖：Top3末位理论 3/15=20%，随机128注约 1-(12/15)^128≈100%")


if __name__ == "__main__":
    main()
