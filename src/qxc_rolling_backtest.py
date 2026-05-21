"""7星彩 rolling backtest (fast per-position scoring)."""
from __future__ import annotations

import argparse
import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FRONT_POS = range(6)
DIGITS = range(10)
BACK_DIGITS = range(15)
_NUM_ALL = 10**6 * 15


@dataclass(frozen=True)
class Draw:
    issue: str
    date: str
    number: tuple[int, int, int, int, int, int, int]


@dataclass
class StrategyState:
    adaptive: bool
    weights: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.035
    decay: float = 0.997
    update_top_n: int = 30
    min_weight: float = -2.5
    max_weight: float = 2.5


INITIAL_WEIGHTS = {
    "pos_hot10": 0.55,
    "pos_hot30": 0.75,
    "pos_hot100": 0.45,
    "repeat_last": 0.20,
    "neighbor_last": 0.35,
    "neighbor_prev": 0.15,
    "transition": 0.45,
    "omission_mid": 0.55,
    "omission_deep": 0.05,
    "back_hot30": 0.65,
    "back_repeat": 0.20,
    "back_neighbor": 0.30,
    "back_omission": 0.40,
}


def parse_draws(path: Path) -> list[Draw]:
    draws: list[Draw] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        nums = [int(x) for x in cells[2].split()]
        if len(nums) == 7:
            draws.append(Draw(issue=cells[0], date=cells[1], number=tuple(nums)))  # type: ignore[arg-type]
    return draws


def window(history: list[Draw], size: int) -> list[Draw]:
    return history[-size:] if len(history) > size else history


def build_position_scores(history: list[Draw], weights: dict[str, float]) -> tuple[list[np.ndarray], np.ndarray]:
    last = history[-1].number
    prev = history[-2].number if len(history) >= 2 else last
    pos_scores: list[np.ndarray] = []
    n = len(history)

    for pos in FRONT_POS:
        s = np.zeros(10, dtype=np.float64)
        for size, wname in ((10, "pos_hot10"), (30, "pos_hot30"), (100, "pos_hot100")):
            draws = window(history, size)
            cnt = Counter(d.number[pos] for d in draws)
            denom = max(len(draws), 1)
            for d in DIGITS:
                s[d] += weights.get(wname, 0.0) * cnt[d] / denom

        miss = np.full(10, n, dtype=np.int32)
        for i, drow in enumerate(reversed(history)):
            digit = drow.number[pos]
            if miss[digit] == n:
                miss[digit] = i
        for d in DIGITS:
            m = int(miss[d])
            s[d] += weights.get("omission_mid", 0.0) * np.exp(-((m - 12.0) ** 2) / 128.0)
            s[d] += weights.get("omission_deep", 0.0) * min(m / 80.0, 1.0)

        transitions: Counter = Counter()
        th = window(history, 250)
        for a, b in zip(th, th[1:]):
            if a.number[pos] == last[pos]:
                transitions[b.number[pos]] += 1
        tmax = max(transitions.values(), default=1)
        for d in DIGITS:
            s[d] += weights.get("transition", 0.0) * transitions[d] / tmax

        ld, pd = last[pos], prev[pos]
        for d in DIGITS:
            if d == ld:
                s[d] += weights.get("repeat_last", 0.0)
            if abs(d - ld) == 1:
                s[d] += weights.get("neighbor_last", 0.0)
            if abs(d - pd) == 1:
                s[d] += weights.get("neighbor_prev", 0.0)
        pos_scores.append(s)

    back = np.zeros(15, dtype=np.float64)
    draws30 = window(history, 30)
    bc = Counter(d.number[6] for d in draws30)
    denom = max(len(draws30), 1)
    for d in BACK_DIGITS:
        back[d] += weights.get("back_hot30", 0.0) * bc[d] / denom
    lb = last[6]
    bmiss = np.full(15, n, dtype=np.int32)
    for i, drow in enumerate(reversed(history)):
        digit = drow.number[6]
        if bmiss[digit] == n:
            bmiss[digit] = i
    for d in BACK_DIGITS:
        if d == lb:
            back[d] += weights.get("back_repeat", 0.0)
        if abs(d - lb) == 1:
            back[d] += weights.get("back_neighbor", 0.0)
        back[d] += weights.get("back_omission", 0.0) * np.exp(-((int(bmiss[d]) - 8.0) ** 2) / 72.0)
    return pos_scores, back


def ticket_score(front: tuple[int, ...], back: int, pos_scores: list[np.ndarray], back_scores: np.ndarray) -> float:
    score = sum(pos_scores[i][front[i]] for i in FRONT_POS) + back_scores[back]
    nums = list(front)
    score += 0.28 * _struct_bonus(sum(nums), max(nums) - min(nums), len(set(nums)), sum(1 for x in nums if x % 2 == 0))
    return score


def _struct_bonus(total: int, span: int, uniq: int, evens: int) -> float:
    # lightweight structural prior from PL5-style weights
    sum_score = np.exp(-((total - 27.0) ** 2) / 180.0)
    span_score = np.exp(-((span - 7.0) ** 2) / 18.0)
    uniq_score = {6: 0.15, 5: 0.10, 4: 0.05, 3: 0.0, 2: -0.05, 1: -0.10}.get(uniq, 0.0)
    even_score = np.exp(-((evens - 3.0) ** 2) / 2.0)
    return sum_score + span_score + uniq_score + even_score


def generate_candidates(pos_scores: list[np.ndarray], back_scores: np.ndarray, top_per_pos: int = 3, top_back: int = 3) -> list[tuple[tuple[int, ...], int, float]]:
    front_picks = [np.argsort(-s)[:top_per_pos].tolist() for s in pos_scores]
    back_picks = np.argsort(-back_scores)[:top_back].tolist()
    out: list[tuple[tuple[int, ...], int, float]] = []
    for combo in itertools.product(*front_picks):
        front = tuple(combo)  # type: ignore[assignment]
        for b in back_picks:
            out.append((front, b, ticket_score(front, b, pos_scores, back_scores)))
    out.sort(key=lambda x: -x[2])
    return out


def prize_tier(bet: tuple[int, ...], actual: tuple[int, ...]) -> int:
    fm = sum(bet[i] == actual[i] for i in range(6))
    bm = bet[6] == actual[6]
    tm = fm + int(bm)
    if tm == 7:
        return 1
    if fm == 6:
        return 2
    if fm >= 5 and bm:
        return 3
    if tm >= 5:
        return 4
    if tm >= 4:
        return 5
    if tm >= 3 or (fm >= 1 and bm) or bm:
        return 6
    return 0


def pos_hits(bet: tuple[int, ...], actual: tuple[int, ...]) -> int:
    return sum(a == b for a, b in zip(bet, actual))


def pos_hot30_ticket(history: list[Draw]) -> tuple[int, ...]:
    draws = window(history, 30)
    front = tuple(Counter(d.number[p] for d in draws).most_common(1)[0][0] for p in FRONT_POS)
    back = Counter(d.number[6] for d in draws).most_common(1)[0][0]
    return front + (back,)


def random_ticket(rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.randint(0, 9) for _ in FRONT_POS) + (rng.randint(0, 14),)


def feature_vector(front: tuple[int, ...], back: int, pos_scores: list[np.ndarray], back_scores: np.ndarray) -> dict[str, float]:
    return {
        "pos_hot10": float(np.mean([pos_scores[p][front[p]] for p in FRONT_POS])),
        "pos_hot30": float(np.mean([pos_scores[p][front[p]] for p in FRONT_POS])),
        "pos_hot100": float(np.mean([pos_scores[p][front[p]] for p in FRONT_POS])),
        "repeat_last": float(sum(1 for p in FRONT_POS if front[p] == 0)),  # placeholder replaced below
    }


def actual_feature_vector(
    actual: tuple[int, ...],
    history: list[Draw],
    weights: dict[str, float],
    pos_scores: list[np.ndarray] | None = None,
    back_scores: np.ndarray | None = None,
) -> dict[str, float]:
    if pos_scores is None or back_scores is None:
        pos_scores, back_scores = build_position_scores(history, weights)
    return {
        "pos_hot10": float(np.mean([pos_scores[p][actual[p]] for p in FRONT_POS])),
        "pos_hot30": float(np.mean([pos_scores[p][actual[p]] for p in FRONT_POS])),
        "pos_hot100": float(np.mean([pos_scores[p][actual[p]] for p in FRONT_POS])),
        "repeat_last": float(sum(1 for p in FRONT_POS if actual[p] == history[-1].number[p]) / 6.0),
        "neighbor_last": float(sum(1 for p in FRONT_POS if abs(actual[p] - history[-1].number[p]) == 1) / 6.0),
        "neighbor_prev": float(
            sum(1 for p in FRONT_POS if abs(actual[p] - history[-2].number[p]) == 1) / 6.0
        ) if len(history) >= 2 else 0.0,
        "transition": float(np.mean([pos_scores[p][actual[p]] for p in FRONT_POS])),
        "omission_mid": 0.0,
        "omission_deep": 0.0,
        "back_hot30": float(back_scores[actual[6]]),
        "back_repeat": float(actual[6] == history[-1].number[6]),
        "back_neighbor": float(abs(actual[6] - history[-1].number[6]) == 1),
        "back_omission": float(back_scores[actual[6]]),
    }


def update_state(state: StrategyState, actual: dict[str, float], selected: list[dict[str, float]]) -> None:
    if not state.adaptive or not selected:
        return
    for name in INITIAL_WEIGHTS:
        exp = sum(x.get(name, 0.0) for x in selected) / len(selected)
        cur = state.weights.get(name, 0.0)
        tgt = INITIAL_WEIGHTS[name]
        val = cur * state.decay + tgt * (1.0 - state.decay) + state.learning_rate * (actual.get(name, 0.0) - exp)
        state.weights[name] = max(state.min_weight, min(state.max_weight, val))


@dataclass
class Acc:
    rounds: int = 0
    exact7: int = 0
    front6_exact: int = 0
    tier6_plus: int = 0
    tier2_plus: int = 0
    sum_pos: float = 0.0
    sum_front6: float = 0.0
    sum_rank_pct: float = 0.0
    pos_dist: Counter = field(default_factory=Counter)


def eval_pick(tickets: list[tuple[int, ...]], actual: tuple[int, ...]) -> tuple[int, int, int, int, int]:
    bp = max(pos_hits(t, actual) for t in tickets)
    bf = max(sum(t[i] == actual[i] for i in range(6)) for t in tickets)
    bt = max(prize_tier(t, actual) for t in tickets)
    e7 = int(any(t == actual for t in tickets))
    f6 = int(any(t[:6] == actual[:6] for t in tickets))
    return bp, bf, bt, e7, f6


def estimate_rank_pct(
    actual_score: float,
    candidate_scores: list[float],
    pos_scores: list[np.ndarray],
    back_scores: np.ndarray,
    rng: random.Random,
    samples: int = 100,
) -> float:
    rand_scores = [
        ticket_score(tuple(rng.randint(0, 9) for _ in FRONT_POS), rng.randint(0, 14), pos_scores, back_scores)
        for _ in range(samples)
    ]
    pool = candidate_scores + rand_scores
    worse = sum(1 for s in pool if s > actual_score)
    return worse / max(len(pool), 1)


def run_window(
    draws: list[Draw],
    *,
    min_history: int,
    last_n: int | None,
    ticket_counts: tuple[int, ...],
    seed: int = 42,
) -> tuple[dict[str, Acc], list[Counter]]:
    rng = random.Random(seed)
    fixed = StrategyState(adaptive=False, weights=dict(INITIAL_WEIGHTS))
    adapt = StrategyState(adaptive=True, weights=dict(INITIAL_WEIGHTS))
    start = max(min_history, len(draws) - last_n) if last_n else min_history

    accs = {f"{m}_{n}": Acc() for m in ("random", "hot30", "fixed", "adapt") for n in ticket_counts}
    pos_hit = [Counter() for _ in range(7)]

    for idx in range(start, len(draws)):
        history = draws[:idx]
        actual = draws[idx].number

        ps_f, bs_f = build_position_scores(history, fixed.weights)
        ps_a, bs_a = build_position_scores(history, adapt.weights)
        cands_f = generate_candidates(ps_f, bs_f)
        cands_a = generate_candidates(ps_a, bs_a)
        actual_sc = ticket_score(actual[:6], actual[6], ps_a, bs_a)
        rank_pct = estimate_rank_pct(actual_sc, [c[2] for c in cands_a], ps_a, bs_a, rng)
        hot = pos_hot30_ticket(history)

        act_f = actual_feature_vector(actual, history, adapt.weights, ps_a, bs_a)
        sel = [actual_feature_vector(c[0] + (c[1],), history, adapt.weights, ps_a, bs_a) for c in cands_a[: adapt.update_top_n]]
        update_state(adapt, act_f, sel)

        for n in ticket_counts:
            packs = {
                "random": [random_ticket(rng) for _ in range(n)],
                "hot30": [hot] * n,
                "fixed": [c[0] + (c[1],) for c in cands_f[:n]],
                "adapt": [c[0] + (c[1],) for c in cands_a[:n]],
            }
            for name, tickets in packs.items():
                a = accs[f"{name}_{n}"]
                bp, bf, bt, e7, f6 = eval_pick(tickets, actual)
                a.rounds += 1
                a.exact7 += e7
                a.front6_exact += f6
                a.tier6_plus += int(bt >= 6)
                a.tier2_plus += int(bt >= 2)
                a.sum_pos += bp
                a.sum_front6 += bf
                if name == "adapt":
                    a.sum_rank_pct += rank_pct
                a.pos_dist[bp] += 1
            if n == 1:
                t = packs["adapt"][0]
                for p in range(7):
                    pos_hit[p][int(t[p] == actual[p])] += 1

    return accs, pos_hit


def pct(n: int, d: int) -> str:
    return f"{100*n/d:.2f}%" if d else "0%"


def print_results(accs: dict[str, Acc], ticket_counts: tuple[int, ...]) -> None:
    labels = {"random": "随机", "hot30": "热号30", "fixed": "融合固定", "adapt": "融合自适应"}
    print(f"{'策略':<14} {'期':>5} {'7全中':>8} {'前6中':>8} {'六等+':>8} {'二等+':>8} {'均中位':>6} {'均前6':>6} {'排名%':>7}")
    print("-" * 88)
    for n in ticket_counts:
        for key in ("random", "hot30", "fixed", "adapt"):
            a = accs[f"{key}_{n}"]
            rp = a.sum_rank_pct / a.rounds if key == "adapt" and a.rounds else 0.0
            print(
                f"{labels[key]+'x'+str(n):<14} {a.rounds:>5} "
                f"{a.exact7:>3}/{a.rounds:<4} {a.front6_exact:>3}/{a.rounds:<4} "
                f"{a.tier6_plus:>3}/{a.rounds:<4} {a.tier2_plus:>3}/{a.rounds:<4} "
                f"{a.sum_pos/a.rounds:>6.2f} {a.sum_front6/a.rounds:>6.2f} "
                f"{rp*100:>6.1f}%"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("docs/7xc.md"))
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--windows", type=str, default="50,100,200,all")
    args = parser.parse_args()
    win_map = {"all": None, "50": 50, "100": 100, "200": 200}
    windows = [win_map.get(x.strip(), int(x.strip()) if x.strip().isdigit() else None) for x in args.windows.split(",")]

    draws = parse_draws(args.data.resolve())
    ticket_counts = (1, 5, 10, 50)
    print(f"数据 {len(draws)} 期 ({draws[0].issue}~{draws[-1].issue})")
    print(f"理论: 7位全中 1/{_NUM_ALL}, 前6全中 1/1000000, 随机期望同位命中 ~0.77/7\n")

    for win in windows:
        title = f"近{win}期" if win else "全量(3240期)"
        print("=" * 88)
        print(title)
        print("=" * 88)
        accs, pos_hit = run_window(draws, min_history=args.min_history, last_n=win, ticket_counts=ticket_counts)
        print_results(accs, ticket_counts)
        if win is None:
            r = accs["adapt_1"].rounds
            print("融合自适应 Top1 按位命中率:")
            for p in range(6):
                print(f"  第{p+1}位 {pos_hit[p][1]}/{r}={pct(pos_hit[p][1],r)} (随机10%)")
            print(f"  第7位 {pos_hit[6][1]}/{r}={pct(pos_hit[6][1],r)} (随机6.7%)")
            print()

    print("=" * 88)
    print("解读要点")
    print("- 随机期望: 均中位≈0.77, 均前6≈0.60; 六等+单注约6~7%")
    print("- 若融合策略均中位/均前6明显高于随机, 说明排序有效; 但仍难转化为全中")
    print("- 排名% 越低越好(模型给实际开奖打的分越高)")


if __name__ == "__main__":
    main()
