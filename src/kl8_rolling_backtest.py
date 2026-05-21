from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable


BALL_RANGE = range(1, 81)
PICK_COUNT = 4
PICK1_COUNT = 1
COMPOUND9_POOL = 10
COMPOUND9_LINES = 10
DANTUO_BANK = 2
DANTUO_DRAG = 5
DANTUO_LINES = 10
DRAW_SIZE = 20
P_RANDOM_PICK1 = DRAW_SIZE / 80.0
# 与常见快乐8选九奖级一致：中 4～9 及「全不中」；若规则变动可用 --compound-no-zero 排除中零
PRIZE_COUNTS_9_WITH_ZERO: frozenset[int] = frozenset({0, 4, 5, 6, 7, 8, 9})
PRIZE_COUNTS_9_NO_ZERO: frozenset[int] = frozenset({4, 5, 6, 7, 8, 9})
WINDOWS = (10, 30, 50, 100)
DEFAULT_MIN_HISTORY = 200


@dataclass(frozen=True)
class Draw:
    issue: str
    date: str
    numbers: frozenset[int]

    @property
    def sorted(self) -> tuple[int, ...]:
        return tuple(sorted(self.numbers))


@dataclass
class StrategyState:
    adaptive: bool
    weights: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.02
    decay: float = 0.998
    update_rank: int = 20
    min_weight: float = -3.0
    max_weight: float = 3.0


INITIAL_WEIGHTS = {
    "hot10": 0.45,
    "hot30": 0.58,
    "hot50": 0.52,
    "hot100": 0.4,
    "repeat_last": 0.36,
    "neighbor1": 0.3,
    "neighbor2": 0.17,
    "pair_last": 0.2,
    "omission_mid": 0.48,
    "omission_deep": 0.09,
    "zone_balance": 0.14,
    "burst3": 0.22,
    "plank_cold": 0.2,
    "carry2": 0.12,
}


def plank_id(n: int) -> int:
    return (n - 1) // 20


def parse_draws(path: Path) -> list[Draw]:
    draws: list[Draw] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("|"):
            continue
        cells = [p.strip() for p in line.split("|")]
        if len(cells) < 23:
            continue
        issue = cells[1]
        if not issue.isdigit() or len(issue) < 6:
            continue
        date = cells[2]
        raw = cells[3:23]
        if len(raw) != 20 or not all(x.isdigit() for x in raw):
            continue
        nums = [int(x) for x in raw]
        if len(set(nums)) != 20:
            continue
        if any(n not in BALL_RANGE for n in nums):
            continue
        draws.append(Draw(issue=issue, date=date, numbers=frozenset(nums)))
    draws.sort(key=lambda d: d.issue)
    return draws


def window_draws(history: list[Draw], size: int) -> list[Draw]:
    if len(history) <= size:
        return history
    return history[-size:]


def omissions_map(history: list[Draw]) -> dict[int, int]:
    out: dict[int, int] = {}
    for b in BALL_RANGE:
        miss = 0
        for draw in reversed(history):
            if b in draw.numbers:
                break
            miss += 1
        else:
            miss = len(history) + 50
        out[b] = miss
    return out


def build_stats(history: list[Draw]) -> dict[str, object]:
    last_nums = history[-1].numbers
    prev_nums = history[-2].numbers if len(history) >= 2 else last_nums

    per_window: dict[int, Counter[int]] = {}
    for w in WINDOWS:
        wd = window_draws(history, w)
        c: Counter[int] = Counter()
        for draw in wd:
            c.update(draw.numbers)
        per_window[w] = c

    pair_hist = Counter()
    tail = window_draws(history, 100)
    for draw in tail:
        nums = sorted(draw.numbers)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_hist[(nums[i], nums[j])] += 1

    zone_of = lambda n: 0 if n <= 26 else (1 if n <= 53 else 2)
    zone_counts = Counter(zone_of(n) for draw in window_draws(history, 50) for n in draw.numbers)
    zone_total = sum(zone_counts.values()) or 1

    w50 = window_draws(history, 50)
    plank_counts = Counter(plank_id(n) for draw in w50 for n in draw.numbers)
    plank_total = sum(plank_counts.values()) or 1

    last3 = window_draws(history, 3)
    burst = Counter()
    for draw in last3:
        burst.update(draw.numbers)

    return {
        "per_window": per_window,
        "last_nums": last_nums,
        "prev_nums": prev_nums,
        "pair_hist": pair_hist,
        "zone_counts": zone_counts,
        "zone_total": zone_total,
        "plank_counts": plank_counts,
        "plank_total": plank_total,
        "burst": burst,
        "streak": streak_map(history),
        "omissions": omissions_map(history),
        "hist_len": len(history),
    }


def streak_map(history: list[Draw]) -> dict[int, int]:
    out: dict[int, int] = {}
    for b in BALL_RANGE:
        streak = 0
        for draw in reversed(history):
            if b in draw.numbers:
                streak += 1
            else:
                break
        out[b] = streak
    return out


def zone_balance_score(ball: int, stats: dict[str, object]) -> float:
    zc: Counter[int] = stats["zone_counts"]  # type: ignore[assignment]
    zt: int = stats["zone_total"]  # type: ignore[assignment]
    z = 0 if ball <= 26 else (1 if ball <= 53 else 2)
    target = zt / 3.0
    actual = zc.get(z, 0)
    if target <= 0:
        return 0.0
    ratio = actual / target
    return max(0.0, min(1.0, ratio / 2.0))


def ball_features(ball: int, stats: dict[str, object]) -> dict[str, float]:
    per_window: dict[int, Counter[int]] = stats["per_window"]  # type: ignore[assignment]
    last_nums: frozenset[int] = stats["last_nums"]  # type: ignore[assignment]
    pair_hist: Counter = stats["pair_hist"]  # type: ignore[assignment]
    omissions: dict[int, int] = stats["omissions"]  # type: ignore[assignment]

    feats: dict[str, float] = {}

    for w in WINDOWS:
        c = per_window[w]
        cnt = c.get(ball, 0)
        expected = w * DRAW_SIZE / len(BALL_RANGE)
        feats[f"hot{w}"] = min(cnt / max(expected, 1e-6), 2.5) / 2.5

    feats["repeat_last"] = 1.0 if ball in last_nums else 0.0

    n1 = sum(1 for x in last_nums if abs(ball - x) == 1)
    feats["neighbor1"] = min(n1 / 6.0, 1.0)
    n2 = sum(1 for x in last_nums if 2 <= abs(ball - x) <= 3)
    feats["neighbor2"] = min(n2 / 8.0, 1.0)

    pair_score = 0.0
    for x in last_nums:
        a, b = (min(ball, x), max(ball, x))
        pair_score += pair_hist.get((a, b), 0)
    ph_max = max(pair_hist.values(), default=1)
    feats["pair_last"] = min(pair_score / (20.0 * ph_max), 1.0)

    miss = omissions[ball]
    feats["omission_mid"] = math.exp(-((miss - 5.0) ** 2) / (2.0 * 6.0**2))
    feats["omission_deep"] = min(miss / 40.0, 1.0)

    feats["zone_balance"] = zone_balance_score(ball, stats)

    burst: Counter = stats["burst"]  # type: ignore[assignment]
    raw_b = burst.get(ball, 0)
    feats["burst3"] = min(raw_b / 6.5, 1.0)

    plank_counts: Counter[int] = stats["plank_counts"]  # type: ignore[assignment]
    plank_total: int = stats["plank_total"]  # type: ignore[assignment]
    pid = plank_id(ball)
    avg_pl = plank_total / 4.0
    act_pl = plank_counts.get(pid, 0)
    if avg_pl > 0:
        feats["plank_cold"] = max(0.0, min(1.0, 1.0 - act_pl / (2.0 * avg_pl)))
    else:
        feats["plank_cold"] = 0.5

    streak_dict: dict[int, int] = stats["streak"]  # type: ignore[assignment]
    feats["carry2"] = min(streak_dict[ball] / 3.0, 1.0)

    return feats


def score_ball(features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def rank_balls(history: list[Draw], state: StrategyState) -> list[tuple[int, float, dict[str, float]]]:
    stats = build_stats(history)
    ranked: list[tuple[int, float, dict[str, float]]] = []
    for b in BALL_RANGE:
        f = ball_features(b, stats)
        ranked.append((b, score_ball(f, state.weights), f))
    ranked.sort(key=lambda t: (-t[1], t[0]))
    return ranked


def pick_top_k(
    ranked: list[tuple[int, float, dict[str, float]]],
    mode: str,
    k: int,
    *,
    max_per_plank: int = 2,
) -> list[int]:
    """mode: top = 分数前 k 个；diverse = 每 20 码段至多 max_per_plank 个后再补齐。"""
    if mode == "top":
        return [b for b, _, _ in ranked[:k]]
    chosen: list[int] = []
    plank_cnt: Counter[int] = Counter()
    for ball, _, _ in ranked:
        if len(chosen) >= k:
            break
        pid = plank_id(ball)
        if plank_cnt[pid] >= max_per_plank:
            continue
        chosen.append(ball)
        plank_cnt[pid] += 1
    if len(chosen) < k:
        for ball, _, _ in ranked:
            if ball not in chosen:
                chosen.append(ball)
                if len(chosen) >= k:
                    break
    return chosen[:k]


def pick_four(
    ranked: list[tuple[int, float, dict[str, float]]],
    mode: str,
    *,
    max_per_plank: int = 2,
) -> list[int]:
    return pick_top_k(ranked, mode, PICK_COUNT, max_per_plank=max_per_plank)


def compound9_line_hits(pool: frozenset[int], omit: int, draw: frozenset[int]) -> int:
    return len((pool - {omit}) & draw)


def compound9_count_winning_lines(
    pool: frozenset[int], draw: frozenset[int], prize_counts: frozenset[int]
) -> tuple[int, int]:
    """返回 (有奖注数 0..10, 单注最大命中数 0..9)。"""
    win_lines = 0
    best = 0
    for x in pool:
        h = compound9_line_hits(pool, x, draw)
        best = max(best, h)
        if h in prize_counts:
            win_lines += 1
    return win_lines, best


def estimate_random_compound9_any_win(
    prize_counts: frozenset[int], trials: int = 80000
) -> float:
    """随机复式10码→10注，「至少有一注命中奖级集合」的频率（蒙特卡洛）。"""
    balls = list(BALL_RANGE)
    ok = 0
    rng = random.Random(42)
    for _ in range(trials):
        pool = frozenset(rng.sample(balls, COMPOUND9_POOL))
        drw = frozenset(rng.sample(balls, DRAW_SIZE))
        w, _ = compound9_count_winning_lines(pool, drw, prize_counts)
        if w > 0:
            ok += 1
    return ok / trials if trials else 0.0


def build_dantuo4_tickets(bank: tuple[int, int], drag: frozenset[int]) -> list[frozenset[int]]:
    if len(drag) != DANTUO_DRAG or len(set(drag)) != DANTUO_DRAG:
        raise ValueError("drag must be 5 distinct numbers")
    bset = frozenset(bank)
    if len(bset) != DANTUO_BANK or not bset.isdisjoint(drag):
        raise ValueError("bank and drag must be disjoint, two bankers")
    return [frozenset(bset | set(pair)) for pair in combinations(sorted(drag), 2)]


def dantuo4_line_stats(draw: frozenset[int], tickets: Iterable[frozenset[int]]) -> tuple[int, int]:
    """(命中≥2 的注数 0..10, 单注最大命中 0..4)。"""
    win_lines = 0
    best = 0
    for t in tickets:
        h = len(t & draw)
        best = max(best, h)
        if h >= 2:
            win_lines += 1
    return win_lines, best


def estimate_random_dantuo4_any_ge2(trials: int = 80000) -> float:
    """随机：在 80 码中均匀抽 7 个，再均匀抽其中 2 个作胆、余 5 个作拖；开奖 20 个随机。"""
    balls = list(BALL_RANGE)
    rng = random.Random(44)
    ok = 0
    for _ in range(trials):
        seven = rng.sample(balls, DANTUO_BANK + DANTUO_DRAG)
        bank_pair = rng.sample(seven, DANTUO_BANK)
        drag = frozenset(x for x in seven if x not in bank_pair)
        bank = (bank_pair[0], bank_pair[1])
        tix = build_dantuo4_tickets(bank, drag)
        drw = frozenset(rng.sample(balls, DRAW_SIZE))
        wl, _ = dantuo4_line_stats(drw, tix)
        if wl > 0:
            ok += 1
    return ok / trials if trials else 0.0


def mean_features(balls: Iterable[int], stats: dict[str, object]) -> dict[str, float]:
    balls = list(balls)
    if not balls:
        return {k: 0.0 for k in INITIAL_WEIGHTS}
    acc = {k: 0.0 for k in INITIAL_WEIGHTS}
    for b in balls:
        f = ball_features(b, stats)
        for k in acc:
            acc[k] += f.get(k, 0.0)
    n = float(len(balls))
    for k in acc:
        acc[k] /= n
    return acc


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
        expected = sum(feat.get(name, 0.0) for feat in selected) / len(selected)
        current = state.weights.get(name, 0.0)
        target = INITIAL_WEIGHTS[name]
        updated = current * state.decay + target * (1.0 - state.decay)
        updated += state.learning_rate * (actual_features.get(name, 0.0) - expected)
        state.weights[name] = max(state.min_weight, min(state.max_weight, updated))


def pick_hypergeometric_at_least_m(n_match: int) -> float:
    """P(hits >= n_match) when picking 4 from 80 and 20 are winning."""
    from math import comb

    def p_k(k: int) -> float:
        return comb(20, k) * comb(60, 4 - k) / comb(80, 4)

    return sum(p_k(k) for k in range(n_match, 5))


P_RANDOM_GE2 = pick_hypergeometric_at_least_m(2)
P_RANDOM_GE3 = pick_hypergeometric_at_least_m(3)
P_RANDOM_EQ4 = pick_hypergeometric_at_least_m(4)


def _metrics_from_ticks(
    hit_counter: Counter[int],
    rounds: int,
    *,
    pick_mode: str,
    p_random_ge2: float = P_RANDOM_GE2,
    p_random_ge3: float = P_RANDOM_GE3,
    p_random_eq4: float = P_RANDOM_EQ4,
) -> dict[str, object]:
    win_ge2 = sum(hit_counter[m] for m in range(2, 5))
    win_ge3 = sum(hit_counter[m] for m in range(3, 5))
    win_eq4 = hit_counter.get(4, 0)
    return {
        "rounds": rounds,
        "pick_mode": pick_mode,
        "hit_counter": hit_counter,
        "rate_ge2": win_ge2 / rounds if rounds else 0.0,
        "rate_ge3": win_ge3 / rounds if rounds else 0.0,
        "rate_eq4": win_eq4 / rounds if rounds else 0.0,
        "p_random_ge2": p_random_ge2,
        "p_random_ge3": p_random_ge3,
        "p_random_eq4": p_random_eq4,
    }


def _predict_round(
    history: list[Draw],
    actual: Draw,
    state: StrategyState,
    *,
    pick_mode: str,
    max_per_plank: int,
    do_update: bool,
) -> int:
    ranked = rank_balls(history, state)
    top4 = pick_four(ranked, pick_mode, max_per_plank=max_per_plank)
    actual_set = actual.numbers
    hits = len(frozenset(top4) & actual_set)

    if do_update and state.adaptive:
        stats = build_stats(history)
        top_pool = ranked[: state.update_rank]
        actual_vec = mean_features(list(actual_set), stats)
        pool_feats = [item[2] for item in top_pool]
        update_state(state, actual_vec, pool_feats)

    return hits


def _predict_round_pick1(
    history: list[Draw],
    actual: Draw,
    state: StrategyState,
    *,
    pick_mode: str,
    max_per_plank: int,
    do_update: bool,
) -> int:
    ranked = rank_balls(history, state)
    ball = pick_top_k(ranked, pick_mode, PICK1_COUNT, max_per_plank=max_per_plank)[0]
    hit = 1 if ball in actual.numbers else 0

    if do_update and state.adaptive:
        stats = build_stats(history)
        top_pool = ranked[: state.update_rank]
        actual_vec = mean_features(list(actual.numbers), stats)
        pool_feats = [item[2] for item in top_pool]
        update_state(state, actual_vec, pool_feats)

    return hit


def _predict_round_compound9(
    history: list[Draw],
    actual: Draw,
    state: StrategyState,
    *,
    pick_mode: str,
    max_per_plank: int,
    prize_counts: frozenset[int],
    do_update: bool,
) -> int:
    ranked = rank_balls(history, state)
    top10 = pick_top_k(ranked, pick_mode, COMPOUND9_POOL, max_per_plank=max_per_plank)
    pool = frozenset(top10)
    wlines, _ = compound9_count_winning_lines(pool, actual.numbers, prize_counts)

    if do_update and state.adaptive:
        stats = build_stats(history)
        top_pool = ranked[: state.update_rank]
        actual_vec = mean_features(list(actual.numbers), stats)
        pool_feats = [item[2] for item in top_pool]
        update_state(state, actual_vec, pool_feats)

    return wlines


def _predict_round_dantuo4(
    history: list[Draw],
    actual: Draw,
    state: StrategyState,
    *,
    pick_mode: str,
    max_per_plank: int,
    do_update: bool,
) -> tuple[int, int]:
    ranked = rank_balls(history, state)
    seven = pick_top_k(
        ranked, pick_mode, DANTUO_BANK + DANTUO_DRAG, max_per_plank=max_per_plank
    )
    bank = (seven[0], seven[1])
    drag = frozenset(seven[2 : DANTUO_BANK + DANTUO_DRAG])
    tix = build_dantuo4_tickets(bank, drag)
    win_lines, best = dantuo4_line_stats(actual.numbers, tix)

    if do_update and state.adaptive:
        stats = build_stats(history)
        top_pool = ranked[: state.update_rank]
        actual_vec = mean_features(list(actual.numbers), stats)
        pool_feats = [item[2] for item in top_pool]
        update_state(state, actual_vec, pool_feats)

    return win_lines, best


def rolling_backtest_dantuo4(
    draws: list[Draw],
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    adaptive: bool = True,
    learning_rate: float = 0.015,
    pick_mode: str = "top",
    max_per_plank: int = 2,
    holdout_last: int = 0,
    random_trials: int = 80000,
) -> tuple[dict[str, object], StrategyState]:
    if len(draws) <= min_history:
        raise ValueError(f"Need more than {min_history} draws, got {len(draws)}.")
    if holdout_last < 0:
        raise ValueError("holdout_last must be >= 0")
    if holdout_last and len(draws) <= min_history + holdout_last:
        raise ValueError(
            f"Need len(draws) > min_history + holdout_last, got {len(draws)} vs "
            f"{min_history + holdout_last}"
        )

    tail_start = len(draws) - holdout_last if holdout_last else len(draws)
    p_random_any = estimate_random_dantuo4_any_ge2(trials=random_trials)

    state = StrategyState(
        adaptive=adaptive,
        weights=dict(INITIAL_WEIGHTS),
        learning_rate=learning_rate if adaptive else 0.0,
        decay=0.999,
    )

    tw_hist: Counter[int] = Counter()
    mx_hist: Counter[int] = Counter()
    for idx in range(min_history, tail_start):
        history = draws[:idx]
        actual = draws[idx]
        wl, best = _predict_round_dantuo4(
            history,
            actual,
            state,
            pick_mode=pick_mode,
            max_per_plank=max_per_plank,
            do_update=adaptive,
        )
        tw_hist[wl] += 1
        mx_hist[best] += 1

    train_rounds = tail_start - min_history
    any_w = sum(tw_hist[i] for i in range(1, DANTUO_LINES + 1))
    mean_lines = (
        sum(i * tw_hist[i] for i in range(DANTUO_LINES + 1)) / train_rounds
        if train_rounds
        else 0.0
    )

    metrics: dict[str, object] = {
        "game": "dantuo4",
        "rounds": train_rounds,
        "min_history": min_history,
        "adaptive": adaptive,
        "pick_mode": pick_mode,
        "tw_hist": tw_hist,
        "max_hit_hist": mx_hist,
        "rate_any_line": any_w / train_rounds if train_rounds else 0.0,
        "mean_winning_lines": mean_lines,
        "p_random_any_line": p_random_any,
        "holdout_last": holdout_last,
        "phase": "train_then_holdout" if holdout_last else "rolling_full",
        "holdout": None,
        "random_trials": random_trials,
    }

    if holdout_last > 0:
        h_tw: Counter[int] = Counter()
        h_mx: Counter[int] = Counter()
        for idx in range(tail_start, len(draws)):
            history = draws[:idx]
            actual = draws[idx]
            wl, best = _predict_round_dantuo4(
                history,
                actual,
                state,
                pick_mode=pick_mode,
                max_per_plank=max_per_plank,
                do_update=False,
            )
            h_tw[wl] += 1
            h_mx[best] += 1
        h_any = sum(h_tw[i] for i in range(1, DANTUO_LINES + 1))
        h_mean = (
            sum(i * h_tw[i] for i in range(DANTUO_LINES + 1)) / holdout_last
            if holdout_last
            else 0.0
        )
        metrics["holdout"] = {
            "rounds": holdout_last,
            "tw_hist": h_tw,
            "max_hit_hist": h_mx,
            "rate_any_line": h_any / holdout_last if holdout_last else 0.0,
            "mean_winning_lines": h_mean,
            "p_random_any_line": p_random_any,
            "first_issue": draws[tail_start].issue,
            "first_date": draws[tail_start].date,
            "last_issue": draws[-1].issue,
            "last_date": draws[-1].date,
            "pick_mode": pick_mode,
        }

    return metrics, state


def rolling_backtest_compound9(
    draws: list[Draw],
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    adaptive: bool = True,
    learning_rate: float = 0.015,
    pick_mode: str = "top",
    max_per_plank: int = 3,
    holdout_last: int = 0,
    prize_counts: frozenset[int] | None = None,
    random_trials: int = 80000,
) -> tuple[dict[str, object], StrategyState]:
    pc = prize_counts if prize_counts is not None else PRIZE_COUNTS_9_WITH_ZERO
    if len(draws) <= min_history:
        raise ValueError(f"Need more than {min_history} draws, got {len(draws)}.")
    if holdout_last < 0:
        raise ValueError("holdout_last must be >= 0")
    if holdout_last and len(draws) <= min_history + holdout_last:
        raise ValueError(
            f"Need len(draws) > min_history + holdout_last, got {len(draws)} vs "
            f"{min_history + holdout_last}"
        )

    tail_start = len(draws) - holdout_last if holdout_last else len(draws)
    p_random_any = estimate_random_compound9_any_win(pc, trials=random_trials)

    state = StrategyState(
        adaptive=adaptive,
        weights=dict(INITIAL_WEIGHTS),
        learning_rate=learning_rate if adaptive else 0.0,
        decay=0.999,
    )

    tw_hist: Counter[int] = Counter()
    for idx in range(min_history, tail_start):
        history = draws[:idx]
        actual = draws[idx]
        wlines = _predict_round_compound9(
            history,
            actual,
            state,
            pick_mode=pick_mode,
            max_per_plank=max_per_plank,
            prize_counts=pc,
            do_update=adaptive,
        )
        tw_hist[wlines] += 1

    train_rounds = tail_start - min_history
    any_w = sum(tw_hist[i] for i in range(1, COMPOUND9_LINES + 1))
    mean_lines = (
        sum(i * tw_hist[i] for i in range(COMPOUND9_LINES + 1)) / train_rounds
        if train_rounds
        else 0.0
    )

    metrics: dict[str, object] = {
        "game": "compound9",
        "rounds": train_rounds,
        "min_history": min_history,
        "adaptive": adaptive,
        "pick_mode": pick_mode,
        "prize_counts_label": "含中零" if 0 in pc else "不含中零",
        "tw_hist": tw_hist,
        "rate_any_line": any_w / train_rounds if train_rounds else 0.0,
        "mean_winning_lines": mean_lines,
        "p_random_any_line": p_random_any,
        "holdout_last": holdout_last,
        "phase": "train_then_holdout" if holdout_last else "rolling_full",
        "holdout": None,
        "random_trials": random_trials,
    }

    if holdout_last > 0:
        h_hist: Counter[int] = Counter()
        for idx in range(tail_start, len(draws)):
            history = draws[:idx]
            actual = draws[idx]
            wlines = _predict_round_compound9(
                history,
                actual,
                state,
                pick_mode=pick_mode,
                max_per_plank=max_per_plank,
                prize_counts=pc,
                do_update=False,
            )
            h_hist[wlines] += 1
        h_any = sum(h_hist[i] for i in range(1, COMPOUND9_LINES + 1))
        h_mean = (
            sum(i * h_hist[i] for i in range(COMPOUND9_LINES + 1)) / holdout_last
            if holdout_last
            else 0.0
        )
        metrics["holdout"] = {
            "rounds": holdout_last,
            "tw_hist": h_hist,
            "rate_any_line": h_any / holdout_last if holdout_last else 0.0,
            "mean_winning_lines": h_mean,
            "p_random_any_line": p_random_any,
            "first_issue": draws[tail_start].issue,
            "first_date": draws[tail_start].date,
            "last_issue": draws[-1].issue,
            "last_date": draws[-1].date,
            "pick_mode": pick_mode,
            "prize_counts_label": metrics["prize_counts_label"],
        }

    return metrics, state


def rolling_backtest_pick1(
    draws: list[Draw],
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    adaptive: bool = True,
    learning_rate: float = 0.015,
    pick_mode: str = "top",
    max_per_plank: int = 2,
    holdout_last: int = 0,
) -> tuple[dict[str, object], StrategyState]:
    if len(draws) <= min_history:
        raise ValueError(f"Need more than {min_history} draws, got {len(draws)}.")
    if holdout_last < 0:
        raise ValueError("holdout_last must be >= 0")
    if holdout_last and len(draws) <= min_history + holdout_last:
        raise ValueError(
            f"Need len(draws) > min_history + holdout_last, got {len(draws)} vs "
            f"{min_history + holdout_last}"
        )

    tail_start = len(draws) - holdout_last if holdout_last else len(draws)

    state = StrategyState(
        adaptive=adaptive,
        weights=dict(INITIAL_WEIGHTS),
        learning_rate=learning_rate if adaptive else 0.0,
        decay=0.999,
    )

    hit_counter: Counter[int] = Counter()
    for idx in range(min_history, tail_start):
        history = draws[:idx]
        actual = draws[idx]
        h = _predict_round_pick1(
            history,
            actual,
            state,
            pick_mode=pick_mode,
            max_per_plank=max_per_plank,
            do_update=adaptive,
        )
        hit_counter[h] += 1

    train_rounds = tail_start - min_history
    hits = hit_counter.get(1, 0)
    metrics: dict[str, object] = {
        "game": "pick1",
        "rounds": train_rounds,
        "min_history": min_history,
        "adaptive": adaptive,
        "pick_mode": pick_mode,
        "hit_counter": hit_counter,
        "rate_hit": hits / train_rounds if train_rounds else 0.0,
        "p_random_hit": P_RANDOM_PICK1,
        "holdout_last": holdout_last,
        "phase": "train_then_holdout" if holdout_last else "rolling_full",
        "holdout": None,
    }

    if holdout_last > 0:
        hh = Counter()
        for idx in range(tail_start, len(draws)):
            history = draws[:idx]
            actual = draws[idx]
            h = _predict_round_pick1(
                history,
                actual,
                state,
                pick_mode=pick_mode,
                max_per_plank=max_per_plank,
                do_update=False,
            )
            hh[h] += 1
        h1 = hh.get(1, 0)
        metrics["holdout"] = {
            "rounds": holdout_last,
            "hit_counter": hh,
            "rate_hit": h1 / holdout_last if holdout_last else 0.0,
            "p_random_hit": P_RANDOM_PICK1,
            "first_issue": draws[tail_start].issue,
            "first_date": draws[tail_start].date,
            "last_issue": draws[-1].issue,
            "last_date": draws[-1].date,
            "pick_mode": pick_mode,
        }

    return metrics, state


def rolling_backtest(
    draws: list[Draw],
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    adaptive: bool = True,
    learning_rate: float = 0.015,
    pick_mode: str = "top",
    max_per_plank: int = 2,
    holdout_last: int = 0,
) -> tuple[dict[str, object], StrategyState]:
    """滚动回测。holdout_last>0 时：仅在前 len−holdout 期做自适应更新；末尾 holdout_last 期固定权重评测。"""
    if len(draws) <= min_history:
        raise ValueError(f"Need more than {min_history} draws, got {len(draws)}.")
    if holdout_last < 0:
        raise ValueError("holdout_last must be >= 0")
    if holdout_last and len(draws) <= min_history + holdout_last:
        raise ValueError(
            f"Need len(draws) > min_history + holdout_last, got {len(draws)} vs "
            f"{min_history + holdout_last}"
        )

    tail_start = len(draws) - holdout_last if holdout_last else len(draws)

    state = StrategyState(
        adaptive=adaptive,
        weights=dict(INITIAL_WEIGHTS),
        learning_rate=learning_rate if adaptive else 0.0,
        decay=0.999,
    )

    hit_counter: Counter[int] = Counter()
    for idx in range(min_history, tail_start):
        history = draws[:idx]
        actual = draws[idx]
        h = _predict_round(
            history,
            actual,
            state,
            pick_mode=pick_mode,
            max_per_plank=max_per_plank,
            do_update=adaptive,
        )
        hit_counter[h] += 1

    train_rounds = tail_start - min_history
    metrics: dict[str, object] = {
        **_metrics_from_ticks(hit_counter, train_rounds, pick_mode=pick_mode),
        "game": "pick4",
        "min_history": min_history,
        "adaptive": adaptive,
        "holdout_last": holdout_last,
        "phase": "train_then_holdout" if holdout_last else "rolling_full",
        "holdout": None,
    }

    if holdout_last > 0:
        hold_hits = Counter()
        for idx in range(tail_start, len(draws)):
            history = draws[:idx]
            actual = draws[idx]
            h = _predict_round(
                history,
                actual,
                state,
                pick_mode=pick_mode,
                max_per_plank=max_per_plank,
                do_update=False,
            )
            hold_hits[h] += 1
        metrics["holdout"] = _metrics_from_ticks(
            hold_hits, holdout_last, pick_mode=pick_mode
        )
        ho = metrics["holdout"]  # type: ignore[assignment]
        ho["first_issue"] = draws[tail_start].issue  # type: ignore[index]
        ho["first_date"] = draws[tail_start].date  # type: ignore[index]
        ho["last_issue"] = draws[-1].issue  # type: ignore[index]
        ho["last_date"] = draws[-1].date  # type: ignore[index]

    return metrics, state


def pct(x: float) -> str:
    return f"{x * 100:.3f}%"


def report_pick1(metrics: dict[str, object], state: StrategyState) -> str:
    rounds: int = int(metrics["rounds"])  # type: ignore[arg-type]
    hit_counter: Counter[int] = metrics["hit_counter"]  # type: ignore[assignment]
    adaptive: bool = bool(metrics["adaptive"])  # type: ignore[arg-type]
    mode = "逐期自适应" if adaptive else "固定权重"
    hold_k = int(metrics.get("holdout_last") or 0)
    pick_md = str(metrics.get("pick_mode", "top"))
    h0 = int(hit_counter.get(0, 0))
    h1 = int(hit_counter.get(1, 0))
    rate = float(metrics["rate_hit"])  # type: ignore[arg-type]
    pr = float(metrics["p_random_hit"])  # type: ignore[arg-type]

    lines: list[str] = []
    lines.append(
        f"快乐8「选一」单注滚动回测（{mode}；选号: {pick_md}；每期取分最高的 1 个号，"
        f"落在开奖20个号中即命中）"
    )
    if hold_k > 0:
        lines.append(
            f"训练段：{rounds} 期（样本外之前；冷启动后{'逐期更新权重' if adaptive else '不更新'}）"
        )
        lines.extend(["", "---", "## 第一段：训练段（样本内）", ""])
    else:
        lines.append(
            f"回测期数: {rounds}（冷启动 {metrics['min_history']} 期后；每期预测后{'更新权重' if adaptive else '不更新权重'}）"
        )
        lines.append("")

    lines.append("命中分布（每期 1 注）：")
    lines.append(f"  未中: {h0} 期  ({pct(h0 / rounds)})")
    lines.append(f"  命中: {h1} 期  ({pct(h1 / rounds)})")
    lines.append("")
    lines.append(f"命中率: {pct(rate)}  （随机单注选一理论: {pct(pr)}）")
    lines.append(f"相对理论倍数: {(rate / pr) if pr else 0.0:.3f}x")
    lines.append("")

    hold = metrics.get("holdout")
    if hold_k > 0 and isinstance(hold, dict):
        hr = int(hold["rounds"])  # type: ignore[arg-type]
        hc: Counter[int] = hold["hit_counter"]  # type: ignore[assignment]
        lines.extend(["---", "## 第二段：末尾样本外（权重冻结）", ""])
        lines.append(f"样本外期数: {hr}，期号: {hold.get('first_issue')}～{hold.get('last_issue')}")
        lines.append("")
        z0 = int(hc.get(0, 0))
        z1 = int(hc.get(1, 0))
        lines.append("命中分布：")
        lines.append(f"  未中: {z0} 期  ({pct(z0 / hr)})")
        lines.append(f"  命中: {z1} 期  ({pct(z1 / hr)})")
        ra = float(hold["rate_hit"])  # type: ignore[arg-type]
        lines.append("")
        lines.append(f"命中率: {pct(ra)}  （理论: {pct(pr)}）  倍数 {(ra / pr) if pr else 0:.3f}x")
        lines.append("")

    if hold_k > 0:
        lines.append("当前权重（冻结；样本外用训练段末尾）：")
    else:
        lines.append("当前信号权重（初始 → 现在）：")
    for name in sorted(INITIAL_WEIGHTS):
        lines.append(f"  {name}: {INITIAL_WEIGHTS[name]:.3f} → {state.weights[name]:.3f}")

    lines.extend(
        [
            "",
            "---",
            "说明",
            "",
            "- 选一单注：无信息条件下每期命中概率为 20/80=25%；历史回测偏离仅反映有限样本与规则排序，不代表可迁移预测优势。",
        ]
    )
    return "\n".join(lines)


def report(metrics: dict[str, object], state: StrategyState) -> str:
    rounds: int = metrics["rounds"]  # type: ignore[assignment]
    hit_counter: Counter[int] = metrics["hit_counter"]  # type: ignore[assignment]
    adaptive: bool = metrics["adaptive"]  # type: ignore[assignment]

    lines: list[str] = []
    mode = "逐期自适应" if adaptive else "固定权重"
    hold_k = int(metrics.get("holdout_last") or 0)
    if hold_k > 0:
        lines.append(
            f"快乐8「选四」两段式评测（训练段样本内滚动 + 末尾 {hold_k} 期冻结权重／{mode}／选号: {metrics.get('pick_mode', 'top')}）"
        )
        lines.append(
            f"训练段：{rounds} 期（仅用期号早于样本外区间的历史预测，冷启动后逐期{'更新权重' if adaptive else '不更新权重'}）"
        )
        lines.extend(["", "---", "## 第一段：训练段（样本内）", ""])
    else:
        lines.append(f"快乐8「选四」滚动回测（{mode}，选号: {metrics.get('pick_mode', 'top')}）")
        lines.append(f"回测期数: {rounds}（冷启动 {metrics['min_history']} 期后；每期预测后照常{'更新权重' if adaptive else '不更新权重'}）")
        lines.append("")
    hint = (
        "四区分散：每 20 码区间至多 2 个"
        if metrics.get("pick_mode") == "diverse"
        else "每期取模型排名前 4 个号"
    )
    lines.append(f"命中个数分布（与开奖 20 个求交；{hint}）")
    for k in range(5):
        c = hit_counter.get(k, 0)
        lines.append(f"  命中 {k} 个: {c} 期  ({pct(c / rounds)})")
    lines.append("")
    lines.append("选四「有奖」口径（通常中 2 及以上）:")
    lines.append(f"  中 ≥2: {pct(metrics['rate_ge2'])}  （完全随机选 4 个的理论: {pct(metrics['p_random_ge2'])}）")
    lines.append(f"  中 ≥3: {pct(metrics['rate_ge3'])}  （理论: {pct(metrics['p_random_ge3'])}）")
    lines.append(f"  中  4: {pct(metrics['rate_eq4'])}  （理论: {pct(metrics['p_random_eq4'])}）")
    rel = (metrics["rate_ge2"] / metrics["p_random_ge2"]) if metrics["p_random_ge2"] else 0.0
    lines.append(f"  ≥2 相对理论倍数: {rel:.3f}x")
    lines.append("")

    hold = metrics.get("holdout")
    if hold_k > 0 and isinstance(hold, dict):
        h_rounds = int(hold["rounds"])  # type: ignore[arg-type]
        h_counter = hold["hit_counter"]  # type: ignore[arg-type]
        lines.extend(["---", "## 第二段：末尾样本外（权重冻结，不开奖后纠错）", ""])
        fi = hold.get("first_issue", "?")
        fd = hold.get("first_date", "?")
        li = hold.get("last_issue", "?")
        ld = hold.get("last_date", "?")
        lines.append(f"期号范围（含端点）: {fi}～{li} ，日期参考: {fd}～{ld}")
        lines.append(f"样本外期数: {h_rounds}")
        lines.append("")
        lines.append(f"命中个数分布（与开奖 20 个求交；{hint}）")
        for k in range(5):
            c = int(h_counter.get(k, 0))
            lines.append(f"  命中 {k} 个: {c} 期  ({pct(c / h_rounds)})")
        lines.append("")
        lines.append("选四「有奖」口径：")
        r2 = float(hold["rate_ge2"])  # type: ignore[arg-type]
        r3 = float(hold["rate_ge3"])  # type: ignore[arg-type]
        r4 = float(hold["rate_eq4"])  # type: ignore[arg-type]
        pr2 = float(hold["p_random_ge2"])  # type: ignore[arg-type]
        pr3 = float(hold["p_random_ge3"])  # type: ignore[arg-type]
        pr4 = float(hold["p_random_eq4"])  # type: ignore[arg-type]
        lines.append(f"  中 ≥2: {pct(r2)}  （理论: {pct(pr2)}）")
        lines.append(f"  中 ≥3: {pct(r3)}  （理论: {pct(pr3)}）")
        lines.append(f"  中  4: {pct(r4)}  （理论: {pct(pr4)}）")
        hrel = (r2 / pr2) if pr2 else 0.0
        lines.append(f"  ≥2 相对理论倍数: {hrel:.3f}x")
        lines.append("")

    lines.append("")
    if hold_k > 0:
        lines.append("当前权重（样本外沿用训练段末尾，不再更新）：")
    else:
        lines.append("当前信号权重（初始 → 现在）：")
    for name in sorted(INITIAL_WEIGHTS):
        lines.append(f"  {name}: {INITIAL_WEIGHTS[name]:.3f} → {state.weights[name]:.3f}")

    lines.extend(
        [
            "",
            "---",
            "关于「单注命中率 50%」",
            "",
            f"- 「每期仅 1 注选四、计≥2 个」在无信息条件下长期约 {pct(P_RANDOM_GE2)}；独立公平开奖前提下，无法用历史走势把「单注」稳定抬到约 50%。",
            "- 复式、买多注会提高「多张里至少有一张≥2」的机会，但这是另一口径，不是单注长期 50%。",
        ]
    )
    return "\n".join(lines)


def _append_compound9_tw_hist(lines: list[str], tw: Counter[int], rounds: int, title: str) -> None:
    lines.append(title)
    for k in range(COMPOUND9_LINES + 1):
        c = int(tw.get(k, 0))
        lines.append(f"  本期 {k} 注中奖: {c} 期  ({pct(c / rounds)})")


def report_dantuo4(metrics: dict[str, object], state: StrategyState) -> str:
    rounds: int = int(metrics["rounds"])  # type: ignore[arg-type]
    tw_hist: Counter[int] = metrics["tw_hist"]  # type: ignore[assignment]
    mx_hist: Counter[int] = metrics["max_hit_hist"]  # type: ignore[assignment]
    adaptive: bool = bool(metrics["adaptive"])  # type: ignore[arg-type]
    mode = "逐期自适应" if adaptive else "固定权重"
    hold_k = int(metrics.get("holdout_last") or 0)
    trials = int(metrics.get("random_trials", 80000))
    pick_md = str(metrics.get("pick_mode", "top"))

    lines: list[str] = []
    lines.append(
        f"快乐8「选四胆拖·2胆+5拖→{DANTUO_LINES}注」（{mode}；选号: {pick_md}；"
        f"每期取分最高的 2 个作胆、接下来 5 个作拖）"
    )
    hint = (
        "四区分散时每 20 码段上限由 --max-per-plank 控制"
        if pick_md == "diverse"
        else "胆拖号码完全由当期模型分排序决定（开奖后更新权重）"
    )

    if hold_k > 0:
        lines.append(
            f"训练段：{rounds} 期（样本外之前；冷启动后{'逐期更新权重' if adaptive else '不更新'}）"
        )
        lines.extend(["", "---", "## 第一段：训练段（样本内）", "", f"说明：{hint}", ""])
    else:
        lines.append(
            f"回测期数: {rounds}（冷启动 {metrics['min_history']} 期后；{hint}）。"
            f"随机胆拖基线蒙特卡洛 trials={trials}。"
        )
        lines.append("")

    rate_any = float(metrics["rate_any_line"])  # type: ignore[arg-type]
    mean_wl = float(metrics["mean_winning_lines"])  # type: ignore[arg-type]
    pr = float(metrics["p_random_any_line"])  # type: ignore[arg-type]

    _append_compound9_tw_hist(lines, tw_hist, rounds, "每期「10注里：有几注」命中≥2（训练段）：")
    lines.append("")
    lines.append("单注最大命中（10注里最好的一注，相对开奖20个）：")
    for k in range(5):
        c = int(mx_hist.get(k, 0))
        lines.append(f"  最大命中 {k} 个: {c} 期  ({pct(c / rounds)})")
    lines.append("")
    lines.append('复式核心指标「至少一注命中≥2」')
    lines.append(f"  训练段比例: {pct(rate_any)}  （蒙特卡洛随机胆拖: {pct(pr)}）")
    lines.append(f"  倍数: {(rate_any / pr) if pr else 0.0:.3f}x")
    lines.append(f"  每期平均「命中≥2 的注数」: {mean_wl:.3f} / {DANTUO_LINES}")
    lines.append("")

    hold = metrics.get("holdout")
    if hold_k > 0 and isinstance(hold, dict):
        hr = int(hold["rounds"])  # type: ignore[arg-type]
        hh: Counter[int] = hold["tw_hist"]  # type: ignore[assignment]
        hm: Counter[int] = hold["max_hit_hist"]  # type: ignore[assignment]
        lines.extend(["---", "## 第二段：末尾样本外（权重冻结）", ""])
        lines.append(f"样本外期数: {hr}，期号: {hold.get('first_issue')}～{hold.get('last_issue')}")
        lines.append("")
        _append_compound9_tw_hist(lines, hh, hr, "每期「≥2 注数」分布（冻结段）：")
        lines.append("")
        lines.append("单注最大命中（冻结段）：")
        for k in range(5):
            c = int(hm.get(k, 0))
            lines.append(f"  最大命中 {k} 个: {c} 期  ({pct(c / hr)})")
        ra = float(hold["rate_any_line"])  # type: ignore[arg-type]
        prh = float(hold["p_random_any_line"])  # type: ignore[arg-type]
        mw = float(hold["mean_winning_lines"])  # type: ignore[arg-type]
        lines.extend(
            [
                "",
                '至少一注≥2',
                f"  比例: {pct(ra)}  （随机胆拖: {pct(prh)}）  倍数 {(ra / prh) if prh else 0:.3f}x",
                f"  平均≥2注数/期: {mw:.3f}",
                "",
            ]
        )

    if hold_k > 0:
        lines.append("当前权重（冻结；样本外用训练段末尾）：")
    else:
        lines.append("当前信号权重（初始 → 现在）：")
    for name in sorted(INITIAL_WEIGHTS):
        lines.append(f"  {name}: {INITIAL_WEIGHTS[name]:.3f} → {state.weights[name]:.3f}")

    lines.extend(
        [
            "",
            "---",
            "规则备忘",
            "",
            f"- 与实体店一致：2胆+5拖 → C(5,2)={DANTUO_LINES} 注选四。",
            "- 「有奖」口径本报告按 **单注命中开奖20个中的个数 ≥2** 计注；与官方奖级表不完全等价时以票面为准。",
            f"- 随机基线：7码均匀、其中胆为均匀抽2个、余5拖；trials={trials}。",
        ]
    )
    return "\n".join(lines)


def report_compound9(metrics: dict[str, object], state: StrategyState) -> str:
    rounds: int = int(metrics["rounds"])  # type: ignore[arg-type]
    tw_hist: Counter[int] = metrics["tw_hist"]  # type: ignore[assignment]
    adaptive: bool = bool(metrics["adaptive"])  # type: ignore[arg-type]
    mode = "逐期自适应" if adaptive else "固定权重"
    hold_k = int(metrics.get("holdout_last") or 0)
    plab = str(metrics.get("prize_counts_label", ""))
    trials = int(metrics.get("random_trials", 80000))
    pick_md = str(metrics.get("pick_mode", "top"))

    lines: list[str] = []
    lines.append(
        f"快乐8「复式选九·10码→{COMPOUND9_LINES}注」（{mode}；奖级口径：{plab}；选号策略: {pick_md}）"
    )
    hint = (
        f"四区分散时每 20 码段上限由 --max-per-plank 控制"
        if pick_md == "diverse"
        else f"每期按当前权重取分最高的 {COMPOUND9_POOL} 个号码组成复式"
    )

    if hold_k > 0:
        lines.append(
            f"训练段：{rounds} 期（仅用样本外之前的历史；冷启动后{'逐期更新权重' if adaptive else '不更新权重'}）"
        )
        lines.extend(["", "---", "## 第一段：训练段（样本内）", "", f"说明：{hint}", ""])
    else:
        lines.append(
            f"回测期数: {rounds}（冷启动 {metrics['min_history']} 期后；{hint}）。随机基线蒙特卡洛 trials={trials}。"
        )
        lines.append("")

    rate_any = float(metrics["rate_any_line"])  # type: ignore[arg-type]
    mean_wl = float(metrics["mean_winning_lines"])  # type: ignore[arg-type]
    pr = float(metrics["p_random_any_line"])  # type: ignore[arg-type]

    _append_compound9_tw_hist(lines, tw_hist, rounds, "每期「10注里：有几注」落在奖级（训练段）：")
    lines.append("")
    lines.append('复式核心指标「至少一中奖注」')
    lines.append(f"  训练段比例: {pct(rate_any)}  （蒙特卡洛随机复式10码: {pct(pr)}）")
    lines.append(f"  倍数: {(rate_any / pr) if pr else 0.0:.3f}x")
    lines.append(f"  每期平均「中奖注数」: {mean_wl:.3f} / {COMPOUND9_LINES}")
    lines.append("")

    hold = metrics.get("holdout")
    if hold_k > 0 and isinstance(hold, dict):
        hr = int(hold["rounds"])  # type: ignore[arg-type]
        hh: Counter[int] = hold["tw_hist"]  # type: ignore[assignment]
        lines.extend(["---", "## 第二段：末尾样本外（权重冻结）", ""])
        lines.append(f"样本外期数: {hr}，期号: {hold.get('first_issue')}～{hold.get('last_issue')}")
        lines.append("")
        _append_compound9_tw_hist(lines, hh, hr, "每期中奖注数分布（冻结段）：")
        ra = float(hold["rate_any_line"])  # type: ignore[arg-type]
        prh = float(hold["p_random_any_line"])  # type: ignore[arg-type]
        mw = float(hold["mean_winning_lines"])  # type: ignore[arg-type]
        lines.extend(
            [
                "",
                '至少一中奖注',
                f"  比例: {pct(ra)}  （随机复式: {pct(prh)}）  倍数 {(ra / prh) if prh else 0:.3f}x",
                f"  平均中奖注数/期: {mw:.3f}",
                "",
            ]
        )

    if hold_k > 0:
        lines.append("当前权重（冻结；样本外用训练段末尾权重）：")
    else:
        lines.append("当前信号权重（初始 → 现在）：")
    for name in sorted(INITIAL_WEIGHTS):
        lines.append(f"  {name}: {INITIAL_WEIGHTS[name]:.3f} → {state.weights[name]:.3f}")

    lines.extend(
        [
            "",
            "---",
            "规则备忘",
            "",
            f"- 复式选九「10个号」= C(10,9)= {COMPOUND9_LINES} 注；本报告奖级是否与当期票面一致请以福彩规则为准（默认可含「九中零」）。",
            f"- 「至少一注有奖」随机基线为蒙特卡洛 {trials} 次，非闭式精确解；不同 trials 可有微小抖动。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except OSError:
            pass

    parser = argparse.ArgumentParser(
        description="快乐8滚动回测：选一、选四、复式选九、选四胆拖，支持逐期自适应权重"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "kl8.md",
        help="kl8 数据 markdown 路径",
    )
    parser.add_argument(
        "--game",
        choices=("pick1", "pick4", "compound9", "dantuo4"),
        default="pick4",
        help="pick1=选一1码；pick4=选四4码；compound9=复式选九10码；dantuo4=选四胆拖2胆5拖→10注",
    )
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument("--no-adaptive", action="store_true", help="关闭逐期学习，仅固定初始权重")
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument(
        "--pick",
        choices=("top", "diverse"),
        default="top",
        help="号码选择：top=分最高前 N 个；diverse=四区 20 码段分散（选一 N=1；选四 N=4；选九 N=10；胆拖 N=7）",
    )
    parser.add_argument(
        "--max-per-plank",
        type=int,
        default=2,
        help="diverse 模式下每个 20 码区间最多选几个（默认 2；仅 --pick diverse 生效）",
    )
    parser.add_argument(
        "--compound-no-zero",
        action="store_true",
        help="复式选九奖级不包含「九中零」档（仅用命中4～9）；默认含中零以贴近常见规则",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=80000,
        metavar="N",
        help="复式选九 / 胆拖随机基线的蒙特卡洛次数（compound9、dantuo4）",
    )
    parser.add_argument(
        "--holdout-last",
        type=int,
        default=0,
        metavar="K",
        help="末尾留出 K 期做样本外：仅用此前数据逐期更新，最后 K 期权重冻结评测（0 表示全开样本内滚动）",
    )
    args = parser.parse_args()

    draws = parse_draws(args.data)
    prize_9 = PRIZE_COUNTS_9_NO_ZERO if args.compound_no_zero else PRIZE_COUNTS_9_WITH_ZERO

    if args.game == "pick1":
        metrics, state = rolling_backtest_pick1(
            draws,
            min_history=args.min_history,
            adaptive=not args.no_adaptive,
            learning_rate=args.learning_rate,
            pick_mode=args.pick,
            max_per_plank=args.max_per_plank,
            holdout_last=args.holdout_last,
        )
        print(report_pick1(metrics, state))
    elif args.game == "compound9":
        metrics, state = rolling_backtest_compound9(
            draws,
            min_history=args.min_history,
            adaptive=not args.no_adaptive,
            learning_rate=args.learning_rate,
            pick_mode=args.pick,
            max_per_plank=args.max_per_plank,
            holdout_last=args.holdout_last,
            prize_counts=prize_9,
            random_trials=args.random_trials,
        )
        print(report_compound9(metrics, state))
    elif args.game == "dantuo4":
        metrics, state = rolling_backtest_dantuo4(
            draws,
            min_history=args.min_history,
            adaptive=not args.no_adaptive,
            learning_rate=args.learning_rate,
            pick_mode=args.pick,
            max_per_plank=args.max_per_plank,
            holdout_last=args.holdout_last,
            random_trials=args.random_trials,
        )
        print(report_dantuo4(metrics, state))
    elif args.game == "pick4":
        metrics, state = rolling_backtest(
            draws,
            min_history=args.min_history,
            adaptive=not args.no_adaptive,
            learning_rate=args.learning_rate,
            pick_mode=args.pick,
            max_per_plank=args.max_per_plank,
            holdout_last=args.holdout_last,
        )
        print(report(metrics, state))
    else:
        raise RuntimeError(f"Unhandled game: {args.game}")


if __name__ == "__main__":
    main()
