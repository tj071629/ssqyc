from __future__ import annotations

import argparse
import itertools
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


RED_RANGE = range(1, 34)
BLUE_RANGE = range(1, 17)
WINDOW_SIZE = 50
DEFAULT_TICKET_COUNT = 5

# Full-history stability search defaults (3394 rolling windows).
FULL_HISTORY_RED_NORMAL = {"hot10": 0.0, "hot20": 0.01, "repeat_last": 0.0, "neighbor_last": 0.0, "omission_mid": 0.04}
FULL_HISTORY_RED_EXTREME = {"hot20": 0.04, "omission_mid": 0.08, "zone_cold": 0.04, "hot50": 0.04}

# Last-300-draw stability fine-tune (--recent-draws 300 --grid-search-stability).
RECENT_300_RED_NORMAL = {"hot10": 0.0, "hot20": 0.01, "repeat_last": 0.0, "neighbor_last": 0.0, "omission_mid": 0.03}
RECENT_300_RED_EXTREME = {"hot20": 0.04, "omission_mid": 0.08, "zone_cold": 0.04, "hot50": 0.02}


@dataclass(frozen=True)
class Draw:
    issue: str
    date: str
    red: tuple[int, ...]
    blue: int


@dataclass(frozen=True)
class Ticket:
    name: str
    red: tuple[int, ...]
    blue: int
    note: str


@dataclass
class StrategyState:
    adaptive: bool
    red_weights: dict[str, float] = field(default_factory=dict)
    red_weights_extreme: dict[str, float] = field(default_factory=dict)
    blue_weights: dict[str, float] = field(default_factory=dict)
    template_scores: dict[str, float] = field(default_factory=dict)
    red_lr: float = 0.0
    blue_lr: float = 0.0
    template_lr: float = 0.0
    red_decay: float = 1.0
    blue_decay: float = 1.0
    template_decay: float = 1.0


RED_FEATURES = (
    "hot10",
    "hot20",
    "hot50",
    "repeat_last",
    "neighbor_last",
    "neighbor_prev",
    "omission_step",
    "omission_mid",
    "omission_deep",
    "zone_cold",
)

BLUE_FEATURES = (
    "hot5",
    "hot10",
    "repeat_last",
    "neighbor_last",
    "neighbor_prev",
    "omission_step",
    "omission_mid",
    "omission_deep",
)

TEMPLATE_NAMES = ("主线票", "重号票", "邻号票", "冷补票", "跳点票", "均衡票")


def format_numbers(numbers: Iterable[int]) -> str:
    return " ".join(f"{number:02d}" for number in numbers)


def slice_recent_draws(draws: list[Draw], recent: int | None) -> list[Draw]:
    """Use only the last `recent` draws (chronological tail) for short-window tuning."""
    if recent is None or recent <= 0:
        return draws
    if recent >= len(draws):
        return draws
    return draws[-recent:]


def parse_history(path: Path) -> list[Draw]:
    draws: list[Draw] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 4 or parts[0] in {"期号", "-------"}:
            continue
        issue, date, red_raw, blue_raw = parts
        try:
            red = tuple(sorted(int(value) for value in red_raw.split()))
            blue = int(blue_raw)
        except ValueError:
            continue
        if len(red) != 6:
            continue
        draws.append(Draw(issue=issue, date=date, red=red, blue=blue))
    return draws


def build_state(
    adaptive: bool,
    overrides: dict[str, float] | None = None,
    extreme_overrides: dict[str, float] | None = None,
    *,
    preset: str = "full",
) -> StrategyState:
    # Weak-adaptation defaults for SSQ:
    # keep online correction conservative to avoid degrading baseline hit rates.
    # Current default strategy:
    # freeze red learning, only allow blue + template micro-adjustments.
    if preset == "recent300":
        best_red_defaults = {name: RECENT_300_RED_NORMAL.get(name, 0.0) for name in RED_FEATURES}
        extreme_defaults = {name: RECENT_300_RED_EXTREME.get(name, best_red_defaults.get(name, 0.0)) for name in RED_FEATURES}
    else:
        best_red_defaults = {name: FULL_HISTORY_RED_NORMAL.get(name, 0.0) for name in RED_FEATURES}
        extreme_defaults = {name: FULL_HISTORY_RED_EXTREME.get(name, best_red_defaults.get(name, 0.0)) for name in RED_FEATURES}
    state = StrategyState(
        adaptive=adaptive,
        red_weights={name: best_red_defaults.get(name, 0.0) for name in RED_FEATURES},
        red_weights_extreme={name: extreme_defaults.get(name, best_red_defaults.get(name, 0.0)) for name in RED_FEATURES},
        blue_weights={name: 0.0 for name in BLUE_FEATURES},
        template_scores={name: 1.0 for name in TEMPLATE_NAMES},
        red_lr=0.0,
        blue_lr=0.008 if adaptive else 0.0,
        template_lr=0.012 if adaptive else 0.0,
        red_decay=1.0,
        blue_decay=0.9985 if adaptive else 1.0,
        template_decay=0.999 if adaptive else 1.0,
    )
    if overrides:
        for name, value in overrides.items():
            if name in state.red_weights:
                state.red_weights[name] = value
    if extreme_overrides:
        for name, value in extreme_overrides.items():
            if name in state.red_weights_extreme:
                state.red_weights_extreme[name] = value
    return state


def omission(draws: list[Draw], attr: str, number: int) -> int:
    for offset, draw in enumerate(reversed(draws), start=1):
        values = draw.red if attr == "red" else (draw.blue,)
        if number in values:
            return offset - 1
    return len(draws)


def red_zone_index(number: int) -> int:
    if 1 <= number <= 11:
        return 0
    if 12 <= number <= 22:
        return 1
    return 2


def red_zone_signature(red: tuple[int, ...]) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    for number in red:
        counts[red_zone_index(number)] += 1
    return tuple(counts)


def is_extreme_window(window: list[Draw]) -> bool:
    recent_6 = window[-6:]
    signatures = [red_zone_signature(draw.red) for draw in recent_6]
    last_sig = signatures[-1]
    extreme_recent = sum(1 for sig in signatures if any(count == 0 for count in sig) or max(sig) >= 4)
    return any(count == 0 for count in last_sig) or max(last_sig) >= 4 or extreme_recent >= 2


def weighted_bonus(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights[name] * features[name] for name in features)


def build_red_context(window: list[Draw]) -> dict[str, object]:
    recent_10 = window[-10:]
    recent_20 = window[-20:]
    last = window[-1]
    prev = window[-2]
    recent_6 = window[-6:]
    return {
        "window": window,
        "last": last,
        "prev": prev,
        "recent_6": recent_6,
        "counter_10": Counter(number for draw in recent_10 for number in draw.red),
        "counter_20": Counter(number for draw in recent_20 for number in draw.red),
        "counter_50": Counter(number for draw in window for number in draw.red),
    }


def build_blue_context(window: list[Draw]) -> dict[str, object]:
    recent_5 = window[-5:]
    recent_10 = window[-10:]
    return {
        "window": window,
        "last": window[-1],
        "prev": window[-2],
        "counter_5": Counter(draw.blue for draw in recent_5),
        "counter_10": Counter(draw.blue for draw in recent_10),
    }


def red_features(number: int, context: dict[str, object]) -> dict[str, float]:
    window = context["window"]
    last = context["last"]
    prev = context["prev"]
    recent_6 = context["recent_6"]
    counter_10 = context["counter_10"]
    counter_20 = context["counter_20"]
    counter_50 = context["counter_50"]
    miss = omission(window, "red", number)
    zone_hits = sum(number in draw.red for draw in recent_6)
    return {
        "hot10": counter_10[number] / 4.0,
        "hot20": counter_20[number] / 8.0,
        "hot50": counter_50[number] / 14.0,
        "repeat_last": float(number in last.red),
        "neighbor_last": float(any(abs(number - value) == 1 for value in last.red)),
        "neighbor_prev": float(any(abs(number - value) == 1 for value in prev.red)),
        "omission_step": min(miss, 14) / 14.0,
        "omission_mid": float(4 <= miss <= 9),
        "omission_deep": float(miss >= 16),
        "zone_cold": float(zone_hits <= 1),
    }


def blue_features(number: int, context: dict[str, object]) -> dict[str, float]:
    window = context["window"]
    last = context["last"]
    prev = context["prev"]
    counter_5 = context["counter_5"]
    counter_10 = context["counter_10"]
    miss = omission(window, "blue", number)
    return {
        "hot5": counter_5[number] / 2.5,
        "hot10": counter_10[number] / 5.0,
        "repeat_last": float(number == last.blue),
        "neighbor_last": float(abs(number - last.blue) == 1),
        "neighbor_prev": float(abs(number - prev.blue) == 1),
        "omission_step": min(miss, 10) / 10.0,
        "omission_mid": float(3 <= miss <= 7),
        "omission_deep": float(miss >= 10),
    }


def red_base_score(number: int, context: dict[str, object]) -> float:
    window = context["window"]
    last = context["last"]
    prev = context["prev"]
    counter_10 = context["counter_10"]
    counter_20 = context["counter_20"]
    counter_50 = context["counter_50"]
    miss = omission(window, "red", number)
    score = 0.0
    score += counter_10[number] * 2.8
    score += counter_20[number] * 1.4
    score += counter_50[number] * 0.28
    if number in last.red:
        score += 3.3
    if any(abs(number - value) == 1 for value in last.red):
        score += 3.1
    if any(abs(number - value) == 1 for value in prev.red):
        score += 1.2
    score += min(miss, 14) * 0.24
    if 4 <= miss <= 9:
        score += 1.1
    if miss >= 16:
        score -= 0.25
    return score


def blue_base_score(number: int, context: dict[str, object]) -> float:
    window = context["window"]
    last = context["last"]
    prev = context["prev"]
    counter_5 = context["counter_5"]
    counter_10 = context["counter_10"]
    miss = omission(window, "blue", number)
    score = 0.0
    score += counter_5[number] * 2.5
    score += counter_10[number] * 1.2
    if number == last.blue:
        score += 2.0
    if abs(number - last.blue) == 1:
        score += 2.4
    if abs(number - prev.blue) == 1:
        score += 0.9
    score += min(miss, 10) * 0.22
    if 3 <= miss <= 7:
        score += 0.7
    if miss >= 10:
        score -= 0.2
    return score


def score_red(window: list[Draw], state: StrategyState) -> tuple[list[int], dict[int, dict[str, float]]]:
    context = build_red_context(window)
    red_weights = state.red_weights_extreme if is_extreme_window(window) else state.red_weights
    scores: dict[int, float] = {}
    feature_table: dict[int, dict[str, float]] = {}
    for number in RED_RANGE:
        features = red_features(number, context)
        feature_table[number] = features
        scores[number] = red_base_score(number, context) + weighted_bonus(red_weights, features)
    ranked = sorted(scores, key=lambda n: (-scores[n], n))
    return ranked, feature_table


def score_blue(window: list[Draw], state: StrategyState) -> tuple[list[int], dict[int, dict[str, float]]]:
    context = build_blue_context(window)
    scores: dict[int, float] = {}
    feature_table: dict[int, dict[str, float]] = {}
    for number in BLUE_RANGE:
        features = blue_features(number, context)
        feature_table[number] = features
        scores[number] = blue_base_score(number, context) + weighted_bonus(state.blue_weights, features)
    ranked = sorted(scores, key=lambda n: (-scores[n], n))
    return ranked, feature_table


def choose_red(
    ranked: list[int],
    *,
    include: Iterable[int] = (),
    exclude: Iterable[int] = (),
    reverse_jump: bool = False,
    reverse_from: int = 10,
) -> tuple[int, ...]:
    selected: list[int] = []
    excluded = set(exclude)
    rank_map = {number: idx for idx, number in enumerate(ranked)}
    zone_counts = [0, 0, 0]
    zone_cap = [3, 3, 3]

    for number in include:
        if number in excluded or number in selected:
            continue
        if len(selected) >= 6:
            break
        selected.append(number)
        zone_counts[red_zone_index(number)] += 1

    for number in ranked:
        if number in excluded or number in selected:
            continue
        if reverse_jump and rank_map[number] < reverse_from:
            continue
        zone = red_zone_index(number)
        if zone_counts[zone] >= zone_cap[zone]:
            continue
        selected.append(number)
        zone_counts[zone] += 1
        if len(selected) == 6:
            break

    if len(selected) < 6:
        for number in ranked:
            if number in excluded or number in selected:
                continue
            selected.append(number)
            if len(selected) == 6:
                break
    return tuple(sorted(selected[:6]))


def choose_blue(ranked: list[int], *, include: Iterable[int] = (), reverse_jump: bool = False, reverse_from: int = 5) -> int:
    include_list = [number for number in include if number in BLUE_RANGE]
    if include_list:
        return include_list[0]
    rank_map = {number: idx for idx, number in enumerate(ranked)}
    for number in ranked:
        if reverse_jump and rank_map[number] < reverse_from:
            continue
        return number
    return ranked[0]


def distinct_red(candidate: tuple[int, ...], existing: list[Ticket], ranked: list[int]) -> tuple[int, ...]:
    existing_reds = {ticket.red for ticket in existing}
    if candidate not in existing_reds:
        return candidate
    base = list(candidate)
    for number in ranked:
        if number in base:
            continue
        alt = sorted(base[:-1] + [number])
        alt_tuple = tuple(alt)
        if alt_tuple not in existing_reds:
            return alt_tuple
    return candidate


def build_red_priority_portfolio(red_ranked: list[int], blue_ranked: list[int], window: list[Draw]) -> list[Ticket]:
    """Build 5 tickets with red-priority overlap control.

    Portfolio: 2 mainline + 2 mirror + 1 detached.
    """
    last = window[-1]
    prev = window[-2]
    core = red_ranked[:4]
    hot = [number for number in red_ranked[:12] if number not in core]
    neighbors_last = sorted({n for value in last.red for n in (value - 1, value + 1) if n in RED_RANGE})
    neighbors_prev = sorted({n for value in prev.red for n in (value - 1, value + 1) if n in RED_RANGE})
    blue_neighbors = [n for n in (last.blue - 1, last.blue + 1, prev.blue - 1, prev.blue + 1) if n in BLUE_RANGE]

    t1_red = choose_red(red_ranked, include=core + hot[:2])
    t2_red = choose_red(red_ranked, include=core[:3] + neighbors_last[:2] + hot[:1])
    t3_red = choose_red(red_ranked, include=core[:2] + neighbors_last[:2] + hot[:2])
    t4_red = choose_red(red_ranked, include=core[:2] + neighbors_prev[:2] + hot[2:4])
    t5_red = choose_red(red_ranked, include=hot[4:6] + neighbors_prev[:2], exclude=core[:2], reverse_jump=True, reverse_from=11)

    tickets = [
        Ticket("主线票", t1_red, choose_blue(blue_ranked, include=[last.blue]), "红球核心集中"),
        Ticket("重号票", t2_red, choose_blue(blue_ranked, include=blue_neighbors[:1]), "主线镜像覆盖"),
        Ticket("邻号票", t3_red, choose_blue(blue_ranked), "红球3核镜像"),
        Ticket("均衡票", t4_red, choose_blue(blue_ranked, include=blue_neighbors[1:2]), "红球2核扩散"),
        Ticket("跳点票", t5_red, choose_blue(blue_ranked, reverse_jump=True, reverse_from=6), "红球脱核防同质"),
    ]
    return tickets


def generate_tickets(window: list[Draw], state: StrategyState, ticket_count: int) -> tuple[list[Ticket], list[int], list[int], dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    red_ranked, red_features_table = score_red(window, state)
    blue_ranked, blue_features_table = score_blue(window, state)
    last = window[-1]
    prev = window[-2]

    neighbors_last = sorted({n for value in last.red for n in (value - 1, value + 1) if n in RED_RANGE})
    hot_red = [number for number in red_ranked[:12] if number not in last.red]
    cold_jump = [number for number in red_ranked[10:24] if number not in last.red]
    blue_neighbors = [n for n in (last.blue - 1, last.blue + 1, prev.blue - 1, prev.blue + 1) if n in BLUE_RANGE]

    template_map = {
        "主线票": Ticket("主线票", choose_red(red_ranked, include=list(last.red[:2]) + neighbors_last[:2]), choose_blue(blue_ranked, include=[last.blue]), "重号+邻号主线"),
        "重号票": Ticket("重号票", choose_red(red_ranked, include=list(last.red[:3])), choose_blue(blue_ranked, include=blue_neighbors[:1]), "重号集中"),
        "邻号票": Ticket("邻号票", choose_red(red_ranked, include=neighbors_last[:4]), choose_blue(blue_ranked, include=blue_neighbors[:1]), "邻号承接"),
        "冷补票": Ticket("冷补票", choose_red(red_ranked, include=cold_jump[:2]), choose_blue(blue_ranked, reverse_jump=True, reverse_from=6), "冷补修正"),
        "跳点票": Ticket("跳点票", choose_red(red_ranked, include=hot_red[:2], reverse_jump=True, reverse_from=10), choose_blue(blue_ranked, reverse_jump=True, reverse_from=5), "逆向跳点"),
        "均衡票": Ticket("均衡票", choose_red(red_ranked, include=hot_red[:3]), choose_blue(blue_ranked), "均衡兜底"),
    }

    # Red-priority fixed 5-ticket portfolio:
    # enforce "2 mainline + 2 mirror + 1 detached" to reduce red-line dilution.
    tickets: list[Ticket] = []
    if ticket_count == 5:
        for t in build_red_priority_portfolio(red_ranked, blue_ranked, window):
            red = distinct_red(t.red, tickets, red_ranked)
            tickets.append(Ticket(t.name, red, t.blue, t.note))
    else:
        ordered_templates = sorted(TEMPLATE_NAMES, key=lambda name: (-state.template_scores[name], TEMPLATE_NAMES.index(name)))
        chosen = ordered_templates[:ticket_count]
        for name in chosen:
            t = template_map[name]
            red = distinct_red(t.red, tickets, red_ranked)
            tickets.append(Ticket(t.name, red, t.blue, t.note))
    return tickets, red_ranked, blue_ranked, red_features_table, blue_features_table


def hit_summary(ticket: Ticket, actual: Draw) -> dict[str, int]:
    red_hits = len(set(ticket.red) & set(actual.red))
    blue_hit = int(ticket.blue == actual.blue)
    return {"red_hits": red_hits, "blue_hits": blue_hit}


def update_weights(
    weights: dict[str, float],
    feature_table: dict[int, dict[str, float]],
    actual_numbers: tuple[int, ...],
    pool_numbers: Iterable[int],
    *,
    learning_rate: float,
    decay: float,
    limit: float,
) -> None:
    for name in list(weights):
        pool_mean = statistics.mean(feature_table[number][name] for number in pool_numbers)
        actual_mean = statistics.mean(feature_table[number][name] for number in actual_numbers)
        weights[name] = max(-limit, min(limit, weights[name] * decay + learning_rate * (actual_mean - pool_mean)))


def update_blue_weights(
    weights: dict[str, float],
    feature_table: dict[int, dict[str, float]],
    actual_number: int,
    pool_numbers: Iterable[int],
    *,
    learning_rate: float,
    decay: float,
    limit: float,
) -> None:
    for name in list(weights):
        pool_mean = statistics.mean(feature_table[number][name] for number in pool_numbers)
        actual_value = feature_table[actual_number][name]
        weights[name] = max(-limit, min(limit, weights[name] * decay + learning_rate * (actual_value - pool_mean)))


def update_template_scores(state: StrategyState, tickets: list[Ticket], hits: list[dict[str, int]]) -> None:
    if not state.adaptive:
        return
    for name in list(state.template_scores):
        state.template_scores[name] = max(0.35, min(3.5, 1.0 + (state.template_scores[name] - 1.0) * state.template_decay))
    for ticket, hit in zip(tickets, hits):
        reward = hit["red_hits"] + hit["blue_hits"] * 1.6
        delta = state.template_lr * (reward - 1.25)
        state.template_scores[ticket.name] = max(0.35, min(3.5, state.template_scores[ticket.name] + delta))


def top_weight_items(weights: dict[str, float], count: int = 4) -> list[tuple[str, float]]:
    return sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:count]


def rolling_backtest(
    draws: list[Draw],
    window_size: int,
    ticket_count: int,
    *,
    adaptive: bool,
    red_weight_overrides: dict[str, float] | None = None,
    red_weight_extreme_overrides: dict[str, float] | None = None,
    preset: str = "full",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    state = build_state(
        adaptive=adaptive,
        overrides=red_weight_overrides,
        extreme_overrides=red_weight_extreme_overrides,
        preset=preset,
    )
    aggregate = defaultdict(int)
    aggregate_lists: dict[str, list[float]] = defaultdict(list)
    segment_aggregate: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    results: list[dict[str, object]] = []

    for start in range(0, len(draws) - window_size):
        window = draws[start : start + window_size]
        actual = draws[start + window_size]
        tickets, red_ranked, blue_ranked, red_table, blue_table = generate_tickets(window, state, ticket_count)
        aggregate["extreme_windows"] += int(is_extreme_window(window))
        hits = [hit_summary(ticket, actual) for ticket in tickets]
        best = max(hits, key=lambda h: (h["red_hits"], h["blue_hits"]))
        any_same_ticket_4p1 = any(h["red_hits"] >= 4 and h["blue_hits"] == 1 for h in hits)

        aggregate["windows"] += 1
        aggregate["best_red_3plus"] += int(best["red_hits"] >= 3)
        aggregate["best_red_4plus"] += int(best["red_hits"] >= 4)
        aggregate["best_blue_1plus"] += int(best["blue_hits"] >= 1)
        aggregate["same_ticket_red4_blue1"] += int(any_same_ticket_4p1)
        aggregate_lists["best_red_hits"].append(best["red_hits"])
        aggregate_lists["best_blue_hits"].append(best["blue_hits"])
        segment = actual.issue[:4]
        segment_aggregate[segment]["windows"] += 1
        segment_aggregate[segment]["best_red_3plus"] += int(best["red_hits"] >= 3)
        segment_aggregate[segment]["best_red_4plus"] += int(best["red_hits"] >= 4)
        segment_aggregate[segment]["best_blue_1plus"] += int(best["blue_hits"] >= 1)

        results.append(
            {
                "train_start": window[0].issue,
                "train_end": window[-1].issue,
                "actual_issue": actual.issue,
                "actual_date": actual.date,
                "actual_red": actual.red,
                "actual_blue": actual.blue,
                "tickets": tickets,
                "hits": hits,
                "best_red_hits": best["red_hits"],
                "best_blue_hits": best["blue_hits"],
            }
        )

        if adaptive:
            update_weights(
                state.red_weights,
                red_table,
                actual.red,
                RED_RANGE,
                learning_rate=state.red_lr,
                decay=state.red_decay,
                limit=1.8,
            )
            update_blue_weights(
                state.blue_weights,
                blue_table,
                actual.blue,
                BLUE_RANGE,
                learning_rate=state.blue_lr,
                decay=state.blue_decay,
                limit=1.8,
            )
            update_template_scores(state, tickets, hits)

    summary = {
        "adaptive": adaptive,
        "windows": aggregate["windows"],
        "best_red_3plus_rate": aggregate["best_red_3plus"] / aggregate["windows"],
        "best_red_4plus_rate": aggregate["best_red_4plus"] / aggregate["windows"],
        "best_blue_1plus_rate": aggregate["best_blue_1plus"] / aggregate["windows"],
        "same_ticket_red4_blue1_rate": aggregate["same_ticket_red4_blue1"] / aggregate["windows"],
        "extreme_window_rate": aggregate["extreme_windows"] / aggregate["windows"],
        "avg_best_red_hits": statistics.mean(aggregate_lists["best_red_hits"]),
        "avg_best_blue_hits": statistics.mean(aggregate_lists["best_blue_hits"]),
        "top_red_weights": top_weight_items(state.red_weights),
        "top_blue_weights": top_weight_items(state.blue_weights),
        "top_templates": sorted(state.template_scores.items(), key=lambda item: (-item[1], item[0]))[:5],
        "segment_breakdown": {
            key: {
                "windows": values["windows"],
                "best_red_3plus_rate": values["best_red_3plus"] / values["windows"],
                "best_red_4plus_rate": values["best_red_4plus"] / values["windows"],
                "best_blue_1plus_rate": values["best_blue_1plus"] / values["windows"],
            }
            for key, values in sorted(segment_aggregate.items())
        },
    }
    return results, summary


def compare_summaries(fixed: dict[str, object], adaptive: dict[str, object]) -> dict[str, float]:
    return {
        "best_red_3plus_delta": adaptive["best_red_3plus_rate"] - fixed["best_red_3plus_rate"],
        "best_red_4plus_delta": adaptive["best_red_4plus_rate"] - fixed["best_red_4plus_rate"],
        "best_blue_1plus_delta": adaptive["best_blue_1plus_rate"] - fixed["best_blue_1plus_rate"],
        "same_ticket_red4_blue1_delta": adaptive["same_ticket_red4_blue1_rate"] - fixed["same_ticket_red4_blue1_rate"],
        "avg_best_red_hits_delta": adaptive["avg_best_red_hits"] - fixed["avg_best_red_hits"],
    }


def red_priority_score(summary: dict[str, object]) -> float:
    # Prioritize red-hit stability: red3+ first, red4+ second.
    return summary["best_red_3plus_rate"] * 100.0 + summary["best_red_4plus_rate"] * 25.0


def red_stability_score(summary: dict[str, object]) -> float:
    base = red_priority_score(summary)
    segments = summary.get("segment_breakdown", {})
    if not segments:
        return base
    red3_rates = [item["best_red_3plus_rate"] for item in segments.values()]
    min_red3 = min(red3_rates)
    std_red3 = statistics.pstdev(red3_rates) if len(red3_rates) > 1 else 0.0
    return base + min_red3 * 30.0 - std_red3 * 35.0


def run_grid_search(draws: list[Draw], window_size: int, ticket_count: int) -> tuple[dict[str, float], dict[str, object], list[dict[str, object]]]:
    grid = {
        "hot10": [0.00, 0.03, 0.06, 0.09],
        "hot20": [0.00, 0.02, 0.04],
        "repeat_last": [0.00, 0.02, 0.05],
        "neighbor_last": [0.00, 0.02, 0.05],
        "omission_mid": [0.00, 0.02, 0.04],
    }
    names = list(grid.keys())
    candidates: list[dict[str, object]] = []
    best_overrides: dict[str, float] | None = None
    best_summary: dict[str, object] | None = None
    best_score = float("-inf")

    for values in itertools.product(*(grid[name] for name in names)):
        overrides = dict(zip(names, values))
        _, summary = rolling_backtest(
            draws,
            window_size,
            ticket_count,
            adaptive=False,
            red_weight_overrides=overrides,
        )
        score = red_priority_score(summary)
        row = {"overrides": overrides, "summary": summary, "score": score}
        candidates.append(row)
        if score > best_score:
            best_score = score
            best_overrides = overrides
            best_summary = summary

    ranked = sorted(candidates, key=lambda item: (-item["score"], -item["summary"]["best_red_3plus_rate"], -item["summary"]["best_red_4plus_rate"]))[:8]
    assert best_overrides is not None and best_summary is not None
    return best_overrides, best_summary, ranked


def run_stability_grid_search(
    draws: list[Draw], window_size: int, ticket_count: int
) -> tuple[dict[str, float], dict[str, float], dict[str, object], list[dict[str, object]]]:
    normal_grid = {
        "hot20": [0.01, 0.02, 0.03],
        "omission_mid": [0.03, 0.04, 0.05],
    }
    extreme_grid = {
        "hot20": [0.04, 0.06, 0.08],
        "omission_mid": [0.04, 0.06, 0.08],
        "zone_cold": [0.02, 0.04],
        "hot50": [0.02, 0.03, 0.04],
    }

    normal_names = list(normal_grid.keys())
    extreme_names = list(extreme_grid.keys())
    candidates: list[dict[str, object]] = []
    best_normal: dict[str, float] | None = None
    best_extreme: dict[str, float] | None = None
    best_summary: dict[str, object] | None = None
    best_score = float("-inf")

    for n_values in itertools.product(*(normal_grid[name] for name in normal_names)):
        normal_overrides = dict(zip(normal_names, n_values))
        for e_values in itertools.product(*(extreme_grid[name] for name in extreme_names)):
            extreme_overrides = dict(zip(extreme_names, e_values))
            _, summary = rolling_backtest(
                draws,
                window_size,
                ticket_count,
                adaptive=False,
                red_weight_overrides=normal_overrides,
                red_weight_extreme_overrides=extreme_overrides,
            )
            score = red_stability_score(summary)
            row = {
                "normal_overrides": normal_overrides,
                "extreme_overrides": extreme_overrides,
                "summary": summary,
                "score": score,
            }
            candidates.append(row)
            if score > best_score:
                best_score = score
                best_normal = normal_overrides
                best_extreme = extreme_overrides
                best_summary = summary

    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["score"],
            -item["summary"]["best_red_3plus_rate"],
            -item["summary"]["best_red_4plus_rate"],
            -item["summary"]["best_blue_1plus_rate"],
        ),
    )[:8]
    assert best_normal is not None and best_extreme is not None and best_summary is not None
    return best_normal, best_extreme, best_summary, ranked


def run_dual_state_grid_search(
    draws: list[Draw], window_size: int, ticket_count: int
) -> tuple[dict[str, float], dict[str, float], dict[str, object], list[dict[str, object]]]:
    normal_grid = {
        "hot20": [0.02, 0.04],
        "omission_mid": [0.02, 0.04],
    }
    extreme_grid = {
        "hot20": [0.04, 0.06],
        "omission_mid": [0.04, 0.06],
        "zone_cold": [0.02, 0.04],
        "hot50": [0.00, 0.03],
    }

    normal_names = list(normal_grid.keys())
    extreme_names = list(extreme_grid.keys())
    candidates: list[dict[str, object]] = []
    best_normal: dict[str, float] | None = None
    best_extreme: dict[str, float] | None = None
    best_summary: dict[str, object] | None = None
    best_score = float("-inf")

    for n_values in itertools.product(*(normal_grid[name] for name in normal_names)):
        normal_overrides = dict(zip(normal_names, n_values))
        for e_values in itertools.product(*(extreme_grid[name] for name in extreme_names)):
            extreme_overrides = dict(zip(extreme_names, e_values))
            _, summary = rolling_backtest(
                draws,
                window_size,
                ticket_count,
                adaptive=False,
                red_weight_overrides=normal_overrides,
                red_weight_extreme_overrides=extreme_overrides,
            )
            score = red_priority_score(summary)
            row = {
                "normal_overrides": normal_overrides,
                "extreme_overrides": extreme_overrides,
                "summary": summary,
                "score": score,
            }
            candidates.append(row)
            if score > best_score:
                best_score = score
                best_normal = normal_overrides
                best_extreme = extreme_overrides
                best_summary = summary

    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["score"],
            -item["summary"]["best_red_3plus_rate"],
            -item["summary"]["best_red_4plus_rate"],
        ),
    )[:8]
    assert best_normal is not None and best_extreme is not None and best_summary is not None
    return best_normal, best_extreme, best_summary, ranked


def build_report(
    fixed_results: list[dict[str, object]],
    fixed_summary: dict[str, object],
    adaptive_results: list[dict[str, object]],
    adaptive_summary: dict[str, object],
    comparison: dict[str, float],
    window_size: int,
    ticket_count: int,
) -> str:
    lines: list[str] = []
    lines.append("# 双色球50期逐期自适应回测复盘报告")
    lines.append("")
    lines.append("## 1. 回测设定")
    lines.append(f"- 固定滚动窗口：`{window_size}` 期。")
    lines.append(f"- 每期出票：`{ticket_count}` 注单式。")
    lines.append("- 执行顺序：`1~50 -> 51`，`2~51 -> 52`，依次滚动至历史末尾。")
    lines.append("- 每期开奖后立刻复盘，并在线更新红球权重、蓝球权重与票型优先级。")
    lines.append("")
    lines.append("## 2. 固定规则 vs 逐期自适应")
    lines.append("| 指标 | 固定规则 | 逐期自适应 | 差值 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 5注至少1注红球3中 | `{fixed_summary['best_red_3plus_rate']:.2%}` | `{adaptive_summary['best_red_3plus_rate']:.2%}` | `{comparison['best_red_3plus_delta']:+.2%}` |")
    lines.append(f"| 5注至少1注红球4中 | `{fixed_summary['best_red_4plus_rate']:.2%}` | `{adaptive_summary['best_red_4plus_rate']:.2%}` | `{comparison['best_red_4plus_delta']:+.2%}` |")
    lines.append(f"| 5注至少1注蓝球命中 | `{fixed_summary['best_blue_1plus_rate']:.2%}` | `{adaptive_summary['best_blue_1plus_rate']:.2%}` | `{comparison['best_blue_1plus_delta']:+.2%}` |")
    lines.append(f"| 同一注红4+蓝1 | `{fixed_summary['same_ticket_red4_blue1_rate']:.2%}` | `{adaptive_summary['same_ticket_red4_blue1_rate']:.2%}` | `{comparison['same_ticket_red4_blue1_delta']:+.2%}` |")
    lines.append(f"| 最优单注平均红球命中 | `{fixed_summary['avg_best_red_hits']:.2f}` | `{adaptive_summary['avg_best_red_hits']:.2f}` | `{comparison['avg_best_red_hits_delta']:+.2f}` |")
    lines.append("")
    lines.append("## 3. 最终学到的偏好")
    lines.append(f"- 红球增强特征：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['top_red_weights'])}`")
    lines.append(f"- 蓝球增强特征：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['top_blue_weights'])}`")
    lines.append(f"- 高分票型：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['top_templates'])}`")
    lines.append("")
    lines.append("## 4. 最近12期样本")
    lines.append("| 训练区间 | 预测期 | 实际开奖 | 最优命中 |")
    lines.append("| --- | --- | --- | --- |")
    for row in adaptive_results[-12:]:
        actual = f"{format_numbers(row['actual_red'])} + {row['actual_blue']:02d}"
        best = f"{row['best_red_hits']}红+{row['best_blue_hits']}蓝"
        lines.append(f"| {row['train_start']}-{row['train_end']} | {row['actual_issue']} | `{actual}` | `{best}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="双色球50期逐期自适应回测")
    parser.add_argument("--history", default="docs/双色球历史开奖号码.md", help="历史开奖 Markdown 文件路径")
    parser.add_argument("--report", default="docs/双色球50期逐期自适应回测复盘报告.md", help="报告输出路径")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="滚动窗口大小")
    parser.add_argument("--ticket-count", type=int, default=DEFAULT_TICKET_COUNT, help="每期输出注数")
    parser.add_argument("--grid-search", action="store_true", help="执行红球优先网格搜索")
    parser.add_argument("--grid-search-dual", action="store_true", help="执行双状态红球网格搜索")
    parser.add_argument("--grid-search-stability", action="store_true", help="执行分段稳定性优先网格搜索")
    parser.add_argument(
        "--recent-draws",
        type=int,
        default=None,
        metavar="N",
        help="仅使用最近 N 期数据（截尾），用于近端样本稳定性微调；与 --grid-search-stability 等联用",
    )
    parser.add_argument(
        "--preset",
        choices=("full", "recent300"),
        default="full",
        help="红球权重预设：full=全历史稳定性最优；recent300=最近300期稳定性微调最优",
    )
    args = parser.parse_args()

    draws = parse_history(Path(args.history))
    draws = slice_recent_draws(draws, args.recent_draws)
    if len(draws) <= args.window_size:
        raise SystemExit("历史数据不足，无法完成滚动回测。")

    if args.grid_search:
        best_overrides, best_summary, ranked = run_grid_search(draws, args.window_size, args.ticket_count)
        print("红球优先网格搜索完成。")
        print(f"最优参数：{best_overrides}")
        print(f"最优 红3+ 命中率：{best_summary['best_red_3plus_rate']:.2%}")
        print(f"最优 红4+ 命中率：{best_summary['best_red_4plus_rate']:.2%}")
        print(f"最优 蓝1 命中率：{best_summary['best_blue_1plus_rate']:.2%}")
        print("Top 5 候选：")
        for idx, row in enumerate(ranked[:5], start=1):
            s = row["summary"]
            print(
                f"{idx}. score={row['score']:.3f} "
                f"red3+={s['best_red_3plus_rate']:.2%} "
                f"red4+={s['best_red_4plus_rate']:.2%} "
                f"blue1+={s['best_blue_1plus_rate']:.2%} "
                f"params={row['overrides']}"
            )
        return

    if args.grid_search_dual:
        best_normal, best_extreme, best_summary, ranked = run_dual_state_grid_search(draws, args.window_size, args.ticket_count)
        print("双状态红球网格搜索完成。")
        print(f"常态最优参数：{best_normal}")
        print(f"极端态最优参数：{best_extreme}")
        print(f"最优 红3+ 命中率：{best_summary['best_red_3plus_rate']:.2%}")
        print(f"最优 红4+ 命中率：{best_summary['best_red_4plus_rate']:.2%}")
        print(f"最优 蓝1 命中率：{best_summary['best_blue_1plus_rate']:.2%}")
        print("Top 5 候选：")
        for idx, row in enumerate(ranked[:5], start=1):
            s = row["summary"]
            print(
                f"{idx}. score={row['score']:.3f} "
                f"red3+={s['best_red_3plus_rate']:.2%} "
                f"red4+={s['best_red_4plus_rate']:.2%} "
                f"blue1+={s['best_blue_1plus_rate']:.2%} "
                f"normal={row['normal_overrides']} "
                f"extreme={row['extreme_overrides']}"
            )
        return

    if args.grid_search_stability:
        best_normal, best_extreme, best_summary, ranked = run_stability_grid_search(draws, args.window_size, args.ticket_count)
        print("分段稳定性优先网格搜索完成。")
        print(f"常态最优参数：{best_normal}")
        print(f"极端态最优参数：{best_extreme}")
        print(f"最优 红3+ 命中率：{best_summary['best_red_3plus_rate']:.2%}")
        print(f"最优 红4+ 命中率：{best_summary['best_red_4plus_rate']:.2%}")
        print(f"最优 蓝1 命中率：{best_summary['best_blue_1plus_rate']:.2%}")
        print("分段表现：")
        for segment, item in best_summary["segment_breakdown"].items():
            print(
                f"- {segment}: red3+={item['best_red_3plus_rate']:.2%}, "
                f"red4+={item['best_red_4plus_rate']:.2%}, "
                f"blue1+={item['best_blue_1plus_rate']:.2%}, windows={item['windows']}"
            )
        print("Top 5 候选：")
        for idx, row in enumerate(ranked[:5], start=1):
            s = row["summary"]
            print(
                f"{idx}. score={row['score']:.3f} "
                f"red3+={s['best_red_3plus_rate']:.2%} "
                f"red4+={s['best_red_4plus_rate']:.2%} "
                f"blue1+={s['best_blue_1plus_rate']:.2%} "
                f"normal={row['normal_overrides']} "
                f"extreme={row['extreme_overrides']}"
            )
        return

    fixed_results, fixed_summary = rolling_backtest(
        draws, args.window_size, args.ticket_count, adaptive=False, preset=args.preset
    )
    adaptive_results, adaptive_summary = rolling_backtest(
        draws, args.window_size, args.ticket_count, adaptive=True, preset=args.preset
    )
    comparison = compare_summaries(fixed_summary, adaptive_summary)
    report = build_report(
        fixed_results,
        fixed_summary,
        adaptive_results,
        adaptive_summary,
        comparison,
        args.window_size,
        args.ticket_count,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"已生成报告：{report_path}")
    print(f"滚动窗口数：{adaptive_summary['windows']}")
    print(f"固定规则 红3+命中率：{fixed_summary['best_red_3plus_rate']:.2%}")
    print(f"自适应规则 红3+命中率：{adaptive_summary['best_red_3plus_rate']:.2%}")
    print(f"固定规则 蓝1命中率：{fixed_summary['best_blue_1plus_rate']:.2%}")
    print(f"自适应规则 蓝1命中率：{adaptive_summary['best_blue_1plus_rate']:.2%}")


if __name__ == "__main__":
    main()
