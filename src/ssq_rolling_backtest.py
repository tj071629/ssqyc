from __future__ import annotations

import argparse
import itertools
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


RED_RANGE = range(1, 34)
BLUE_RANGE = range(1, 17)
WINDOW_SIZE = 50
DEFAULT_TICKET_COUNT = 5
# Portfolio filter: max overlap of 6 reds with any prior draw must be exactly this count.
EXACT_FULL_HISTORY_RED_OVERLAP = 4

# Full-history stability search defaults (3394 rolling windows).
FULL_HISTORY_RED_NORMAL = {"hot10": 0.0, "hot20": 0.01, "repeat_last": 0.0, "neighbor_last": 0.0, "omission_mid": 0.04}
FULL_HISTORY_RED_EXTREME = {"hot20": 0.04, "omission_mid": 0.08, "zone_cold": 0.04, "hot50": 0.04}

# Last-300-draw stability fine-tune (--recent-draws 300 --grid-search-stability).
RECENT_300_RED_NORMAL = {"hot10": 0.0, "hot20": 0.01, "repeat_last": 0.0, "neighbor_last": 0.0, "omission_mid": 0.03}
RECENT_300_RED_EXTREME = {"hot20": 0.04, "omission_mid": 0.08, "zone_cold": 0.04, "hot50": 0.02}
# Blue: `blue_base_score` + weighted_bonus(blue_weights, blue_features). Tuned 2026-05-11 via --grid-search-blue-near300
# (KPI: any_blue_hit on last-300 tail; picked minimal non-zero set that beats baseline without hurting red3+).


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


RECENT_300_BLUE_WEIGHTS: dict[str, float] = {
    "hot5": 0.0,
    "hot10": 0.0,
    "repeat_last": 0.0,
    "neighbor_last": 0.0,
    "neighbor_prev": 0.0,
    "omission_step": 0.0,
    "omission_mid": 0.08,
    "omission_deep": 0.0,
}


def build_state(
    adaptive: bool,
    overrides: dict[str, float] | None = None,
    extreme_overrides: dict[str, float] | None = None,
    *,
    preset: str = "full",
    blue_weight_overrides: dict[str, float] | None = None,
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
    if preset == "recent300":
        for name in BLUE_FEATURES:
            state.blue_weights[name] = RECENT_300_BLUE_WEIGHTS.get(name, 0.0)
    if blue_weight_overrides:
        for name, value in blue_weight_overrides.items():
            if name in state.blue_weights:
                state.blue_weights[name] = value
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


def is_zone1_break_last(window: list[Draw]) -> bool:
    """True when the training window ends on a draw with zero reds in zone1 (01-11)."""
    return red_zone_signature(window[-1].red)[0] == 0


def zone1_break_follow_red_counts(draws: list[Draw]) -> dict[int, int]:
    """For each draw with no zone1 reds, count reds in the immediately following draw (full `draws` timeline)."""
    counts: Counter[int] = Counter()
    for i in range(len(draws) - 1):
        if red_zone_signature(draws[i].red)[0] != 0:
            continue
        for number in draws[i + 1].red:
            counts[number] += 1
    return dict(counts)


def refine_red_ranked_zone1_follow_ties(scores: dict[int, float], follow_counts: dict[int, int]) -> list[int]:
    """Same as baseline red order, but when model scores tie, prefer reds that historically hit more often right after a zone1 break."""
    return sorted(RED_RANGE, key=lambda n: (-scores[n], -follow_counts.get(n, 0), n))


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
        # Tail blues (13-16) deep-cold hits (e.g. 2026055 blue 13) were over-penalized vs small-band hot bias.
        score -= 0.05 if number >= 13 else 0.2
    return score


def is_small_blue_band_hot(window: list[Draw]) -> bool:
    """True when recent draws cluster in 01-11 (blue-ball专规 v0.5 §2.12)."""
    return sum(1 for draw in window[-10:] if draw.blue <= 11) >= 6


def pick_blue_pool_five(window: list[Draw], blue_ranked: list[int]) -> list[int]:
    """Up to 5 distinct blues: repeat/neighbor, optional mid 08-12, model top, optional tail 13-16."""
    last = window[-1]
    prev = window[-2]
    rank_index = {number: idx for idx, number in enumerate(blue_ranked)}
    blues: list[int] = []

    def add(number: int) -> None:
        if number in BLUE_RANGE and number not in blues:
            blues.append(number)

    add(choose_blue(blue_ranked, include=[last.blue]))
    for neighbor in (last.blue - 1, last.blue + 1, prev.blue - 1, prev.blue + 1):
        if neighbor in BLUE_RANGE:
            add(choose_blue(blue_ranked, include=[neighbor]))

    if is_small_blue_band_hot(window):
        for number in sorted(range(8, 13), key=lambda b: (rank_index.get(b, 99), -omission(window, "blue", b), b)):
            add(number)
            if len(blues) >= 3:
                break

    for number in blue_ranked:
        add(number)
        if len(blues) >= 5:
            break

    if is_small_blue_band_hot(window):
        for number in sorted(range(13, 17), key=lambda b: (-omission(window, "blue", b), rank_index.get(b, 99), b)):
            if number in blues:
                continue
            if len(blues) >= 5:
                blues[-1] = number
            else:
                blues.append(number)
            break

    return blues[:5]


def apply_diverse_blues(tickets: list[Ticket], window: list[Draw], blue_ranked: list[int]) -> list[Ticket]:
    """Assign five distinct blues from `pick_blue_pool_five` to improve any-blue hit rate."""
    pool = pick_blue_pool_five(window, blue_ranked)
    used: list[int] = []
    diversified: list[Ticket] = []
    for ticket in tickets:
        blue: int | None = None
        for candidate in pool + [number for number in blue_ranked if number not in pool]:
            if candidate not in used:
                blue = candidate
                break
        blue = blue if blue is not None else pool[0]
        used.append(blue)
        note = ticket.note if ticket.blue == blue else f"{ticket.note}·蓝分散"
        diversified.append(Ticket(ticket.name, ticket.red, blue, note))
    return diversified


def score_red(window: list[Draw], state: StrategyState) -> tuple[list[int], dict[int, dict[str, float]], dict[int, float]]:
    context = build_red_context(window)
    red_weights = state.red_weights_extreme if is_extreme_window(window) else state.red_weights
    scores: dict[int, float] = {}
    feature_table: dict[int, dict[str, float]] = {}
    for number in RED_RANGE:
        features = red_features(number, context)
        feature_table[number] = features
        scores[number] = red_base_score(number, context) + weighted_bonus(red_weights, features)
    ranked = sorted(scores, key=lambda n: (-scores[n], n))
    return ranked, feature_table, scores


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


def choose_red_by_zone_targets(
    ranked: list[int],
    z1_need: int,
    z2_need: int,
    z3_need: int,
    *,
    include: Iterable[int] = (),
    exclude: Iterable[int] = (),
    zone_skips: tuple[int, int, int] | None = None,
) -> tuple[int, ...]:
    """Pick 6 reds with exact zone counts (low / mid / high). Fills from `ranked` per zone; relaxes skips if a zone runs dry."""
    assert z1_need + z2_need + z3_need == 6
    zone_skips = zone_skips or (0, 0, 0)
    excluded = set(exclude)
    picked: list[int] = []
    needs = [z1_need, z2_need, z3_need]
    counts = [0, 0, 0]

    for number in sorted(set(include)):
        if number in excluded or number in picked:
            continue
        zone = red_zone_index(number)
        if counts[zone] < needs[zone]:
            picked.append(number)
            counts[zone] += 1

    for zone in range(3):
        skip_remaining = zone_skips[zone]
        for number in ranked:
            if counts[zone] >= needs[zone]:
                break
            if number in excluded or number in picked:
                continue
            if red_zone_index(number) != zone:
                continue
            if skip_remaining > 0:
                skip_remaining -= 1
                continue
            picked.append(number)
            counts[zone] += 1

    for zone in range(3):
        for number in ranked:
            if counts[zone] >= needs[zone]:
                break
            if number in excluded or number in picked:
                continue
            if red_zone_index(number) != zone:
                continue
            picked.append(number)
            counts[zone] += 1

    if len(picked) < 6:
        for number in ranked:
            if number in picked or number in excluded:
                continue
            picked.append(number)
            if len(picked) == 6:
                break

    return tuple(sorted(picked[:6]))


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


def distinct_red_preserve_zone_signature(
    candidate: tuple[int, ...],
    existing: list[Ticket],
    ranked: list[int],
    required_sig: tuple[int, int, int],
) -> tuple[int, ...]:
    """Like distinct_red but only swaps within the same zone so (z1,z2,z3) stays valid for 断一区后 structured tickets."""
    existing_reds = {ticket.red for ticket in existing}
    if candidate not in existing_reds:
        return candidate
    if red_zone_signature(candidate) != required_sig:
        return distinct_red(candidate, existing, ranked)
    base = sorted(candidate)
    for idx in range(6):
        zone = red_zone_index(base[idx])
        for number in ranked:
            if number in base or red_zone_index(number) != zone:
                continue
            alt = sorted(base[:idx] + [number] + base[idx + 1 :])
            alt_tuple = tuple(alt)
            if alt_tuple not in existing_reds and red_zone_signature(alt_tuple) == required_sig:
                return alt_tuple
    return distinct_red(candidate, existing, ranked)


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


def _zone1_break_neighbor_order(last: Draw) -> list[int]:
    neighbors_last = sorted({n for value in last.red for n in (value - 1, value + 1) if n in RED_RANGE})
    low_first = [n for n in neighbors_last if red_zone_index(n) == 0]
    mid = [n for n in neighbors_last if red_zone_index(n) == 1]
    high = [n for n in neighbors_last if red_zone_index(n) == 2]
    return low_first + mid + high


def build_portfolio_after_zone1_break(red_ranked: list[int], blue_ranked: list[int], window: list[Draw]) -> list[Ticket]:
    """After a zone1 break (last window draw has no 01-11), bias next-ticket reds toward historical follow-ups.

    Empirical skew (from project stats): zone2 often ~2; 2:2:2 / 3:2:1 / 2:3:1 common; zone3 usually >=1 (often >=2).
    """
    last = window[-1]
    prev = window[-2]
    neighbor_pref = _zone1_break_neighbor_order(last)
    blue_neighbors = [n for n in (last.blue - 1, last.blue + 1, prev.blue - 1, prev.blue + 1) if n in BLUE_RANGE]

    t1_red = choose_red_by_zone_targets(red_ranked, 2, 2, 2, include=neighbor_pref[:4])
    t2_red = choose_red_by_zone_targets(red_ranked, 2, 2, 2, include=neighbor_pref[2:6], zone_skips=(1, 0, 1))
    t3_red = choose_red_by_zone_targets(red_ranked, 2, 3, 1, include=neighbor_pref[:3])
    t4_red = choose_red_by_zone_targets(red_ranked, 3, 2, 1, include=neighbor_pref[:4])
    t5_red = choose_red_by_zone_targets(red_ranked, 1, 2, 3, include=neighbor_pref[:2])

    return [
        Ticket("主线票", t1_red, choose_blue(blue_ranked, include=[last.blue]), "断一区后·均衡2:2:2+邻号低区回补"),
        Ticket("重号票", t2_red, choose_blue(blue_ranked, include=blue_neighbors[:1]), "断一区后·均衡2:2:2变序"),
        Ticket("邻号票", t3_red, choose_blue(blue_ranked), "断一区后·二区偏重2:3:1"),
        Ticket("均衡票", t4_red, choose_blue(blue_ranked, include=blue_neighbors[1:2]), "断一区后·低区回补3:2:1"),
        Ticket("跳点票", t5_red, choose_blue(blue_ranked, reverse_jump=True, reverse_from=6), "断一区后·高区加重1:2:3"),
    ]


def build_zone1_break_supplement_tickets(
    red_ranked: list[int],
    blue_ranked: list[int],
    window: list[Draw],
    existing: list[Ticket],
    *,
    count: int = 4,
) -> list[Ticket]:
    """Optional extra tickets after the 5-note zone1-follow portfolio (not used in rolling backtest by default)."""
    last = window[-1]
    prev = window[-2]
    neighbor_pref = _zone1_break_neighbor_order(last)
    blue_neighbors = [n for n in (last.blue - 1, last.blue + 1, prev.blue - 1, prev.blue + 1) if n in BLUE_RANGE]
    cold_jump = [number for number in red_ranked[10:24] if number not in last.red]

    blue_nb0 = choose_blue(blue_ranked, include=blue_neighbors[:1])
    blue_nb1 = choose_blue(blue_ranked, include=blue_neighbors[1:2]) if len(blue_neighbors) > 1 else choose_blue(blue_ranked)

    specs: list[tuple[str, tuple[int, ...], int, str, bool]] = [
        (
            "加推1",
            choose_red(red_ranked, include=cold_jump[:2]),
            choose_blue(blue_ranked, reverse_jump=True, reverse_from=6),
            "断一区后·加推冷补线",
            False,
        ),
        (
            "加推2",
            choose_red_by_zone_targets(red_ranked, 1, 3, 2, include=neighbor_pref[:3]),
            blue_nb0,
            "断一区后·加推1:3:2",
            True,
        ),
        (
            "加推3",
            choose_red_by_zone_targets(red_ranked, 4, 1, 1, include=neighbor_pref[:4]),
            choose_blue(blue_ranked),
            "断一区后·加推4:1:1",
            True,
        ),
        (
            "加推4",
            choose_red_by_zone_targets(red_ranked, 3, 1, 2, include=neighbor_pref[:4]),
            blue_nb1,
            "断一区后·加推3:1:2",
            True,
        ),
    ]

    ex = list(existing)
    out: list[Ticket] = []
    for name, red, blue, note, fixed_sig in specs[: max(0, min(count, len(specs)))]:
        if fixed_sig:
            sig = red_zone_signature(red)
            red2 = distinct_red_preserve_zone_signature(red, ex, red_ranked, sig)
        else:
            red2 = distinct_red(red, ex, red_ranked)
        t = Ticket(name, red2, blue, note)
        ex.append(t)
        out.append(t)
    return out


def max_red_overlap_with_prior_reds(red: tuple[int, ...], prior_red_sets: Sequence[frozenset[int]]) -> int:
    """Largest count of shared reds between `red` and any single prior draw."""
    current = frozenset(red)
    if not prior_red_sets:
        return 0
    return max(len(current & prior_set) for prior_set in prior_red_sets)


def _single_swap_red_candidates(
    red: tuple[int, ...],
    ranked: list[int],
    *,
    preserve_zone_signature: tuple[int, int, int] | None,
) -> Iterable[tuple[int, ...]]:
    current = list(red)
    for idx in range(6):
        zone = red_zone_index(current[idx]) if preserve_zone_signature is not None else None
        for number in ranked:
            if number in current:
                continue
            if preserve_zone_signature is not None and red_zone_index(number) != zone:
                continue
            yield tuple(sorted(current[:idx] + [number] + current[idx + 1 :]))


def adjust_red_exact_full_history_overlap(
    red: tuple[int, ...],
    prior_red_sets: Sequence[frozenset[int]],
    ranked: list[int],
    *,
    target: int = EXACT_FULL_HISTORY_RED_OVERLAP,
    existing_reds: set[tuple[int, ...]] | None = None,
    preserve_zone_signature: tuple[int, int, int] | None = None,
    max_passes: int = 96,
) -> tuple[int, ...]:
    """Greedy single-ball swaps until max prior overlap equals `target` (default 4)."""
    existing_reds = existing_reds or set()
    best = tuple(sorted(red))
    if not prior_red_sets:
        return best

    def sort_key(candidate: tuple[int, ...]) -> tuple[int, int, int]:
        overlap = max_red_overlap_with_prior_reds(candidate, prior_red_sets)
        duplicate = int(candidate in existing_reds)
        rank_penalty = sum(ranked.index(number) if number in ranked else 99 for number in candidate)
        return (duplicate, abs(overlap - target), rank_penalty)

    for _ in range(max_passes):
        overlap = max_red_overlap_with_prior_reds(best, prior_red_sets)
        if overlap == target and best not in existing_reds:
            return best
        improved: tuple[int, ...] | None = None
        improved_key = sort_key(best)
        for candidate in _single_swap_red_candidates(
            best, ranked, preserve_zone_signature=preserve_zone_signature
        ):
            candidate_key = sort_key(candidate)
            if candidate_key < improved_key:
                improved_key = candidate_key
                improved = candidate
        if improved is None:
            break
        best = improved
    return best


def apply_exact_four_red_overlap_portfolio(
    tickets: list[Ticket],
    prior_draws: list[Draw],
    red_ranked: list[int],
    *,
    zone1_follow: bool,
) -> list[Ticket]:
    """Enforce exact full-history max red overlap (default 4) on each ticket."""
    prior_red_sets = [frozenset(draw.red) for draw in prior_draws]
    existing: set[tuple[int, ...]] = set()
    adjusted: list[Ticket] = []
    suffix = "·全历史红重合4"
    for ticket in tickets:
        preserve_sig = red_zone_signature(ticket.red) if zone1_follow else None
        red = adjust_red_exact_full_history_overlap(
            ticket.red,
            prior_red_sets,
            red_ranked,
            existing_reds=existing,
            preserve_zone_signature=preserve_sig,
        )
        existing.add(red)
        note = ticket.note if suffix in ticket.note else f"{ticket.note}{suffix}"
        adjusted.append(Ticket(ticket.name, red, ticket.blue, note))
    return adjusted


def generate_tickets(
    window: list[Draw],
    state: StrategyState,
    ticket_count: int,
    *,
    zone1_follow_red_counts: dict[int, int] | None = None,
    supplement_zone1_break: int = 0,
    prior_draws: list[Draw] | None = None,
    enforce_exact_four_red_overlap: bool = True,
) -> tuple[list[Ticket], list[int], list[int], dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    red_ranked, red_features_table, red_scores = score_red(window, state)
    if (
        ticket_count == 5
        and is_zone1_break_last(window)
        and zone1_follow_red_counts is not None
    ):
        red_ranked = refine_red_ranked_zone1_follow_ties(red_scores, zone1_follow_red_counts)
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
        zone1_follow = is_zone1_break_last(window)
        portfolio = (
            build_portfolio_after_zone1_break(red_ranked, blue_ranked, window)
            if zone1_follow
            else build_red_priority_portfolio(red_ranked, blue_ranked, window)
        )
        for t in portfolio:
            if zone1_follow:
                sig = red_zone_signature(t.red)
                red = distinct_red_preserve_zone_signature(t.red, tickets, red_ranked, sig)
            else:
                red = distinct_red(t.red, tickets, red_ranked)
            tickets.append(Ticket(t.name, red, t.blue, t.note))
        tickets = apply_diverse_blues(tickets, window, blue_ranked)
        if zone1_follow and supplement_zone1_break > 0:
            tickets.extend(
                build_zone1_break_supplement_tickets(
                    red_ranked,
                    blue_ranked,
                    window,
                    tickets,
                    count=supplement_zone1_break,
                )
            )
    else:
        ordered_templates = sorted(TEMPLATE_NAMES, key=lambda name: (-state.template_scores[name], TEMPLATE_NAMES.index(name)))
        chosen = ordered_templates[:ticket_count]
        for name in chosen:
            t = template_map[name]
            red = distinct_red(t.red, tickets, red_ranked)
            tickets.append(Ticket(t.name, red, t.blue, t.note))
    if enforce_exact_four_red_overlap and prior_draws:
        zone1_follow = is_zone1_break_last(window)
        tickets = apply_exact_four_red_overlap_portfolio(
            tickets, prior_draws, red_ranked, zone1_follow=zone1_follow
        )
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
    blue_weight_overrides: dict[str, float] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    state = build_state(
        adaptive=adaptive,
        overrides=red_weight_overrides,
        extreme_overrides=red_weight_extreme_overrides,
        preset=preset,
        blue_weight_overrides=blue_weight_overrides,
    )
    aggregate = defaultdict(int)
    aggregate_lists: dict[str, list[float]] = defaultdict(list)
    segment_aggregate: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    results: list[dict[str, object]] = []

    zone1_follow_red = zone1_break_follow_red_counts(draws)
    for start in range(0, len(draws) - window_size):
        window = draws[start : start + window_size]
        actual = draws[start + window_size]
        tickets, red_ranked, blue_ranked, red_table, blue_table = generate_tickets(
            window,
            state,
            ticket_count,
            zone1_follow_red_counts=zone1_follow_red,
            prior_draws=draws[: start + window_size],
        )
        aggregate["extreme_windows"] += int(is_extreme_window(window))
        hits = [hit_summary(ticket, actual) for ticket in tickets]
        best = max(hits, key=lambda h: (h["red_hits"], h["blue_hits"]))
        any_same_ticket_4p1 = any(h["red_hits"] >= 4 and h["blue_hits"] == 1 for h in hits)
        any_blue_hit = int(any(h["blue_hits"] >= 1 for h in hits))

        aggregate["windows"] += 1
        aggregate["best_red_3plus"] += int(best["red_hits"] >= 3)
        aggregate["best_red_4plus"] += int(best["red_hits"] >= 4)
        aggregate["best_blue_1plus"] += int(best["blue_hits"] >= 1)
        aggregate["any_blue_hit"] += any_blue_hit
        aggregate["same_ticket_red4_blue1"] += int(any_same_ticket_4p1)
        aggregate_lists["best_red_hits"].append(best["red_hits"])
        aggregate_lists["best_blue_hits"].append(best["blue_hits"])
        segment = actual.issue[:4]
        segment_aggregate[segment]["windows"] += 1
        segment_aggregate[segment]["best_red_3plus"] += int(best["red_hits"] >= 3)
        segment_aggregate[segment]["best_red_4plus"] += int(best["red_hits"] >= 4)
        segment_aggregate[segment]["best_blue_1plus"] += int(best["blue_hits"] >= 1)
        segment_aggregate[segment]["any_blue_hit"] += any_blue_hit

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
        "any_blue_hit_rate": aggregate["any_blue_hit"] / aggregate["windows"],
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
                "any_blue_hit_rate": values["any_blue_hit"] / values["windows"],
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
        "any_blue_hit_delta": adaptive["any_blue_hit_rate"] - fixed["any_blue_hit_rate"],
        "same_ticket_red4_blue1_delta": adaptive["same_ticket_red4_blue1_rate"] - fixed["same_ticket_red4_blue1_rate"],
        "avg_best_red_hits_delta": adaptive["avg_best_red_hits"] - fixed["avg_best_red_hits"],
    }


def run_near300_blue_grid_search(
    draws: list[Draw],
    window_size: int,
    ticket_count: int,
) -> tuple[dict[str, float], dict[str, object], list[dict[str, object]]]:
    """Grid `blue_weights` on last-300 tail; KPI: any_blue_hit_rate then best_red_3plus_rate (fixed, preset recent300)."""
    draws_tail = slice_recent_draws(draws, 300)
    axes: tuple[tuple[str, tuple[float, ...]], ...] = (
        ("neighbor_last", (0.0, 0.02, 0.04, 0.06)),
        ("repeat_last", (0.0, 0.02, 0.04)),
        ("omission_mid", (0.0, 0.04, 0.08)),
        ("omission_deep", (0.0, 0.04)),
    )
    names = [a[0] for a in axes]
    grids = [a[1] for a in axes]
    ranked_rows: list[dict[str, object]] = []
    for values in itertools.product(*grids):
        weights = {name: 0.0 for name in BLUE_FEATURES}
        for name, value in zip(names, values):
            weights[name] = value
        _, summary = rolling_backtest(
            draws_tail,
            window_size,
            ticket_count,
            adaptive=False,
            preset="recent300",
            blue_weight_overrides=weights,
        )
        ranked_rows.append({"blue_weights": dict(weights), "summary": summary})
    ranked_rows.sort(
        key=lambda r: (
            -r["summary"]["any_blue_hit_rate"],
            -r["summary"]["best_red_3plus_rate"],
            -r["summary"]["best_red_4plus_rate"],
        )
    )
    best = ranked_rows[0]
    return best["blue_weights"], best["summary"], ranked_rows


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
    lines.append(
        f"| 最优注蓝球命中(按红优先取最优注) | `{fixed_summary['best_blue_1plus_rate']:.2%}` | `{adaptive_summary['best_blue_1plus_rate']:.2%}` | `{comparison['best_blue_1plus_delta']:+.2%}` |"
    )
    lines.append(
        f"| 5注任一蓝球命中 | `{fixed_summary['any_blue_hit_rate']:.2%}` | `{adaptive_summary['any_blue_hit_rate']:.2%}` | `{comparison['any_blue_hit_delta']:+.2%}` |"
    )
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
    parser.add_argument(
        "--grid-search-blue-near300",
        action="store_true",
        help="在最近300期截尾上网格搜索蓝球特征权重（preset=recent300），主 KPI：五注任一蓝中；辅 KPI：红3+",
    )
    args = parser.parse_args()

    draws = parse_history(Path(args.history))
    draws = slice_recent_draws(draws, args.recent_draws)
    if len(draws) <= args.window_size:
        raise SystemExit("历史数据不足，无法完成滚动回测。")

    if args.grid_search_blue_near300:
        draws_full = parse_history(Path(args.history))
        tail = slice_recent_draws(draws_full, 300)
        if len(tail) <= args.window_size:
            raise SystemExit("历史数据不足，无法完成近端蓝球网格搜索。")
        _, baseline_s = rolling_backtest(
            tail,
            args.window_size,
            args.ticket_count,
            adaptive=False,
            preset="recent300",
            blue_weight_overrides={name: 0.0 for name in BLUE_FEATURES},
        )
        best_w, best_s, all_rows = run_near300_blue_grid_search(draws_full, args.window_size, args.ticket_count)
        print("近端蓝球权重网格搜索完成（最近300期截尾 + preset=recent300 + 固定72组组合）。")
        print(
            f"基线(蓝特征权全0): 五注任一蓝中={baseline_s['any_blue_hit_rate']:.2%} "
            f"红3+={baseline_s['best_red_3plus_rate']:.2%}"
        )
        print(
            f"最优: 五注任一蓝中={best_s['any_blue_hit_rate']:.2%} "
            f"红3+={best_s['best_red_3plus_rate']:.2%} "
            f"最优注蓝(红优)={best_s['best_blue_1plus_rate']:.2%}"
        )
        print("推荐 blue_weights（非零项）:", {k: v for k, v in best_w.items() if v != 0.0})
        print("Top 8:")
        for idx, row in enumerate(all_rows[:8], start=1):
            s = row["summary"]
            w = row["blue_weights"]
            compact = {k: v for k, v in w.items() if v != 0.0}
            print(
                f"  {idx}. any_blue={s['any_blue_hit_rate']:.2%} red3+={s['best_red_3plus_rate']:.2%} "
                f"red4+={s['best_red_4plus_rate']:.2%} w={compact}"
            )
        _, full_s = rolling_backtest(
            draws_full,
            args.window_size,
            args.ticket_count,
            adaptive=False,
            preset="recent300",
            blue_weight_overrides=best_w,
        )
        print(
            f"全历史校验(套用最优蓝权): 五注任一蓝中={full_s['any_blue_hit_rate']:.2%} "
            f"红3+={full_s['best_red_3plus_rate']:.2%}"
        )
        return

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
    print(f"固定规则 最优注蓝中(红优)：{fixed_summary['best_blue_1plus_rate']:.2%}")
    print(f"自适应规则 最优注蓝中(红优)：{adaptive_summary['best_blue_1plus_rate']:.2%}")
    print(f"固定规则 五注任一蓝中：{fixed_summary['any_blue_hit_rate']:.2%}")
    print(f"自适应规则 五注任一蓝中：{adaptive_summary['any_blue_hit_rate']:.2%}")


if __name__ == "__main__":
    main()
