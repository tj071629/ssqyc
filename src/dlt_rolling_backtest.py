from __future__ import annotations

import argparse
import copy
import itertools
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


FRONT_RANGE = range(1, 36)
BACK_RANGE = range(1, 13)
WINDOW_SIZE = 50
DEFAULT_TICKET_COUNT = 5
MANDATORY_TEMPLATES = ("主承接票", "断区反抽票", "逆向跳号票")


@dataclass(frozen=True)
class Draw:
    issue: str
    date: str
    front: tuple[int, ...]
    back: tuple[int, ...]


@dataclass(frozen=True)
class Ticket:
    name: str
    front: tuple[int, ...]
    back: tuple[int, ...]
    note: str


@dataclass
class StrategyState:
    adaptive: bool
    front_bonus_weights: dict[str, float] = field(default_factory=dict)
    back_bonus_weights: dict[str, float] = field(default_factory=dict)
    template_scores: dict[str, float] = field(default_factory=dict)
    front_learning_rate: float = 0.0
    back_learning_rate: float = 0.0
    template_learning_rate: float = 0.0
    front_weight_decay: float = 1.0
    back_weight_decay: float = 1.0
    template_decay: float = 1.0
    template_back_hit_weight: float = 1.7
    template_back_hit_bonus: float = 0.55
    template_front_hit_bonus: float = 0.25
    template_front3_hit_bonus: float = 0.0
    template_joint_3p1_bonus: float = 0.0
    template_reward_baseline: float = 1.1


@dataclass(frozen=True)
class EarlyStopConfig:
    enabled: bool = False
    metric: str = "same_ticket_front3_back1"
    eval_interval: int = 50
    eval_window: int = 200
    patience: int = 3
    min_delta: float = 0.0005


FRONT_FEATURE_NAMES = (
    "hot10",
    "hot20",
    "hot50",
    "repeat_last",
    "neighbor_last",
    "neighbor_prev",
    "jump2_last",
    "omission_step",
    "omission_mid",
    "omission_warm",
    "omission_deep",
    "pair_synergy",
    "zone_cold",
)

BACK_FEATURE_NAMES = (
    "hot5",
    "hot10",
    "repeat_last",
    "neighbor_last",
    "neighbor_prev",
    "omission_step",
    "omission_mid",
    "omission_deep",
)

TEMPLATE_NAMES = (
    "主承接票",
    "骨架回补票",
    "结构平衡票",
    "断区反抽票",
    "逆向跳号票",
    "重号集中票",
    "中区加压票",
    "极端延续票",
    "回稳反抽票",
    "后区活跃票",
    "后区回补票",
)


def snapshot_state(state: StrategyState) -> dict[str, object]:
    return {
        "front_bonus_weights": copy.deepcopy(state.front_bonus_weights),
        "back_bonus_weights": copy.deepcopy(state.back_bonus_weights),
        "template_scores": copy.deepcopy(state.template_scores),
    }


def restore_state(state: StrategyState, snapshot: dict[str, object]) -> None:
    state.front_bonus_weights = copy.deepcopy(snapshot["front_bonus_weights"])
    state.back_bonus_weights = copy.deepcopy(snapshot["back_bonus_weights"])
    state.template_scores = copy.deepcopy(snapshot["template_scores"])


def metric_value_for_row(metric: str, row: dict[str, object]) -> float:
    if metric == "same_ticket_front3_back1":
        return float(any(hit["front_hits"] >= 3 and hit["back_hits"] >= 1 for hit in row["hits"]))
    if metric == "best_front_2plus":
        return float(row["best_front_hits"] >= 2)
    if metric == "best_front_3plus":
        return float(row["best_front_hits"] >= 3)
    if metric == "best_back_1plus":
        return float(row["best_back_hits"] >= 1)
    if metric == "best_back_exact":
        return float(row["best_back_hits"] == 2)
    raise ValueError(f"不支持的早停指标: {metric}")


def parse_history(path: Path) -> list[Draw]:
    draws: list[Draw] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 4 or parts[0] in {"期号", "-------"}:
            continue
        issue, date, front_raw, back_raw = parts
        front = tuple(sorted(int(value) for value in front_raw.split()))
        back = tuple(sorted(int(value) for value in back_raw.split()))
        if len(front) != 5 or len(back) != 2:
            continue
        draws.append(Draw(issue=issue, date=date, front=front, back=back))
    return draws


def build_state(
    adaptive: bool,
    *,
    front_learning_rate: float = 0.0,
    back_learning_rate: float = 0.0015,
    template_learning_rate: float = 0.03,
    front_weight_decay: float = 1.0,
    back_weight_decay: float = 0.999,
    template_decay: float = 0.9975,
    template_back_hit_weight: float = 1.1,
    template_back_hit_bonus: float = 0.3,
    template_front_hit_bonus: float = 0.25,
    template_front3_hit_bonus: float = 0.0,
    template_joint_3p1_bonus: float = 0.0,
    template_reward_baseline: float = 1.1,
) -> StrategyState:
    return StrategyState(
        adaptive=adaptive,
        front_bonus_weights={name: 0.0 for name in FRONT_FEATURE_NAMES},
        back_bonus_weights={name: 0.0 for name in BACK_FEATURE_NAMES},
        template_scores={name: 1.0 for name in TEMPLATE_NAMES},
        front_learning_rate=front_learning_rate if adaptive else 0.0,
        back_learning_rate=back_learning_rate if adaptive else 0.0,
        template_learning_rate=template_learning_rate if adaptive else 0.0,
        front_weight_decay=front_weight_decay if adaptive else 1.0,
        back_weight_decay=back_weight_decay if adaptive else 1.0,
        template_decay=template_decay if adaptive else 1.0,
        template_back_hit_weight=template_back_hit_weight,
        template_back_hit_bonus=template_back_hit_bonus,
        template_front_hit_bonus=template_front_hit_bonus,
        template_front3_hit_bonus=template_front3_hit_bonus,
        template_joint_3p1_bonus=template_joint_3p1_bonus,
        template_reward_baseline=template_reward_baseline,
    )


def omission(draws: list[Draw], attr: str, number: int) -> int:
    for offset, draw in enumerate(reversed(draws), start=1):
        values = draw.front if attr == "front" else draw.back
        if number in values:
            return offset - 1
    return len(draws)


def zone_index(number: int) -> int:
    if 1 <= number <= 12:
        return 0
    if 13 <= number <= 24:
        return 1
    return 2


def zone_signature(front: Iterable[int]) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    for number in front:
        counts[zone_index(number)] += 1
    return tuple(counts)


def format_numbers(numbers: Iterable[int]) -> str:
    return " ".join(f"{number:02d}" for number in numbers)


def build_front_context(window: list[Draw]) -> dict[str, object]:
    recent_10 = window[-10:]
    recent_20 = window[-20:]
    recent_6 = window[-6:]
    last = window[-1]
    prev = window[-2]
    return {
        "window": window,
        "last": last,
        "prev": prev,
        "recent_6": recent_6,
        "front_counter_10": Counter(number for draw in recent_10 for number in draw.front),
        "front_counter_20": Counter(number for draw in recent_20 for number in draw.front),
        "front_counter_50": Counter(number for draw in window for number in draw.front),
        "pair_counter": Counter(
            pair
            for draw in window
            for pair in itertools.combinations(draw.front, 2)
        ),
    }


def build_back_context(window: list[Draw]) -> dict[str, object]:
    recent_5 = window[-5:]
    recent_10 = window[-10:]
    return {
        "window": window,
        "last": window[-1],
        "prev": window[-2],
        "counter_5": Counter(number for draw in recent_5 for number in draw.back),
        "counter_10": Counter(number for draw in recent_10 for number in draw.back),
    }


def front_base_score(number: int, context: dict[str, object]) -> float:
    window = context["window"]
    last = context["last"]
    prev = context["prev"]
    recent_6 = context["recent_6"]
    front_counter_10 = context["front_counter_10"]
    front_counter_20 = context["front_counter_20"]
    front_counter_50 = context["front_counter_50"]
    pair_counter = context["pair_counter"]

    miss = omission(window, "front", number)
    score = 0.0
    score += front_counter_10[number] * 3.2
    score += front_counter_20[number] * 1.5
    score += front_counter_50[number] * 0.35

    if number in last.front:
        score += 4.0
    if any(abs(number - value) == 1 for value in last.front):
        score += 4.6
    if any(abs(number - value) == 1 for value in prev.front):
        score += 1.9
    if any(abs(number - value) == 2 for value in last.front):
        score += 0.9

    score += min(miss, 12) * 0.32
    if 4 <= miss <= 9:
        score += 1.2
    if 10 <= miss <= 16:
        score += 0.7
    if miss >= 20:
        score -= 0.3

    synergy = sum(pair_counter[tuple(sorted((number, mate)))] for mate in last.front if mate != number)
    score += synergy * 0.15

    zone_hits = [0, 0, 0]
    for draw in recent_6:
        zone_hits[zone_index(number)] += number in draw.front
    if zone_hits[zone_index(number)] <= 1:
        score += 0.4

    return score


def back_base_score(number: int, context: dict[str, object]) -> float:
    window = context["window"]
    last = context["last"]
    prev = context["prev"]
    counter_5 = context["counter_5"]
    counter_10 = context["counter_10"]

    miss = omission(window, "back", number)
    score = 0.0
    score += counter_5[number] * 3.0
    score += counter_10[number] * 1.2
    if number in last.back:
        score += 2.8
    if any(abs(number - value) == 1 for value in last.back):
        score += 3.1
    if any(abs(number - value) == 1 for value in prev.back):
        score += 1.0
    score += min(miss, 8) * 0.3
    if 3 <= miss <= 6:
        score += 0.8
    if miss >= 10:
        score -= 0.4
    return score


def front_adaptive_features(number: int, context: dict[str, object]) -> dict[str, float]:
    window = context["window"]
    last = context["last"]
    prev = context["prev"]
    recent_6 = context["recent_6"]
    front_counter_10 = context["front_counter_10"]
    front_counter_20 = context["front_counter_20"]
    front_counter_50 = context["front_counter_50"]
    pair_counter = context["pair_counter"]

    miss = omission(window, "front", number)
    zone_hits = [0, 0, 0]
    for draw in recent_6:
        zone_hits[zone_index(number)] += number in draw.front
    synergy = sum(pair_counter[tuple(sorted((number, mate)))] for mate in last.front if mate != number)

    return {
        "hot10": front_counter_10[number] / 4.0,
        "hot20": front_counter_20[number] / 8.0,
        "hot50": front_counter_50[number] / 14.0,
        "repeat_last": float(number in last.front),
        "neighbor_last": float(any(abs(number - value) == 1 for value in last.front)),
        "neighbor_prev": float(any(abs(number - value) == 1 for value in prev.front)),
        "jump2_last": float(any(abs(number - value) == 2 for value in last.front)),
        "omission_step": min(miss, 12) / 12.0,
        "omission_mid": float(4 <= miss <= 9),
        "omission_warm": float(10 <= miss <= 16),
        "omission_deep": float(miss >= 20),
        "pair_synergy": min(synergy / 18.0, 1.5),
        "zone_cold": float(zone_hits[zone_index(number)] <= 1),
    }


def back_adaptive_features(number: int, context: dict[str, object]) -> dict[str, float]:
    window = context["window"]
    last = context["last"]
    prev = context["prev"]
    counter_5 = context["counter_5"]
    counter_10 = context["counter_10"]
    miss = omission(window, "back", number)
    return {
        "hot5": counter_5[number] / 3.0,
        "hot10": counter_10[number] / 5.0,
        "repeat_last": float(number in last.back),
        "neighbor_last": float(any(abs(number - value) == 1 for value in last.back)),
        "neighbor_prev": float(any(abs(number - value) == 1 for value in prev.back)),
        "omission_step": min(miss, 8) / 8.0,
        "omission_mid": float(3 <= miss <= 6),
        "omission_deep": float(miss >= 10),
    }


def weighted_bonus(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def score_front_numbers(window: list[Draw], state: StrategyState) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    context = build_front_context(window)
    scores: dict[int, float] = {}
    feature_table: dict[int, dict[str, float]] = {}
    for number in FRONT_RANGE:
        features = front_adaptive_features(number, context)
        feature_table[number] = features
        scores[number] = front_base_score(number, context) + weighted_bonus(state.front_bonus_weights, features)
    return scores, feature_table


def score_back_numbers(window: list[Draw], state: StrategyState) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    context = build_back_context(window)
    scores: dict[int, float] = {}
    feature_table: dict[int, dict[str, float]] = {}
    for number in BACK_RANGE:
        features = back_adaptive_features(number, context)
        feature_table[number] = features
        scores[number] = back_base_score(number, context) + weighted_bonus(state.back_bonus_weights, features)
    return scores, feature_table


def sort_candidates(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda number: (-scores[number], number))


def infer_target_zone(window: list[Draw], *, allow_break_zone: bool) -> tuple[int, int, int]:
    recent_6 = window[-6:]
    signatures = [zone_signature(draw.front) for draw in recent_6]
    zero_counts = [sum(signature[idx] == 0 for signature in signatures) for idx in range(3)]
    last_signature = signatures[-1]
    if allow_break_zone or sum(last_signature[idx] >= 4 for idx in range(3)) or max(zero_counts) >= 2:
        dominant = max(range(3), key=zero_counts.__getitem__)
        template = [1, 2, 2]
        if dominant == 0:
            template = [0, 2, 3]
        elif dominant == 1:
            template = [2, 0, 3]
        else:
            template = [2, 3, 0]
        return tuple(template)
    return 1, 2, 2


def classify_window_state(window: list[Draw]) -> dict[str, object]:
    recent_6 = window[-6:]
    recent_3 = window[-3:]
    last = window[-1]
    prev = window[-2]
    signatures = [zone_signature(draw.front) for draw in recent_6]
    last_signature = signatures[-1]
    prev_signature = signatures[-2]

    extreme_recent = sum(int(any(count == 0 for count in signature) or max(signature) >= 4) for signature in signatures)
    zero_counts = [sum(signature[idx] == 0 for signature in signatures) for idx in range(3)]
    dominant_zero_zone = max(range(3), key=zero_counts.__getitem__)
    dominant_zone = max(range(3), key=lambda idx: last_signature[idx])

    if any(count == 0 for count in last_signature) or max(last_signature) >= 4 or extreme_recent >= 2:
        front_state = "extreme"
    elif (any(count == 0 for count in prev_signature) or max(prev_signature) >= 4) and last_signature in {(2, 1, 2), (1, 2, 2), (2, 2, 1)}:
        front_state = "rebound"
    elif sum(zone_signature(draw.front)[1] >= 3 for draw in recent_3) >= 2:
        front_state = "mid-heavy"
    else:
        front_state = "steady"

    back_recent_5 = window[-5:]
    back_counter_5 = Counter(number for draw in back_recent_5 for number in draw.back)
    omission_values = {number: omission(window, "back", number) for number in BACK_RANGE}
    active_back = max(back_counter_5.values()) >= 3 or len(set(last.back) & set(prev.back)) >= 1
    replenish_back = sum(3 <= omission_values[number] <= 6 for number in BACK_RANGE) >= 4
    if active_back:
        back_state = "active"
    elif replenish_back:
        back_state = "replenish"
    else:
        back_state = "mixed"

    return {
        "front_state": front_state,
        "back_state": back_state,
        "last_signature": last_signature,
        "prev_signature": prev_signature,
        "dominant_zero_zone": dominant_zero_zone,
        "dominant_zone": dominant_zone,
        "extreme_recent": extreme_recent,
    }


def target_zone_from_state(state_info: dict[str, object], mode: str) -> tuple[int, int, int]:
    dominant_zero_zone = state_info["dominant_zero_zone"]
    dominant_zone = state_info["dominant_zone"]

    if mode == "extreme":
        if dominant_zero_zone == 0:
            return 0, 2, 3
        if dominant_zero_zone == 1:
            return 2, 0, 3
        return 2, 3, 0

    if mode == "rebound":
        if dominant_zone == 0:
            return 1, 2, 2
        if dominant_zone == 1:
            return 2, 1, 2
        return 2, 2, 1

    return 1, 2, 2


def choose_template_names(state_info: dict[str, object], state: StrategyState, count: int) -> list[str]:
    front_state = state_info["front_state"]
    back_state = state_info["back_state"]

    if front_state == "extreme":
        selected = ["断区反抽票", "极端延续票", "主承接票"]
        priority = ["逆向跳号票", "骨架回补票", "后区活跃票", "后区回补票", "结构平衡票", "重号集中票", "中区加压票", "回稳反抽票"]
    elif front_state == "rebound":
        selected = ["结构平衡票", "回稳反抽票", "主承接票"]
        priority = ["骨架回补票", "逆向跳号票", "后区回补票", "后区活跃票", "中区加压票", "重号集中票", "断区反抽票", "极端延续票"]
    elif front_state == "mid-heavy":
        selected = ["中区加压票", "结构平衡票", "主承接票"]
        priority = ["骨架回补票", "后区活跃票", "逆向跳号票", "后区回补票", "重号集中票", "回稳反抽票", "断区反抽票", "极端延续票"]
    else:
        selected = ["结构平衡票", "主承接票", "逆向跳号票"]
        priority = ["骨架回补票", "重号集中票", "后区活跃票", "后区回补票", "回稳反抽票", "中区加压票", "断区反抽票", "极端延续票"]

    if back_state == "active":
        priority = sorted(priority, key=lambda name: (name not in {"后区活跃票", "主承接票"}, -state.template_scores[name], TEMPLATE_NAMES.index(name)))
    elif back_state == "replenish":
        priority = sorted(priority, key=lambda name: (name not in {"后区回补票", "骨架回补票"}, -state.template_scores[name], TEMPLATE_NAMES.index(name)))
    else:
        priority = sorted(priority, key=lambda name: (-state.template_scores[name], TEMPLATE_NAMES.index(name)))

    seen = set()
    final_names: list[str] = []
    for name in selected + priority:
        if name in seen:
            continue
        seen.add(name)
        final_names.append(name)
        if len(final_names) >= count:
            break
    return final_names


def unique_pairs(pairs: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    result: list[tuple[int, ...]] = []
    for pair in pairs:
        normalized = tuple(sorted(pair))
        if len(normalized) != 2 or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def back_pair_overlap(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return len(set(left) & set(right))


def score_back_pair(
    pair: tuple[int, ...],
    back_ranked: list[int],
    state_info: dict[str, object],
    last_back: tuple[int, ...],
    hot_back: list[int],
    warm_back: list[int],
    back_neighbors: list[int],
) -> float:
    rank_map = {number: idx for idx, number in enumerate(back_ranked)}
    back_state = state_info["back_state"]

    score = 0.0
    for number in pair:
        score += max(0, 12 - rank_map.get(number, 20))

    repeat_hits = sum(number in last_back for number in pair)
    neighbor_hits = sum(number in back_neighbors for number in pair)
    warm_hits = sum(number in warm_back[:5] for number in pair)
    hot_hits = sum(number in hot_back[:5] for number in pair)

    if back_state == "active":
        score += repeat_hits * 3.0
        score += neighbor_hits * 1.8
        score += hot_hits * 1.2
        if pair == tuple(sorted(last_back)):
            score -= 0.8
    elif back_state == "replenish":
        score += warm_hits * 2.5
        score += neighbor_hits * 1.5
        score += hot_hits * 0.8
        score += repeat_hits * 0.5
    else:
        score += hot_hits * 1.6
        score += warm_hits * 1.2
        score += neighbor_hits * 1.3
        score += repeat_hits * 0.8

    if abs(pair[0] - pair[1]) == 1:
        score += 0.3
    return score


def select_back_pairs(
    candidate_pairs: list[tuple[int, ...]],
    back_ranked: list[int],
    state_info: dict[str, object],
    last_back: tuple[int, ...],
    hot_back: list[int],
    warm_back: list[int],
    back_neighbors: list[int],
    count: int,
) -> list[tuple[int, ...]]:
    counts = Counter()
    selected: list[tuple[int, ...]] = []

    while len(selected) < count and candidate_pairs:
        best_pair = None
        best_score = None
        for pair in candidate_pairs:
            score = score_back_pair(pair, back_ranked, state_info, last_back, hot_back, warm_back, back_neighbors)
            for number in pair:
                if counts[number] == 0:
                    score += 1.8
                elif counts[number] == 1:
                    score += 0.5
                else:
                    score -= 1.0
            score -= sum(0.35 * back_pair_overlap(pair, existing) for existing in selected)
            if best_score is None or score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            break
        selected.append(best_pair)
        counts.update(best_pair)
        candidate_pairs = [pair for pair in candidate_pairs if pair != best_pair]

    return selected


def build_back_pair_pool(
    back_ranked: list[int],
    state_info: dict[str, object],
    last_back: tuple[int, ...],
    hot_back: list[int],
    warm_back: list[int],
    back_neighbors: list[int],
) -> list[tuple[int, ...]]:
    back_state = state_info["back_state"]
    if back_state == "active":
        pool = list(dict.fromkeys(list(last_back) + back_neighbors + hot_back[:5] + warm_back[:3] + back_ranked[:6]))
    elif back_state == "replenish":
        pool = list(dict.fromkeys(warm_back[:6] + back_neighbors + hot_back[:4] + list(last_back) + back_ranked[:6]))
    else:
        pool = list(dict.fromkeys(hot_back[:5] + back_neighbors + warm_back[:4] + list(last_back) + back_ranked[:6]))

    pool = pool[:10]
    candidate_pairs = [tuple(sorted(pair)) for pair in itertools.combinations(pool, 2)]
    candidate_pairs.extend(
        [
            choose_back_numbers(back_ranked),
            choose_back_numbers(back_ranked, include=list(last_back)),
            choose_back_numbers(back_ranked, include=hot_back[:1] + back_neighbors[:1]),
            choose_back_numbers(back_ranked, include=warm_back[:1] + back_neighbors[:1]),
            choose_back_numbers(back_ranked, include=back_neighbors[:2]),
            choose_back_numbers(back_ranked, include=list(last_back[:1]) + warm_back[:1]),
        ]
    )
    candidate_pairs = unique_pairs(candidate_pairs)
    return select_back_pairs(candidate_pairs, back_ranked, state_info, last_back, hot_back, warm_back, back_neighbors, count=5)


def build_back_pair_routes(
    back_ranked: list[int],
    state_info: dict[str, object],
    last_back: tuple[int, ...],
    hot_back: list[int],
    warm_back: list[int],
    back_neighbors: list[int],
) -> list[tuple[int, ...]]:
    """Return diversified 5-pair routes for 5 tickets.

    Route design: active continuation / neighbor carry / warm-replenish / reverse jump / balanced fallback.
    """
    routes: list[tuple[int, ...]] = []
    routes.append(choose_back_numbers(back_ranked, include=list(last_back[:1]) + back_neighbors[:1]))
    routes.append(choose_back_numbers(back_ranked, include=back_neighbors[:2]))
    routes.append(choose_back_numbers(back_ranked, include=warm_back[:1] + hot_back[:1]))
    routes.append(choose_back_numbers(back_ranked, include=warm_back[:1], reverse_jump=True, reverse_start_rank=4))
    routes.append(choose_back_numbers(back_ranked, include=hot_back[:1] + list(last_back[:1])))

    pool = build_back_pair_pool(back_ranked, state_info, last_back, hot_back, warm_back, back_neighbors)
    for pair in pool:
        if len(routes) >= 5:
            break
        routes.append(pair)

    return unique_pairs(routes)[:5]


def front_overlap(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return len(set(left) & set(right))


def select_core_numbers(
    front_ranked: list[int],
    feature_table: dict[int, dict[str, float]],
    state_info: dict[str, object],
) -> tuple[list[int], list[int]]:
    rank_map = {number: idx for idx, number in enumerate(front_ranked)}
    front_state = state_info["front_state"]
    dominant_zone = state_info["dominant_zone"]

    def score(number: int) -> float:
        features = feature_table[number]
        value = 28 - rank_map[number]
        value += features["repeat_last"] * 8.0
        value += features["neighbor_last"] * 7.0
        value += features["hot10"] * 6.0
        value += features["hot20"] * 3.0
        value += features["pair_synergy"] * 2.5
        value += features["omission_mid"] * 1.8
        if front_state == "extreme" and zone_index(number) == dominant_zone:
            value += 3.2
        if front_state == "mid-heavy" and 13 <= number <= 24:
            value += 2.4
        return value

    ordered = sorted(front_ranked[:18], key=lambda number: (-score(number), number))
    core: list[int] = []
    for number in ordered:
        if len(core) >= 3:
            break
        if len(core) < 2:
            core.append(number)
            continue
        if zone_index(number) != zone_index(core[0]) or front_state == "extreme":
            core.append(number)
    if len(core) < 3:
        for number in ordered:
            if number not in core:
                core.append(number)
            if len(core) == 3:
                break

    support = [number for number in ordered if number not in core][:5]
    return core[:3], support


def mutate_front_to_overlap(
    front: tuple[int, ...],
    reference: tuple[int, ...],
    ranked: list[int],
    *,
    min_overlap: int,
    max_overlap: int,
    required: Iterable[int] = (),
    forbidden: Iterable[int] = (),
) -> tuple[int, ...]:
    selected = list(front)
    required_set = set(required)
    forbidden_set = set(forbidden)
    ref_set = set(reference)

    while front_overlap(tuple(selected), reference) < min_overlap:
        missing = [number for number in reference if number not in selected and number not in forbidden_set]
        if not missing:
            break
        incoming = missing[0]
        outgoing = next((number for number in reversed(selected) if number not in required_set and number not in ref_set), None)
        if outgoing is None:
            outgoing = next((number for number in reversed(selected) if number not in required_set), None)
        if outgoing is None:
            break
        selected.remove(outgoing)
        selected.append(incoming)
        selected = sorted(set(selected))
        if len(selected) > 5:
            selected = selected[:5]

    while front_overlap(tuple(selected), reference) > max_overlap:
        outgoing = next((number for number in reversed(selected) if number not in required_set and number in ref_set), None)
        if outgoing is None:
            break
        incoming = next((number for number in ranked if number not in selected and number not in ref_set and number not in forbidden_set), None)
        if incoming is None:
            break
        selected.remove(outgoing)
        selected.append(incoming)
        selected = sorted(selected)

    return tuple(sorted(selected))


def reshape_ticket_fronts(
    tickets: list[Ticket],
    selected_names: list[str],
    front_ranked: list[int],
    feature_table: dict[int, dict[str, float]],
    window: list[Draw],
    state_info: dict[str, object],
) -> list[Ticket]:
    if len(tickets) < 3:
        return tickets

    core, support = select_core_numbers(front_ranked, feature_table, state_info)
    a_layer = core[:2]
    main_target = zone_signature(tickets[0].front)
    main_front = choose_front_numbers(front_ranked, window, include=core, target_zone=main_target, allow_break_zone=state_info["front_state"] == "extreme")
    updated: list[Ticket] = [Ticket(tickets[0].name, main_front, tickets[0].back, tickets[0].note + "；核心组集中")]

    mirror_target = zone_signature(tickets[1].front)
    mirror_seed = list(a_layer) + support[:1]
    mirror_front = choose_front_numbers(front_ranked, window, include=mirror_seed, target_zone=mirror_target, allow_break_zone=state_info["front_state"] == "extreme")
    mirror_front = mutate_front_to_overlap(mirror_front, main_front, front_ranked, min_overlap=3, max_overlap=3, required=a_layer)
    updated.append(Ticket(tickets[1].name, mirror_front, tickets[1].back, tickets[1].note + "；主线镜像覆盖"))

    for idx in range(2, len(tickets) - 1):
        ticket = tickets[idx]
        include = []
        if idx == 2:
            include = [core[0], support[0]]
        elif idx == 3:
            include = [support[1], support[2]] if len(support) >= 3 else support[:2]
        front = choose_front_numbers(
            front_ranked,
            window,
            include=include,
            target_zone=zone_signature(ticket.front),
            allow_break_zone="断区" in ticket.name or state_info["front_state"] == "extreme",
            reverse_jump="逆向" in ticket.name,
            reverse_start_rank=10,
        )
        front = mutate_front_to_overlap(front, main_front, front_ranked, min_overlap=1, max_overlap=2, required=include)
        updated.append(Ticket(ticket.name, front, ticket.back, ticket.note + "；结构切换覆盖"))

    detached = tickets[-1]
    detached_seed = support[1:3] if len(support) >= 3 else support[:2]
    detached_front = choose_front_numbers(
        front_ranked,
        window,
        include=detached_seed,
        exclude=a_layer,
        target_zone=zone_signature(detached.front),
        allow_break_zone=False,
        reverse_jump=True,
        reverse_start_rank=12,
    )
    detached_front = mutate_front_to_overlap(detached_front, main_front, front_ranked, min_overlap=0, max_overlap=1, forbidden=a_layer)
    updated.append(Ticket(detached.name, detached_front, detached.back, detached.note + "；脱核防分散"))

    counts = Counter(number for ticket in updated for number in ticket.front)
    for core_number in a_layer:
        if counts[core_number] >= 2:
            continue
        for idx in range(2, len(updated) - 1):
            ticket = updated[idx]
            if core_number in ticket.front:
                continue
            front = choose_front_numbers(front_ranked, window, include=[core_number], target_zone=zone_signature(ticket.front))
            front = mutate_front_to_overlap(front, main_front, front_ranked, min_overlap=1, max_overlap=2, required=[core_number])
            updated[idx] = Ticket(ticket.name, front, ticket.back, ticket.note + "；A层补强")
            counts = Counter(number for row in updated for number in row.front)
            break

    final_tickets: list[Ticket] = []
    for ticket in updated:
        final_tickets.append(distinct_ticket(ticket, final_tickets, window, front_ranked))
    return final_tickets


def choose_front_numbers(
    ranked: list[int],
    window: list[Draw],
    *,
    include: Iterable[int] = (),
    exclude: Iterable[int] = (),
    target_zone: tuple[int, int, int] | None = None,
    allow_break_zone: bool = False,
    reverse_jump: bool = False,
    reverse_start_rank: int = 8,
) -> tuple[int, ...]:
    selected: list[int] = []
    excluded = set(exclude)
    rank_map = {number: idx for idx, number in enumerate(ranked)}
    target_zone = target_zone or infer_target_zone(window, allow_break_zone=allow_break_zone)
    zone_counts = [0, 0, 0]

    for number in include:
        if number in excluded or number in selected:
            continue
        selected.append(number)
        zone_counts[zone_index(number)] += 1

    for number in ranked:
        if number in excluded or number in selected:
            continue
        if reverse_jump and rank_map[number] < reverse_start_rank:
            continue
        zone = zone_index(number)
        if not allow_break_zone and zone_counts[zone] >= target_zone[zone]:
            continue
        selected.append(number)
        zone_counts[zone] += 1
        if len(selected) == 5:
            break

    if len(selected) < 5:
        for number in ranked:
            if number in excluded or number in selected:
                continue
            selected.append(number)
            if len(selected) == 5:
                break

    return tuple(sorted(selected[:5]))


def choose_back_numbers(
    ranked: list[int],
    *,
    include: Iterable[int] = (),
    exclude: Iterable[int] = (),
    reverse_jump: bool = False,
    reverse_start_rank: int = 3,
) -> tuple[int, ...]:
    selected: list[int] = []
    excluded = set(exclude)
    rank_map = {number: idx for idx, number in enumerate(ranked)}
    for number in include:
        if number not in excluded and number not in selected:
            selected.append(number)
    for number in ranked:
        if number in excluded or number in selected:
            continue
        if reverse_jump and rank_map[number] < reverse_start_rank:
            continue
        selected.append(number)
        if len(selected) == 2:
            break
    return tuple(sorted(selected[:2]))


def exact_front_seen(window: list[Draw], front: tuple[int, ...]) -> bool:
    return any(draw.front == front for draw in window)


def distinct_ticket(candidate: Ticket, existing: list[Ticket], window: list[Draw], ranked: list[int]) -> Ticket:
    if not exact_front_seen(window, candidate.front) and all(ticket.front != candidate.front for ticket in existing):
        return candidate

    base_front = list(candidate.front)
    for replacement in ranked:
        if replacement in base_front:
            continue
        mutated = base_front[:]
        mutated[-1] = replacement
        mutated_tuple = tuple(sorted(mutated))
        if not exact_front_seen(window, mutated_tuple) and all(ticket.front != mutated_tuple for ticket in existing):
            return Ticket(candidate.name, mutated_tuple, candidate.back, candidate.note + "（去重修正）")
    return candidate


def generate_tickets(window: list[Draw], state: StrategyState, count: int = DEFAULT_TICKET_COUNT) -> tuple[list[Ticket], list[int], list[int], dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    front_scores, front_feature_table = score_front_numbers(window, state)
    back_scores, back_feature_table = score_back_numbers(window, state)
    front_ranked = sort_candidates(front_scores)
    back_ranked = sort_candidates(back_scores)

    last = window[-1]
    last_neighbors = sorted(
        {
            neighbor
            for value in last.front
            for neighbor in (value - 1, value + 1)
            if neighbor in FRONT_RANGE
        }
    )
    back_neighbors = sorted(
        {
            neighbor
            for value in last.back
            for neighbor in (value - 1, value + 1)
            if neighbor in BACK_RANGE
        }
    )
    state_info = classify_window_state(window)
    hot_front = [number for number in front_ranked[:12] if number not in last.front]
    hot_back = [number for number in back_ranked[:6] if number not in last.back]
    repeats = [number for number in front_ranked if number in last.front]
    warm_back = [number for number in back_ranked if 3 <= omission(window, "back", number) <= 6]
    cold_front = [number for number in front_ranked[8:20] if number not in last.front]

    template_map: dict[str, Ticket] = {
        "主承接票": Ticket(
            name="主承接票",
            front=choose_front_numbers(front_ranked, window, include=list(last.front[:1]) + last_neighbors[:2]),
            back=choose_back_numbers(back_ranked, include=list(last.back[:1]) + back_neighbors[:1]),
            note="重号+边码主线",
        ),
        "骨架回补票": Ticket(
            name="骨架回补票",
            front=choose_front_numbers(front_ranked, window, include=hot_front[:3]),
            back=choose_back_numbers(back_ranked, include=hot_back[:2]),
            note="热号骨架+温冷回补",
        ),
        "结构平衡票": Ticket(
            name="结构平衡票",
            front=choose_front_numbers(front_ranked, window, target_zone=(2, 1, 2)),
            back=choose_back_numbers(back_ranked),
            note="回稳结构",
        ),
        "断区反抽票": Ticket(
            name="断区反抽票",
            front=choose_front_numbers(front_ranked, window, allow_break_zone=True),
            back=choose_back_numbers(back_ranked, include=back_neighbors[:1]),
            note="强断区信号放行",
        ),
        "逆向跳号票": Ticket(
            name="逆向跳号票",
            front=choose_front_numbers(front_ranked, window, include=hot_front[:2], reverse_jump=True, reverse_start_rank=10),
            back=choose_back_numbers(back_ranked, reverse_jump=True, reverse_start_rank=4),
            note="加入前20外补位号",
        ),
        "重号集中票": Ticket(
            name="重号集中票",
            front=choose_front_numbers(front_ranked, window, include=repeats[:2] + last_neighbors[:1], target_zone=(1, 2, 2)),
            back=choose_back_numbers(back_ranked, include=list(last.back[:1])),
            note="重号延续优先",
        ),
        "中区加压票": Ticket(
            name="中区加压票",
            front=choose_front_numbers(front_ranked, window, include=[number for number in hot_front if 13 <= number <= 24][:3], target_zone=(1, 3, 1)),
            back=choose_back_numbers(back_ranked, include=hot_back[:1] + back_neighbors[:1]),
            note="中区强化与温热配对",
        ),
        "极端延续票": Ticket(
            name="极端延续票",
            front=choose_front_numbers(front_ranked, window, include=hot_front[:2] + repeats[:1], target_zone=target_zone_from_state(state_info, "extreme"), allow_break_zone=True),
            back=choose_back_numbers(back_ranked, include=list(last.back[:1]) + hot_back[:1]),
            note="沿上一期极端结构延续",
        ),
        "回稳反抽票": Ticket(
            name="回稳反抽票",
            front=choose_front_numbers(front_ranked, window, include=last_neighbors[:2], target_zone=target_zone_from_state(state_info, "rebound")),
            back=choose_back_numbers(back_ranked, include=warm_back[:1] + back_neighbors[:1]),
            note="极端后回稳反抽",
        ),
        "后区活跃票": Ticket(
            name="后区活跃票",
            front=choose_front_numbers(front_ranked, window, include=hot_front[:2] + last_neighbors[:1], target_zone=(1, 2, 2)),
            back=choose_back_numbers(back_ranked, include=list(last.back)[:2]),
            note="后区重号/活跃延续",
        ),
        "后区回补票": Ticket(
            name="后区回补票",
            front=choose_front_numbers(front_ranked, window, include=cold_front[:2], target_zone=(2, 1, 2), reverse_jump=True, reverse_start_rank=12),
            back=choose_back_numbers(back_ranked, include=warm_back[:2]),
            note="后区中段遗漏回补",
        ),
    }

    selected_names = choose_template_names(state_info, state, count)
    if count == 5:
        # Force 5-ticket portfolio: 2 mainline + 2 divergence + 1 reverse-cold.
        if state_info["front_state"] == "extreme":
            selected_names = ["主承接票", "断区反抽票", "结构平衡票", "后区回补票", "逆向跳号票"]
        elif state_info["front_state"] == "rebound":
            selected_names = ["主承接票", "结构平衡票", "骨架回补票", "后区活跃票", "逆向跳号票"]
        elif state_info["front_state"] == "mid-heavy":
            selected_names = ["主承接票", "中区加压票", "结构平衡票", "后区回补票", "逆向跳号票"]
        else:
            selected_names = ["主承接票", "结构平衡票", "骨架回补票", "后区活跃票", "逆向跳号票"]

    back_pair_pool = build_back_pair_pool(back_ranked, state_info, last.back, hot_back, warm_back, back_neighbors)
    back_pair_routes = build_back_pair_routes(back_ranked, state_info, last.back, hot_back, warm_back, back_neighbors)
    pair_index = 0
    tickets: list[Ticket] = []
    for name in selected_names[:count]:
        template = template_map[name]
        if count == 5 and len(back_pair_routes) >= 5:
            back_pair = back_pair_routes[len(tickets)]
        elif name == "后区活跃票" and state_info["back_state"] == "active":
            back_pair = back_pair_pool[0]
        elif name == "后区回补票" and state_info["back_state"] == "replenish":
            back_pair = back_pair_pool[0]
        else:
            back_pair = back_pair_pool[pair_index % len(back_pair_pool)]
            pair_index += 1
        template = Ticket(template.name, template.front, back_pair, template.note)
        ticket = distinct_ticket(template, tickets, window, front_ranked)
        tickets.append(ticket)
    tickets = reshape_ticket_fronts(tickets, selected_names, front_ranked, front_feature_table, window, state_info)
    return tickets, front_ranked, back_ranked, front_feature_table, back_feature_table, state_info


def hit_summary(ticket: Ticket, actual: Draw) -> dict[str, int]:
    front_hits = len(set(ticket.front) & set(actual.front))
    back_hits = len(set(ticket.back) & set(actual.back))
    return {"front_hits": front_hits, "back_hits": back_hits}


def actual_feature_flags(window: list[Draw], actual: Draw, front_ranked: list[int], back_ranked: list[int]) -> dict[str, int]:
    last = window[-1]
    prev = window[-2]
    front_top10 = sum(number in front_ranked[:10] for number in actual.front)
    front_top15 = sum(number in front_ranked[:15] for number in actual.front)
    front_repeat = sum(number in last.front for number in actual.front)
    front_neighbor = sum(
        any(abs(number - source) == 1 for source in last.front)
        for number in actual.front
    )
    front_prev_neighbor = sum(
        any(abs(number - source) == 1 for source in prev.front)
        for number in actual.front
    )
    back_top6 = sum(number in back_ranked[:6] for number in actual.back)
    back_repeat = sum(number in last.back for number in actual.back)
    back_neighbor = sum(any(abs(number - source) == 1 for source in last.back) for number in actual.back)
    return {
        "front_top10": front_top10,
        "front_top15": front_top15,
        "front_repeat": front_repeat,
        "front_neighbor": front_neighbor,
        "front_prev_neighbor": front_prev_neighbor,
        "back_top6": back_top6,
        "back_repeat": back_repeat,
        "back_neighbor": back_neighbor,
    }


def update_bonus_weights(
    weights: dict[str, float],
    feature_table: dict[int, dict[str, float]],
    actual_numbers: tuple[int, ...],
    all_numbers: Iterable[int],
    learning_rate: float,
    decay: float,
    limit: float,
) -> None:
    for name in list(weights):
        pool_mean = statistics.mean(feature_table[number][name] for number in all_numbers)
        actual_mean = statistics.mean(feature_table[number][name] for number in actual_numbers)
        weights[name] = max(-limit, min(limit, weights[name] * decay + learning_rate * (actual_mean - pool_mean)))


def update_template_scores(state: StrategyState, tickets: list[Ticket], hits: list[dict[str, int]]) -> None:
    if not state.adaptive:
        return

    for name in list(state.template_scores):
        state.template_scores[name] = max(0.35, min(3.5, 1.0 + (state.template_scores[name] - 1.0) * state.template_decay))

    for ticket, hit in zip(tickets, hits):
        reward = hit["front_hits"] + hit["back_hits"] * state.template_back_hit_weight
        bonus = state.template_front_hit_bonus if hit["front_hits"] >= 2 else 0.0
        bonus += state.template_back_hit_bonus if hit["back_hits"] >= 1 else 0.0
        bonus += state.template_front3_hit_bonus if hit["front_hits"] >= 3 else 0.0
        bonus += state.template_joint_3p1_bonus if hit["front_hits"] >= 3 and hit["back_hits"] >= 1 else 0.0
        delta = state.template_learning_rate * (reward + bonus - state.template_reward_baseline)
        state.template_scores[ticket.name] = max(0.35, min(3.5, state.template_scores[ticket.name] + delta))


def top_weight_items(weights: dict[str, float], count: int = 4) -> list[tuple[str, float]]:
    return sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:count]


def bottom_weight_items(weights: dict[str, float], count: int = 3) -> list[tuple[str, float]]:
    return sorted(weights.items(), key=lambda item: (item[1], item[0]))[:count]


def rolling_backtest(
    draws: list[Draw],
    window_size: int,
    ticket_count: int,
    *,
    adaptive: bool,
    adaptive_config: dict[str, float] | None = None,
    early_stop_config: EarlyStopConfig | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    results: list[dict[str, object]] = []
    aggregate = defaultdict(int)
    aggregate_lists: dict[str, list[float]] = defaultdict(list)
    state_aggregate: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    state = build_state(adaptive=adaptive, **(adaptive_config or {}))
    early_stop = early_stop_config or EarlyStopConfig()
    learning_active = adaptive
    best_state_snapshot = snapshot_state(state)
    best_eval_score = float("-inf")
    stale_evals = 0
    early_stop_triggered = False
    early_stop_trigger_window = None

    for start in range(0, len(draws) - window_size):
        window = draws[start : start + window_size]
        actual = draws[start + window_size]
        tickets, front_ranked, back_ranked, front_feature_table, back_feature_table, state_info = generate_tickets(
            window,
            state,
            count=ticket_count,
        )
        hits = [hit_summary(ticket, actual) for ticket in tickets]
        best = max(hits, key=lambda item: (item["front_hits"], item["back_hits"]))
        flags = actual_feature_flags(window, actual, front_ranked, back_ranked)
        signature = zone_signature(actual.front)
        odd_count = sum(number % 2 for number in actual.front)
        extreme_zone = int(any(count == 0 for count in signature))
        high_stack = int(signature[2] >= 4 or signature[0] >= 4)
        selected_template_scores = {ticket.name: round(state.template_scores[ticket.name], 3) for ticket in tickets}

        aggregate["windows"] += 1
        aggregate["best_front_2plus"] += int(best["front_hits"] >= 2)
        aggregate["best_front_3plus"] += int(best["front_hits"] >= 3)
        aggregate["best_back_1plus"] += int(best["back_hits"] >= 1)
        aggregate["best_back_exact"] += int(best["back_hits"] == 2)
        aggregate["same_ticket_front3_back1"] += int(
            any(hit["front_hits"] >= 3 and hit["back_hits"] >= 1 for hit in hits)
        )
        aggregate["extreme_zone_windows"] += extreme_zone
        aggregate["high_stack_windows"] += high_stack

        for key, value in flags.items():
            aggregate[key] += value

        front_state_key = f"front:{state_info['front_state']}"
        back_state_key = f"back:{state_info['back_state']}"
        for group_key in (front_state_key, back_state_key):
            state_aggregate[group_key]["windows"] += 1
            state_aggregate[group_key]["best_front_2plus"] += int(best["front_hits"] >= 2)
            state_aggregate[group_key]["best_back_1plus"] += int(best["back_hits"] >= 1)

        aggregate_lists["best_front_hits"].append(best["front_hits"])
        aggregate_lists["best_back_hits"].append(best["back_hits"])
        aggregate_lists["front_top15"].append(flags["front_top15"])
        aggregate_lists["front_neighbor"].append(flags["front_neighbor"])
        aggregate_lists["back_top6"].append(flags["back_top6"])

        results.append(
            {
                "train_start": window[0].issue,
                "train_end": window[-1].issue,
                "actual_issue": actual.issue,
                "actual_date": actual.date,
                "actual_front": actual.front,
                "actual_back": actual.back,
                "best_front_hits": best["front_hits"],
                "best_back_hits": best["back_hits"],
                "zone_signature": signature,
                "odd_even": (odd_count, 5 - odd_count),
                "flags": flags,
                "tickets": tickets,
                "hits": hits,
                "template_scores": selected_template_scores,
                "state_info": state_info,
            }
        )

        if learning_active:
            update_bonus_weights(
                state.front_bonus_weights,
                front_feature_table,
                actual.front,
                FRONT_RANGE,
                state.front_learning_rate,
                state.front_weight_decay,
                limit=1.75,
            )
            update_bonus_weights(
                state.back_bonus_weights,
                back_feature_table,
                actual.back,
                BACK_RANGE,
                state.back_learning_rate,
                state.back_weight_decay,
                limit=1.75,
            )
            update_template_scores(state, tickets, hits)

        if adaptive and early_stop.enabled:
            window_count = aggregate["windows"]
            if window_count % max(1, early_stop.eval_interval) == 0 and len(results) >= max(1, early_stop.eval_window):
                recent_rows = results[-early_stop.eval_window :]
                eval_score = statistics.mean(metric_value_for_row(early_stop.metric, row) for row in recent_rows)
                if eval_score > best_eval_score + early_stop.min_delta:
                    best_eval_score = eval_score
                    stale_evals = 0
                    best_state_snapshot = snapshot_state(state)
                else:
                    stale_evals += 1
                    if stale_evals >= early_stop.patience and learning_active:
                        restore_state(state, best_state_snapshot)
                        learning_active = False
                        early_stop_triggered = True
                        early_stop_trigger_window = window_count

    summary = {
        "adaptive": adaptive,
        "windows": aggregate["windows"],
        "front_learning_rate": state.front_learning_rate,
        "back_learning_rate": state.back_learning_rate,
        "template_learning_rate": state.template_learning_rate,
        "best_front_2plus_rate": aggregate["best_front_2plus"] / aggregate["windows"],
        "best_front_3plus_rate": aggregate["best_front_3plus"] / aggregate["windows"],
        "best_back_1plus_rate": aggregate["best_back_1plus"] / aggregate["windows"],
        "best_back_exact_rate": aggregate["best_back_exact"] / aggregate["windows"],
        "same_ticket_front3_back1_rate": aggregate["same_ticket_front3_back1"] / aggregate["windows"],
        "extreme_zone_rate": aggregate["extreme_zone_windows"] / aggregate["windows"],
        "high_stack_rate": aggregate["high_stack_windows"] / aggregate["windows"],
        "avg_best_front_hits": statistics.mean(aggregate_lists["best_front_hits"]),
        "avg_best_back_hits": statistics.mean(aggregate_lists["best_back_hits"]),
        "avg_actual_front_top15": statistics.mean(aggregate_lists["front_top15"]),
        "avg_actual_front_neighbor": statistics.mean(aggregate_lists["front_neighbor"]),
        "avg_actual_back_top6": statistics.mean(aggregate_lists["back_top6"]),
        "top_front_weights": top_weight_items(state.front_bonus_weights),
        "bottom_front_weights": bottom_weight_items(state.front_bonus_weights),
        "top_back_weights": top_weight_items(state.back_bonus_weights),
        "bottom_back_weights": bottom_weight_items(state.back_bonus_weights),
        "top_templates": sorted(state.template_scores.items(), key=lambda item: (-item[1], item[0]))[:5],
        "early_stop": {
            "enabled": early_stop.enabled,
            "metric": early_stop.metric,
            "eval_interval": early_stop.eval_interval,
            "eval_window": early_stop.eval_window,
            "patience": early_stop.patience,
            "min_delta": early_stop.min_delta,
            "triggered": early_stop_triggered,
            "trigger_window": early_stop_trigger_window,
            "best_eval_score": (best_eval_score if best_eval_score != float("-inf") else None),
        },
        "state_breakdown": {
            key: {
                "windows": values["windows"],
                "best_front_2plus_rate": values["best_front_2plus"] / values["windows"],
                "best_back_1plus_rate": values["best_back_1plus"] / values["windows"],
            }
            for key, values in state_aggregate.items()
        },
    }
    return results, summary


def compare_summaries(fixed: dict[str, object], adaptive: dict[str, object]) -> dict[str, float]:
    return {
        "best_front_2plus_delta": adaptive["best_front_2plus_rate"] - fixed["best_front_2plus_rate"],
        "best_front_3plus_delta": adaptive["best_front_3plus_rate"] - fixed["best_front_3plus_rate"],
        "best_back_1plus_delta": adaptive["best_back_1plus_rate"] - fixed["best_back_1plus_rate"],
        "best_back_exact_delta": adaptive["best_back_exact_rate"] - fixed["best_back_exact_rate"],
        "same_ticket_front3_back1_delta": adaptive["same_ticket_front3_back1_rate"] - fixed["same_ticket_front3_back1_rate"],
        "avg_best_front_hits_delta": adaptive["avg_best_front_hits"] - fixed["avg_best_front_hits"],
        "avg_best_back_hits_delta": adaptive["avg_best_back_hits"] - fixed["avg_best_back_hits"],
    }


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
    lines.append("# 大乐透50期逐期自适应回测复盘报告")
    lines.append("")
    lines.append("## 1. 回测设定")
    lines.append(f"- 历史窗口：固定 `{window_size}` 期。")
    lines.append(f"- 每期输出：程序化生成 `{ticket_count}` 注单式预测。")
    lines.append("- 执行方式：先用固定规则跑完整历史，再用逐期在线修订规则跑同一批历史，比较两者差异。")
    lines.append("- 自适应逻辑：每预测1期，先识别前区状态并动态分配5注模板，再根据真实开奖号码更新后区特征权重和票型优先级，下一期直接使用已更新后的规则。")
    lines.append("")
    lines.append("## 2. 固定规则 vs 逐期自适应")
    lines.append("| 指标 | 固定规则 | 逐期自适应 | 差值 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 5注至少1注命中前区2码 | `{fixed_summary['best_front_2plus_rate']:.2%}` | `{adaptive_summary['best_front_2plus_rate']:.2%}` | `{comparison['best_front_2plus_delta']:+.2%}` |")
    lines.append(f"| 5注至少1注命中前区3码及以上 | `{fixed_summary['best_front_3plus_rate']:.2%}` | `{adaptive_summary['best_front_3plus_rate']:.2%}` | `{comparison['best_front_3plus_delta']:+.2%}` |")
    lines.append(f"| 5注至少1注命中后区1码 | `{fixed_summary['best_back_1plus_rate']:.2%}` | `{adaptive_summary['best_back_1plus_rate']:.2%}` | `{comparison['best_back_1plus_delta']:+.2%}` |")
    lines.append(f"| 5注后区2码全中 | `{fixed_summary['best_back_exact_rate']:.2%}` | `{adaptive_summary['best_back_exact_rate']:.2%}` | `{comparison['best_back_exact_delta']:+.2%}` |")
    lines.append(f"| 同一注同时命中前区3码+后区1码 | `{fixed_summary['same_ticket_front3_back1_rate']:.2%}` | `{adaptive_summary['same_ticket_front3_back1_rate']:.2%}` | `{comparison['same_ticket_front3_back1_delta']:+.2%}` |")
    lines.append(f"| 最优单注平均前区命中 | `{fixed_summary['avg_best_front_hits']:.2f}` | `{adaptive_summary['avg_best_front_hits']:.2f}` | `{comparison['avg_best_front_hits_delta']:+.2f}` |")
    lines.append(f"| 最优单注平均后区命中 | `{fixed_summary['avg_best_back_hits']:.2f}` | `{adaptive_summary['avg_best_back_hits']:.2f}` | `{comparison['avg_best_back_hits_delta']:+.2f}` |")
    lines.append("")
    lines.append("## 3. 自适应最终学到的规则偏好")
    if adaptive_summary["front_learning_rate"] > 0:
        lines.append(f"- 前区正向增强特征：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['top_front_weights'])}`")
        lines.append(f"- 前区负向抑制特征：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['bottom_front_weights'])}`")
    else:
        lines.append("- 前区号码层未启用在线权重学习，本轮前区主要通过票型优先级调整来避免把固定规则拉坏。")
    if adaptive_summary["back_learning_rate"] > 0:
        lines.append(f"- 后区正向增强特征：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['top_back_weights'])}`")
        lines.append(f"- 后区负向抑制特征：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['bottom_back_weights'])}`")
    else:
        lines.append("- 后区号码层未启用在线权重学习。")
    lines.append(f"- 最终高分票型：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['top_templates'])}`")
    lines.append("")
    lines.append("## 4. 分状态命中表现")
    lines.append("| 状态 | 期数 | 前区2码命中率 | 后区1码命中率 |")
    lines.append("| --- | --- | --- | --- |")
    for key in sorted(adaptive_summary["state_breakdown"]):
        item = adaptive_summary["state_breakdown"][key]
        lines.append(f"| `{key}` | `{item['windows']}` | `{item['best_front_2plus_rate']:.2%}` | `{item['best_back_1plus_rate']:.2%}` |")
    lines.append("")
    lines.append("## 5. 程序归纳出的修正规则")
    lines.append("- 最近1期承接层不再只是固定顺序，而是以“主承接票”形式参与动态竞争；当这一路线持续命中时，会自动保留在高优先级票型里。")
    lines.append("- 断区票、逆向跳号票不再机械固定5注同配，而是根据 `extreme / rebound / steady / mid-heavy` 状态动态分配。")
    lines.append("- 候选池命中通常高于最终出票命中，因此收口阶段改为“票型竞争 + 动态优先级”，不再固定死顺序。")
    lines.append("- 后区不再机械重复押同一 exact pair，而是根据活跃、邻接、遗漏特征的实时表现调整。")
    lines.append("")
    lines.append("## 6. 最近20期自适应滚动样本摘录")
    lines.append("")
    lines.append("| 训练区间 | 预测期 | 实际开奖 | 最优命中 | 区间结构 | 说明 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in adaptive_results[-20:]:
        best = f"{row['best_front_hits']}前+{row['best_back_hits']}后"
        actual = f"{format_numbers(row['actual_front'])} + {format_numbers(row['actual_back'])}"
        zone_text = ":".join(str(value) for value in row["zone_signature"])
        flags = row["flags"]
        state_note = f"{row['state_info']['front_state']}/{row['state_info']['back_state']}"
        note = f"{state_note}；前15入围{flags['front_top15']}个，前区邻接{flags['front_neighbor']}个，后区前6入围{flags['back_top6']}个"
        lines.append(
            f"| {row['train_start']}-{row['train_end']} | {row['actual_issue']} | `{actual}` | `{best}` | `{zone_text}` | {note} |"
        )
    lines.append("")
    lines.append("## 7. 最近3期自适应详细样例")
    for row in adaptive_results[-3:]:
        lines.append("")
        lines.append(f"### {row['actual_issue']} 期")
        lines.append(f"- 训练区间：`{row['train_start']}~{row['train_end']}`")
        lines.append(f"- 实际开奖：前区 `{format_numbers(row['actual_front'])}`，后区 `{format_numbers(row['actual_back'])}`")
        lines.append(f"- 区间结构：`{':'.join(str(value) for value in row['zone_signature'])}`")
        lines.append(f"- 状态识别：前区 `{row['state_info']['front_state']}`，后区 `{row['state_info']['back_state']}`")
        lines.append(f"- 最优命中：`{row['best_front_hits']}前 + {row['best_back_hits']}后`")
        lines.append(f"- 当期入选票型分值：`{', '.join(f'{name}={score:.2f}' for name, score in row['template_scores'].items())}`")
        for ticket, hit in zip(row["tickets"], row["hits"]):
            lines.append(
                f"- {ticket.name}：`{format_numbers(ticket.front)} + {format_numbers(ticket.back)}` -> `{hit['front_hits']}前 + {hit['back_hits']}后`；{ticket.note}"
            )
    lines.append("")
    return "\n".join(lines)


def build_rule_summary(fixed_summary: dict[str, object], adaptive_summary: dict[str, object], comparison: dict[str, float]) -> str:
    if adaptive_summary["front_learning_rate"] > 0:
        front_rule_line = f"- 自适应最终增强的前区特征主要是：`{', '.join(name for name, _ in adaptive_summary['top_front_weights'])}`；说明这些信号在逐期复盘中持续被证实，应优先保留在主线。"
    else:
        front_rule_line = "- 当前默认自适应版本没有直接改写前区号码层权重，而是只让票型优先级在线学习；这样做是因为前区一旦学习步长过大，历史回测会立刻劣化。"

    return "\n".join(
        [
            "## 3.1 基于50期滚动回测的逐期在线修订",
            "- 新增一套严格在线回测流程：任取最近 `50` 期作为训练集，生成 `5` 注候选票，只与下一期对比；复盘后立即更新规则参数，再用更新后的规则去预测后一期。",
            f"- 当前全量历史样本共滚动 `{adaptive_summary['windows']}` 次；固定规则下，5注至少命中前区 `2` 码的比例为 `{fixed_summary['best_front_2plus_rate']:.2%}`，逐期自适应后为 `{adaptive_summary['best_front_2plus_rate']:.2%}`，变化 `{comparison['best_front_2plus_delta']:+.2%}`。",
            f"- 固定规则下，5注至少命中后区 `1` 码的比例为 `{fixed_summary['best_back_1plus_rate']:.2%}`，逐期自适应后为 `{adaptive_summary['best_back_1plus_rate']:.2%}`，变化 `{comparison['best_back_1plus_delta']:+.2%}`。",
            front_rule_line,
            f"- 自适应最终增强的后区特征主要是：`{', '.join(name for name, _ in adaptive_summary['top_back_weights'])}`；说明后区应继续围绕活跃、邻接与遗漏步进信号构建。",
            f"- 自适应最终高分票型为：`{', '.join(name for name, _ in adaptive_summary['top_templates'])}`；后续实战出票时，这些票型应优先保留。",
            "- 新增执行要求：每完成1期复盘，必须先识别当前属于 `extreme / rebound / steady / mid-heavy` 哪种前区状态，再按状态动态分配5注模板；随后才刷新后区权重和票型优先级。",
            "- 新增复盘要求：除统计命中数外，必须记录权重上调了哪些特征、哪些票型被降权，以及是否因为收口阶段压错票型而丢失候选池内的实际号码。",
            "- 详细在线滚动结果单独维护在 `docs/大乐透50期逐期自适应回测复盘报告.md`，规则文档只保留长期稳定结论。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="大乐透50期逐期自适应回测")
    parser.add_argument(
        "--history",
        default="docs/大乐透历史开奖号码.md",
        help="历史开奖 Markdown 文件路径",
    )
    parser.add_argument(
        "--report",
        default="docs/大乐透50期逐期自适应回测复盘报告.md",
        help="回测报告输出路径",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="滚动训练窗口大小",
    )
    parser.add_argument(
        "--ticket-count",
        type=int,
        default=DEFAULT_TICKET_COUNT,
        help="每期输出的预测注数",
    )
    parser.add_argument(
        "--print-rule-summary",
        action="store_true",
        help="打印适合写入规则文档的摘要",
    )
    parser.add_argument("--early-stop", action="store_true", help="启用自动早停学习器")
    parser.add_argument(
        "--early-stop-metric",
        default="same_ticket_front3_back1",
        choices=["same_ticket_front3_back1", "best_front_2plus", "best_front_3plus", "best_back_1plus", "best_back_exact"],
        help="早停监控指标",
    )
    parser.add_argument("--early-stop-eval-interval", type=int, default=50, help="每隔多少滚动窗口评估一次早停")
    parser.add_argument("--early-stop-eval-window", type=int, default=200, help="每次评估使用最近多少窗口")
    parser.add_argument("--early-stop-patience", type=int, default=3, help="连续多少次无提升后触发早停")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005, help="视为提升的最小阈值")
    args = parser.parse_args()

    draws = parse_history(Path(args.history))
    if len(draws) <= args.window_size:
        raise SystemExit("历史数据不足，无法完成滚动回测。")

    fixed_results, fixed_summary = rolling_backtest(draws, args.window_size, args.ticket_count, adaptive=False)
    adaptive_results, adaptive_summary = rolling_backtest(
        draws,
        args.window_size,
        args.ticket_count,
        adaptive=True,
        early_stop_config=EarlyStopConfig(
            enabled=args.early_stop,
            metric=args.early_stop_metric,
            eval_interval=args.early_stop_eval_interval,
            eval_window=args.early_stop_eval_window,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
        ),
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
    print(f"滚动窗口：{adaptive_summary['windows']}")
    print(f"固定规则 前区2码命中率：{fixed_summary['best_front_2plus_rate']:.2%}")
    print(f"自适应规则 前区2码命中率：{adaptive_summary['best_front_2plus_rate']:.2%}")
    print(f"固定规则 后区1码命中率：{fixed_summary['best_back_1plus_rate']:.2%}")
    print(f"自适应规则 后区1码命中率：{adaptive_summary['best_back_1plus_rate']:.2%}")
    if args.early_stop:
        early_stop_info = adaptive_summary["early_stop"]
        print(
            "早停状态："
            f"触发={early_stop_info['triggered']}，"
            f"触发窗口={early_stop_info['trigger_window']}，"
            f"最佳评估分={early_stop_info['best_eval_score']}"
        )
    if args.print_rule_summary:
        print("")
        print(build_rule_summary(fixed_summary, adaptive_summary, comparison))


if __name__ == "__main__":
    main()
