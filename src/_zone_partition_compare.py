"""Compare 3/5/7 front zone partitions: break patterns and rolling hit rates."""
from __future__ import annotations

import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlt_rolling_backtest as m

HISTORY = Path(__file__).resolve().parents[1] / "docs" / "大乐透历史开奖号码.md"
WINDOW = 50


def parse_draws() -> list[m.Draw]:
    return m.parse_history(HISTORY)


def zone_index_3(n: int) -> int:
    return m.zone_index(n)


def zone_index_5(n: int) -> int:
    if n <= 7:
        return 0
    if n <= 14:
        return 1
    if n <= 21:
        return 2
    if n <= 28:
        return 3
    return 4


def zone_index_7(n: int) -> int:
    return (n - 1) // 5


PARTITIONS: dict[str, tuple[int, callable]] = {
    "3区(01-12/13-24/25-35)": (3, zone_index_3),
    "5区(每7码)": (5, zone_index_5),
    "7区(每5码)": (7, zone_index_7),
}


def signature(front: tuple[int, ...], zfn) -> tuple[int, ...]:
    counts = [0] * 7
    nz = max(zfn(35), zfn(1)) + 1
    counts = [0] * nz
    for n in front:
        counts[zfn(n)] += 1
    return tuple(counts)


def empty_zones(sig: tuple[int, ...]) -> list[int]:
    return [i for i, c in enumerate(sig) if c == 0]


def predict_break_zones(recent_sigs: list[tuple[int, ...]], n_zones: int, *, mode: str) -> set[int]:
    """mode=rotation: zone with longest streak without being empty.
    mode=hot_break: zone with highest empty count in recent window (most often breaks).
    mode=single_dominant: one zone most likely empty (argmax zero_counts)."""
    zero_counts = [0] * n_zones
    for sig in recent_sigs:
        for i in range(n_zones):
            if sig[i] == 0:
                zero_counts[i] += 1

    if mode == "hot_break":
        # predict up to floor(n_zones/3) zones that break most often in recent_6
        k = max(1, n_zones // 3)
        ordered = sorted(range(n_zones), key=lambda i: (-zero_counts[i], i))
        return set(ordered[:k])

    if mode == "rotation":
        # zone that has NOT been empty longest in recent window (predict it breaks next)
        last_empty_at = {}
        for i in range(n_zones):
            last_empty_at[i] = -999
        for t, sig in enumerate(recent_sigs):
            for i in range(n_zones):
                if sig[i] == 0:
                    last_empty_at[i] = t
        # longest time since empty -> predict break (1 zone for 3-partition, 2 for 7)
        k = 1 if n_zones <= 3 else (2 if n_zones <= 5 else 2)
        ordered = sorted(range(n_zones), key=lambda i: (last_empty_at[i], i))
        return set(ordered[:k])

    # single_dominant: only the top 1 zone by zero_counts
    best = max(range(n_zones), key=lambda i: (zero_counts[i], -i))
    return {best}


def pick_front_allowed_numbers(
    allowed_numbers: set[int],
    front_ranked: list[int],
    count: int = 5,
) -> tuple[int, ...]:
    out: list[int] = []
    for n in front_ranked:
        if n in allowed_numbers:
            out.append(n)
        if len(out) == count:
            break
    if len(out) < count:
        for n in front_ranked:
            if n not in out:
                out.append(n)
            if len(out) == count:
                break
    return tuple(sorted(out[:count]))


def hist_stats(draws: list[m.Draw], name: str, n_zones: int, zfn) -> None:
    empty_counts = Counter()
    per_zone_empty_rate = [0] * n_zones
    for d in draws:
        sig = signature(d.front, zfn)
        ez = len(empty_zones(sig))
        empty_counts[ez] += 1
        for i in range(n_zones):
            if sig[i] == 0:
                per_zone_empty_rate[i] += 1
    n = len(draws)
    print(f"\n--- {name} 历史形态 (n={n}) ---")
    print(f"  平均每期空区个数: {statistics.mean([len(empty_zones(signature(d.front, zfn))) for d in draws]):.2f}")
    for k in sorted(empty_counts):
        print(f"  空{k}个区: {empty_counts[k]}/{n} = {empty_counts[k]/n:.2%}")
    print("  各区断区频率(该区分区0个号):")
    for i in range(n_zones):
        print(f"    区{i+1}: {per_zone_empty_rate[i]/n:.2%}")


def rolling_compare(draws: list[m.Draw], name: str, n_zones: int, zfn) -> dict[str, float]:
    state = m.build_state(adaptive=True)
    modes = ("rotation", "hot_break", "single_dominant")
    agg = {mode: defaultdict(int) for mode in modes}
    lists = {mode: defaultdict(list) for mode in modes}

    for start in range(0, len(draws) - WINDOW):
        window = draws[start : start + WINDOW]
        actual = draws[start + WINDOW]
        recent_6 = [signature(d.front, zfn) for d in window[-6:]]
        fs, fft = m.score_front_numbers(window, state)
        bs, bft = m.score_back_numbers(window, state)
        fr = m.sort_candidates(fs)
        act_sig = signature(actual.front, zfn)

        for mode in modes:
            break_zones = predict_break_zones(recent_6, n_zones, mode=mode)
            # zone break accuracy: all predicted break zones are actually empty
            zone_ok = all(act_sig[z] == 0 for z in break_zones)
            agg[mode]["zone_pred_ok"] += int(zone_ok)
            # at least one predicted zone is empty
            agg[mode]["zone_any_hit"] += int(any(act_sig[z] == 0 for z in break_zones))
            # numbers only from non-break zones
            allowed = {n for n in m.FRONT_RANGE if zfn(n) not in break_zones}
            pred5 = pick_front_allowed_numbers(allowed, fr, 5)
            hits = len(set(pred5) & set(actual.front))
            lists[mode]["hits"].append(hits)
            agg[mode]["front_2plus"] += int(hits >= 2)
            agg[mode]["front_3plus"] += int(hits >= 3)
            # how many actual numbers fell in predicted-break zones (should be 0 if perfect)
            wrong_zone_nums = sum(1 for n in actual.front if zfn(n) in break_zones)
            lists[mode]["wrong_zone"].append(wrong_zone_nums)

        m.update_bonus_weights(
            state.front_bonus_weights, fft, actual.front, m.FRONT_RANGE,
            state.front_learning_rate, state.front_weight_decay, 1.75,
        )
        m.update_bonus_weights(
            state.back_bonus_weights, bft, actual.back, m.BACK_RANGE,
            state.back_learning_rate, state.back_weight_decay, 1.75,
        )

    nwin = len(draws) - WINDOW
    out: dict[str, float] = {}
    print(f"\n=== {name} 滚动预测 (n={nwin}, 50期窗, 自适应选号) ===")
    for mode in modes:
        print(f"  [{mode}]")
        print(f"    预测断区全对(所预测区均为0): {agg[mode]['zone_pred_ok']/nwin:.2%}")
        print(f"    预测断区至少中1区: {agg[mode]['zone_any_hit']/nwin:.2%}")
        print(f"    限制选5码后≥2前: {agg[mode]['front_2plus']/nwin:.2%}")
        print(f"    限制选5码后≥3前: {agg[mode]['front_3plus']/nwin:.2%}")
        print(f"    平均前区命中: {statistics.mean(lists[mode]['hits']):.3f}")
        print(f"    开奖号落在「预测断区」个数: {statistics.mean(lists[mode]['wrong_zone']):.2f}")
        out[f"{mode}_front_2plus"] = agg[mode]["front_2plus"] / nwin
        out[f"{mode}_front_3plus"] = agg[mode]["front_3plus"] / nwin
        out[f"{mode}_zone_any"] = agg[mode]["zone_any_hit"] / nwin
    return out


def rolling_baseline_and_break2(draws: list[m.Draw]) -> None:
    state = m.build_state(adaptive=True)
    n2, n3, b2_actual, b2_pred_ok = 0, 0, 0, 0
    nwin = len(draws) - WINDOW
    for start in range(0, nwin):
        window = draws[start : start + WINDOW]
        actual = draws[start + WINDOW]
        fs, fft = m.score_front_numbers(window, state)
        fr = m.sort_candidates(fs)
        pred5 = tuple(sorted(fr[:5]))
        hits = len(set(pred5) & set(actual.front))
        n2 += int(hits >= 2)
        n3 += int(hits >= 3)
        is_b2 = m.front_is_break_zone2(actual.front)
        b2_actual += int(is_b2)
        if is_b2:
            allowed = {n for n in m.FRONT_RANGE if not (13 <= n <= 24)}
            pred_b2 = pick_front_allowed_numbers(allowed, fr, 5)
            b2_pred_ok += int(len(set(pred_b2) & set(actual.front)) >= 2)
        m.update_bonus_weights(
            state.front_bonus_weights, fft, actual.front, m.FRONT_RANGE,
            state.front_learning_rate, state.front_weight_decay, 1.75,
        )
        _, bft = m.score_back_numbers(window, state)
        m.update_bonus_weights(
            state.back_bonus_weights, bft, actual.back, m.BACK_RANGE,
            state.back_learning_rate, state.back_weight_decay, 1.75,
        )
    print(f"\n=== 基线：不限区，直接取评分Top5 (n={nwin}) ===")
    print(f"  ≥2前: {n2/nwin:.2%}  ≥3前: {n3/nwin:.2%}")
    print(f"\n=== 断二区(13-24) 历史兑现 ===")
    print(f"  实际开奖断二区: {b2_actual/nwin:.2%}")
    print(f"  断二区成立时、仅在非中区选Top5的≥2前: {b2_pred_ok/max(b2_actual,1):.2%} ({b2_pred_ok}/{b2_actual}期)")


def main() -> None:
    draws = parse_draws()
    print(f"总期数: {len(draws)}")
    rolling_baseline_and_break2(draws)
    best_2plus = ("", 0.0, "")
    for name, (n_zones, zfn) in PARTITIONS.items():
        hist_stats(draws, name, n_zones, zfn)
        metrics = rolling_compare(draws, name, n_zones, zfn)
        for mode in ("rotation", "hot_break", "single_dominant"):
            v = metrics[f"{mode}_front_2plus"]
            if v > best_2plus[1]:
                best_2plus = (name, v, mode)

    print("\n" + "=" * 60)
    print(f"综合最高「限制选号≥2前」: {best_2plus[0]} / 策略[{best_2plus[2]}] = {best_2plus[1]:.2%}")


if __name__ == "__main__":
    main()
