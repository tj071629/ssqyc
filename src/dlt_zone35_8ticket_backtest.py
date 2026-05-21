"""8-ticket rolling backtest: baseline vs 3-zone vs 5-zone vs 3+5 combined."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlt_rolling_backtest as m

HISTORY = Path(__file__).resolve().parents[1] / "docs" / "大乐透历史开奖号码.md"
WINDOW = 50
TICKETS = 8

POLICIES: dict[str, str] = {
    "A_基线(断二区收口)": "baseline",
    "B_不限区收口": "none",
    "C_3区热断(6注+2豁免)": "zone3",
    "D_5区热断(6注+2豁免)": "zone5",
    "E_3+5结合": "zone35",
}


def summarize(hits_list: list[list[dict[str, int]]], windows: int) -> dict[str, float]:
    agg = defaultdict(int)
    for hits in hits_list:
        best = max(hits, key=lambda item: (item["front_hits"], item["back_hits"]))
        agg["best_front_2plus"] += int(best["front_hits"] >= 2)
        agg["best_front_3plus"] += int(best["front_hits"] >= 3)
        agg["best_back_1plus"] += int(best["back_hits"] >= 1)
        agg["best_back_exact"] += int(best["back_hits"] == 2)
        agg["any_front_2plus"] += int(any(h["front_hits"] >= 2 for h in hits))
        agg["any_back_1plus"] += int(any(h["back_hits"] >= 1 for h in hits))
        agg["tickets_front_2plus"] += sum(int(h["front_hits"] >= 2) for h in hits)
    n = windows
    return {
        "best_front_2plus": agg["best_front_2plus"] / n,
        "best_front_3plus": agg["best_front_3plus"] / n,
        "best_back_1plus": agg["best_back_1plus"] / n,
        "best_back_exact": agg["best_back_exact"] / n,
        "any_front_2plus": agg["any_front_2plus"] / n,
        "any_back_1plus": agg["any_back_1plus"] / n,
        "avg_ticket_front_2plus": agg["tickets_front_2plus"] / (n * TICKETS),
    }


def run_policy(draws: list[m.Draw], zone_policy: str) -> dict[str, float]:
    state = m.build_state(adaptive=True)
    all_hits: list[list[dict[str, int]]] = []
    zone_pred_ok = 0
    break2_actual = 0
    break2_hit_2plus = 0

    for start in range(0, len(draws) - WINDOW):
        window = draws[start : start + WINDOW]
        actual = draws[start + WINDOW]
        tickets, front_ranked, _, front_ft, back_ft, state_info = m.generate_tickets(
            window, state, count=TICKETS, zone_policy=zone_policy
        )
        hits = [m.hit_summary(ticket, actual) for ticket in tickets]
        all_hits.append(hits)

        if zone_policy in {"zone3", "zone35"}:
            pred_z3 = m.predict_hot_break_zone(window, m.zone_index, 3)
            sig = m.zone_signature(actual.front)
            if sig[pred_z3] == 0:
                zone_pred_ok += 1
        elif zone_policy == "zone5":
            pred_z5 = m.predict_hot_break_zone(window, m.zone_index_5, 5)
            sig5 = m.zone_signature_n(actual.front, m.zone_index_5)
            if sig5[pred_z5] == 0:
                zone_pred_ok += 1

        if m.front_is_break_zone2(actual.front):
            break2_actual += 1
            best = max(hits, key=lambda item: (item["front_hits"], item["back_hits"]))
            if best["front_hits"] >= 2:
                break2_hit_2plus += 1

        m.update_bonus_weights(
            state.front_bonus_weights, front_ft, actual.front, m.FRONT_RANGE,
            state.front_learning_rate, state.front_weight_decay, 1.75,
        )
        m.update_bonus_weights(
            state.back_bonus_weights, back_ft, actual.back, m.BACK_RANGE,
            state.back_learning_rate, state.back_weight_decay, 1.75,
        )
        m.update_template_scores(state, tickets, hits)

    windows = len(draws) - WINDOW
    stats = summarize(all_hits, windows)
    stats["zone_break_pred_ok"] = zone_pred_ok / windows if zone_policy in {"zone3", "zone5", "zone35"} else 0.0
    stats["actual_break2_rate"] = break2_actual / windows
    stats["break2_cond_best_2plus"] = break2_hit_2plus / max(break2_actual, 1)
    return stats


def main() -> None:
    draws = m.parse_history(HISTORY)
    print(f"历史期数: {len(draws)}  滚动窗: {WINDOW}  每期: {TICKETS}注  自适应: 是\n")
    rows: list[tuple[str, dict[str, float]]] = []
    for label, policy in POLICIES.items():
        stats = run_policy(draws, policy)
        rows.append((label, stats))
        print(f"--- {label} ({policy}) ---")
        print(f"  8注最优≥2前: {stats['best_front_2plus']:.2%}")
        print(f"  8注最优≥3前: {stats['best_front_3plus']:.2%}")
        print(f"  8注至少1注≥2前: {stats['any_front_2plus']:.2%}")
        print(f"  8注最优后区≥1: {stats['best_back_1plus']:.2%}")
        print(f"  8注最优后区2全中: {stats['best_back_exact']:.2%}")
        print(f"  单注均值≥2前占比: {stats['avg_ticket_front_2plus']:.2%}")
        if stats["zone_break_pred_ok"]:
            print(f"  热断区预测命中: {stats['zone_break_pred_ok']:.2%}")
        print()

    best = max(rows, key=lambda item: item[1]["best_front_2plus"])
    print("=" * 60)
    print(f"8注最优≥2前 最高: {best[0]} = {best[1]['best_front_2plus']:.2%}")
    print("\n排名(按8注最优≥2前):")
    for label, stats in sorted(rows, key=lambda item: -item[1]["best_front_2plus"]):
        print(f"  {label}: {stats['best_front_2plus']:.2%}  |  ≥3前 {stats['best_front_3plus']:.2%}  |  后≥1 {stats['best_back_1plus']:.2%}")


if __name__ == "__main__":
    main()
