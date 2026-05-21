"""Compare SSQ compound bet formats 7+1 vs 6+5 using ssq_rolling_backtest scoring."""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from ssq_rolling_backtest import (  # noqa: E402
    WINDOW_SIZE,
    Draw,
    StrategyState,
    build_state,
    choose_red,
    generate_tickets,
    is_zone1_break_last,
    parse_history,
    pick_blue_pool_five,
    refine_red_ranked_zone1_follow_ties,
    score_blue,
    score_red,
    slice_recent_draws,
    zone1_break_follow_red_counts,
)


def prize_tier(red_hits: int, blue_hit: int) -> int:
    """0=no prize; 1..7 = 一等奖..幸运奖 (per docs/双色球下期选号规则)."""
    if red_hits == 6 and blue_hit:
        return 1
    if red_hits == 6:
        return 2
    if red_hits == 5 and blue_hit:
        return 3
    if red_hits == 5 or (red_hits == 4 and blue_hit):
        return 4
    if red_hits == 4 or (red_hits == 3 and blue_hit):
        return 5
    if blue_hit and red_hits <= 2:
        return 6
    if red_hits == 3:
        return 7
    return 0


def red_ranked_for_window(
    window: list[Draw],
    state: StrategyState,
    zone1_follow: dict[int, int] | None,
) -> list[int]:
    ranked, _, scores = score_red(window, state)
    if is_zone1_break_last(window) and zone1_follow is not None:
        ranked = refine_red_ranked_zone1_follow_ties(scores, zone1_follow)
    return ranked


def build_compound_7_1(window: list[Draw], state: StrategyState, zone1_follow: dict[int, int] | None) -> list[tuple[tuple[int, ...], int]]:
    red_ranked = red_ranked_for_window(window, state, zone1_follow)
    blue_ranked, _ = score_blue(window, state)
    pool7 = red_ranked[:7]
    blue = pick_blue_pool_five(window, blue_ranked)[0]
    lines: list[tuple[tuple[int, ...], int]] = []
    for combo in itertools.combinations(pool7, 6):
        lines.append((tuple(sorted(combo)), blue))
    return lines


def build_compound_6_5(window: list[Draw], state: StrategyState, zone1_follow: dict[int, int] | None) -> list[tuple[tuple[int, ...], int]]:
    red_ranked = red_ranked_for_window(window, state, zone1_follow)
    blue_ranked, _ = score_blue(window, state)
    core = red_ranked[:4]
    hot = [n for n in red_ranked[:12] if n not in core]
    red6 = choose_red(red_ranked, include=core + hot[:2])
    blues = pick_blue_pool_five(window, blue_ranked)
    return [(red6, b) for b in blues]


def evaluate_lines(lines: list[tuple[tuple[int, ...], int]], actual: Draw) -> dict[str, object]:
    tiers: list[int] = []
    red_hits_list: list[int] = []
    blue_hits_list: list[int] = []
    for red, blue in lines:
        rh = len(set(red) & set(actual.red))
        bh = int(blue == actual.blue)
        red_hits_list.append(rh)
        blue_hits_list.append(bh)
        tiers.append(prize_tier(rh, bh))
    best_idx = max(range(len(lines)), key=lambda i: (tiers[i], red_hits_list[i], blue_hits_list[i]))
    return {
        "line_count": len(lines),
        "tiers": tiers,
        "any_prize": int(any(t > 0 for t in tiers)),
        "any_blue": int(any(blue_hits_list)),
        "any_red_3plus": int(any(r >= 3 for r in red_hits_list)),
        "any_red_4plus": int(any(r >= 4 for r in red_hits_list)),
        "any_5p1": int(any(r >= 5 and b for r, b in zip(red_hits_list, blue_hits_list))),
        "any_6p0": int(any(r == 6 for r in red_hits_list)),
        "any_6p1": int(any(r == 6 and b for r, b in zip(red_hits_list, blue_hits_list))),
        "best_red": red_hits_list[best_idx],
        "best_blue": blue_hits_list[best_idx],
        "best_tier": tiers[best_idx],
    }


def aggregate_rates(rows: list[dict[str, int]], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def run_compare(
    draws: list[Draw],
    window_size: int,
    preset: str,
    label: str,
) -> dict[str, object]:
    state = build_state(adaptive=False, preset=preset)
    zone1_follow = zone1_break_follow_red_counts(draws)
    rows_71: list[dict[str, int]] = []
    rows_65: list[dict[str, int]] = []
    rows_5: list[dict[str, int]] = []

    for start in range(0, len(draws) - window_size):
        window = draws[start : start + window_size]
        actual = draws[start + window_size]

        lines71 = build_compound_7_1(window, state, zone1_follow)
        lines65 = build_compound_6_5(window, state, zone1_follow)
        ev71 = evaluate_lines(lines71, actual)
        ev65 = evaluate_lines(lines65, actual)
        rows_71.append({k: int(v) for k, v in ev71.items() if k != "tiers" and k != "line_count"})
        rows_65.append({k: int(v) for k, v in ev65.items() if k != "tiers" and k != "line_count"})

        tickets, _, _, _, _ = generate_tickets(
            window,
            state,
            5,
            zone1_follow_red_counts=zone1_follow,
            prior_draws=draws[: start + window_size],
        )
        tiers5: list[int] = []
        for t in tickets:
            rh = len(set(t.red) & set(actual.red))
            bh = int(t.blue == actual.blue)
            tiers5.append(prize_tier(rh, bh))
        rows_5.append(
            {
                "any_prize": int(any(t > 0 for t in tiers5)),
                "any_blue": int(any(int(t.blue == actual.blue) for t in tickets)),
                "any_red_3plus": int(any(len(set(t.red) & set(actual.red)) >= 3 for t in tickets)),
                "any_red_4plus": int(any(len(set(t.red) & set(actual.red)) >= 4 for t in tickets)),
                "any_5p1": int(any(len(set(t.red) & set(actual.red)) >= 5 and t.blue == actual.blue for t in tickets)),
                "any_6p0": int(any(len(set(t.red) & set(actual.red)) == 6 for t in tickets)),
                "any_6p1": int(any(len(set(t.red) & set(actual.red)) == 6 and t.blue == actual.blue for t in tickets)),
            }
        )

    n = len(rows_71)
    keys = ("any_prize", "any_blue", "any_red_3plus", "any_red_4plus", "any_5p1", "any_6p0", "any_6p1")

    def pack(rows: list[dict[str, int]], lines_per_bet: int, cost_yuan: int) -> dict[str, float]:
        out: dict[str, float] = {"windows": n, "lines_per_period": lines_per_bet, "cost_yuan_per_period": cost_yuan}
        for k in keys:
            out[k] = aggregate_rates(rows, k)
        return out

    return {
        "label": label,
        "windows": n,
        "preset": preset,
        "compound_7_1": pack(rows_71, 7, 14),
        "compound_6_5": pack(rows_65, 5, 10),
        "single_5x": pack(rows_5, 5, 10),
    }


def print_report(result: dict[str, object]) -> None:
    print(f"\n=== {result['label']} | preset={result['preset']} | windows={result['windows']} ===")
    print("口径: 50期窗滚动预测下一期；奖级按规则文档(含六等/幸运奖)。")
    print("选号: score_red/blue + recent300蓝权 + 断一区后红序tie；6+5蓝池含小号热段尾区强制(专规v0.5)。")
    print()
    hdr = f"{'指标':<22} {'7+1(7注/14元)':>16} {'6+5(5注/10元)':>16} {'5注单式(10元)':>16}"
    print(hdr)
    print("-" * len(hdr))
    k71 = result["compound_7_1"]
    k65 = result["compound_6_5"]
    k5 = result["single_5x"]
    metrics = [
        ("至少一注有奖", "any_prize"),
        ("至少一注蓝中", "any_blue"),
        ("至少一注红3+", "any_red_3plus"),
        ("至少一注红4+", "any_red_4plus"),
        ("至少一注5+1(三等)", "any_5p1"),
        ("至少一注6+0(二等)", "any_6p0"),
        ("至少一注6+1(一等)", "any_6p1"),
    ]
    for name, key in metrics:
        print(f"{name:<22} {k71[key]:>15.2%} {k65[key]:>15.2%} {k5[key]:>15.2%}")
    print()
    print("同成本10元近似对照(命中率/注数折算，仅作参考):")
    print(f"  7+1 蓝中率/注: {k71['any_blue']/7:.2%}  |  6+5: {k65['any_blue']/5:.2%}  |  5单式: {k5['any_blue']/5:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="双色球 7+1 vs 6+5 复式历史对比")
    parser.add_argument("--history", default="docs/双色球历史开奖号码.md")
    parser.add_argument("--recent-draws", type=int, default=300)
    parser.add_argument("--preset", default="recent300", choices=("recent300", "full"))
    parser.add_argument("--full-also", action="store_true", help="同时输出全历史结果")
    args = parser.parse_args()

    path = _REPO / args.history
    full = parse_history(path)
    near = slice_recent_draws(full, args.recent_draws)
    r_near = run_compare(near, WINDOW_SIZE, args.preset, f"近端最近{args.recent_draws}期")
    print_report(r_near)
    if args.full_also:
        r_full = run_compare(full, WINDOW_SIZE, args.preset, "全历史")
        print_report(r_full)

    k71 = r_near["compound_7_1"]
    k65 = r_near["compound_6_5"]
    print("\n【近端结论摘要】")
    if k71["any_prize"] > k65["any_prize"]:
        print(f"- 「至少一注有奖」7+1 更高: {k71['any_prize']:.2%} vs {k65['any_prize']:.2%}（7+1多2注成本）")
    elif k65["any_prize"] > k71["any_prize"]:
        print(f"- 「至少一注有奖」6+5 更高: {k65['any_prize']:.2%} vs {k71['any_prize']:.2%}")
    else:
        print(f"- 「至少一注有奖」两者接近: {k71['any_prize']:.2%}")
    if k65["any_blue"] > k71["any_blue"]:
        print(f"- 「蓝球覆盖」6+5 更好: {k65['any_blue']:.2%} vs {k71['any_blue']:.2%}（符合扩蓝思路）")
    elif k71["any_blue"] > k65["any_blue"]:
        print(f"- 「蓝球覆盖」7+1 更好: {k71['any_blue']:.2%} vs {k65['any_blue']:.2%}")
    if k71["any_red_3plus"] > k65["any_red_3plus"]:
        print(f"- 「红球3+」7+1 更好: {k71['any_red_3plus']:.2%} vs {k65['any_red_3plus']:.2%}（多1个红球位）")
    elif k65["any_red_3plus"] > k71["any_red_3plus"]:
        print(f"- 「红球3+」6+5 更好: {k65['any_red_3plus']:.2%} vs {k71['any_red_3plus']:.2%}")


if __name__ == "__main__":
    main()
