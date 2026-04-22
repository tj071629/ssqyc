from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import dlt_rolling_backtest as base


@dataclass(frozen=True)
class DanTuoPlan:
    style: str
    dans: tuple[int, int]
    tuos: tuple[int, int, int, int, int]
    back: tuple[int, int]
    note: str


def _pool_zone_target(state_info: dict[str, object]) -> tuple[int, int, int]:
    front_state = state_info["front_state"]
    dominant_zero_zone = state_info["dominant_zero_zone"]
    if front_state == "extreme":
        if dominant_zero_zone == 0:
            return 1, 2, 4
        if dominant_zero_zone == 1:
            return 2, 1, 4
        return 3, 3, 1
    if front_state == "mid-heavy":
        return 1, 4, 2
    return 2, 3, 2


def _choose_dans(
    front_ranked: list[int],
    feature_table: dict[int, dict[str, float]],
    state_info: dict[str, object],
    style: str,
) -> tuple[int, int]:
    rank_map = {number: idx for idx, number in enumerate(front_ranked)}
    front_state = state_info["front_state"]
    dominant_zone = state_info["dominant_zone"]
    dominant_zero_zone = state_info["dominant_zero_zone"]

    def score(number: int) -> float:
        features = feature_table[number]
        value = 24 - rank_map[number]
        value += features["repeat_last"] * 9.0
        value += features["neighbor_last"] * 7.0
        value += features["hot10"] * 6.0
        value += features["hot20"] * 4.0
        value += features["pair_synergy"] * 2.5
        value -= features["omission_deep"] * 2.0
        if front_state == "extreme" and base.zone_index(number) == dominant_zone:
            value += 3.0
        if style == "中区加压票" and 13 <= number <= 24:
            value += 4.0
        if style == "回稳反抽票" and base.zone_index(number) != dominant_zero_zone:
            value += 2.0
        if style == "逆向跳号票" and rank_map[number] >= 8:
            value += 2.0
        return value

    ordered = sorted(front_ranked[:16], key=lambda number: (-score(number), number))
    first = ordered[0]
    second = None
    for candidate in ordered[1:]:
        if front_state != "extreme" and base.zone_index(candidate) == base.zone_index(first):
            continue
        second = candidate
        break
    if second is None:
        second = ordered[1]
    return tuple(sorted((first, second)))


def _choose_tuos(
    front_ranked: list[int],
    feature_table: dict[int, dict[str, float]],
    state_info: dict[str, object],
    style: str,
    dans: tuple[int, int],
) -> tuple[int, int, int, int, int]:
    rank_map = {number: idx for idx, number in enumerate(front_ranked)}
    target = _pool_zone_target(state_info)
    zone_counts = [0, 0, 0]
    for number in dans:
        zone_counts[base.zone_index(number)] += 1

    front_state = state_info["front_state"]
    dominant_zone = state_info["dominant_zone"]
    dominant_zero_zone = state_info["dominant_zero_zone"]

    def score(number: int) -> float:
        features = feature_table[number]
        value = 26 - rank_map[number]
        value += features["hot10"] * 5.0
        value += features["neighbor_last"] * 4.5
        value += features["neighbor_prev"] * 2.0
        value += features["jump2_last"] * 1.5
        value += features["omission_mid"] * 2.2
        value += features["omission_warm"] * 1.4
        if front_state == "extreme" and base.zone_index(number) == dominant_zone:
            value += 2.8
        if style == "回稳反抽票" and base.zone_index(number) != dominant_zero_zone:
            value += 2.0
        if style == "中区加压票" and 13 <= number <= 24:
            value += 2.8
        if style == "逆向跳号票" and rank_map[number] >= 10:
            value += 2.5
        return value

    ordered = sorted((number for number in front_ranked[:22] if number not in dans), key=lambda number: (-score(number), number))
    tuos: list[int] = []
    for number in ordered:
        zone = base.zone_index(number)
        if zone_counts[zone] >= target[zone]:
            continue
        tuos.append(number)
        zone_counts[zone] += 1
        if len(tuos) == 5:
            break

    if len(tuos) < 5:
        for number in ordered:
            if number in tuos:
                continue
            tuos.append(number)
            if len(tuos) == 5:
                break

    # Keep one deeper-ranked diversity number so the dantuo pool does not collapse into a pure hot cluster.
    if all(rank_map[number] < 10 for number in tuos):
        diversity = next((number for number in front_ranked[10:24] if number not in dans and number not in tuos), None)
        if diversity is not None:
            tuos[-1] = diversity

    return tuple(sorted(tuos[:5]))


def generate_dantuo_plan(window: list[base.Draw], state: base.StrategyState) -> tuple[DanTuoPlan, list[int], list[int], dict[int, dict[str, float]], dict[int, dict[str, float]], dict[str, object]]:
    front_scores, front_feature_table = base.score_front_numbers(window, state)
    back_scores, back_feature_table = base.score_back_numbers(window, state)
    front_ranked = base.sort_candidates(front_scores)
    back_ranked = base.sort_candidates(back_scores)
    state_info = base.classify_window_state(window)
    style = base.choose_template_names(state_info, state, 1)[0]

    dans = _choose_dans(front_ranked, front_feature_table, state_info, style)
    tuos = _choose_tuos(front_ranked, front_feature_table, state_info, style, dans)

    last = window[-1]
    hot_back = [number for number in back_ranked[:6] if number not in last.back]
    warm_back = [number for number in back_ranked if 3 <= base.omission(window, "back", number) <= 6]
    back_neighbors = sorted(
        {
            neighbor
            for value in last.back
            for neighbor in (value - 1, value + 1)
            if neighbor in base.BACK_RANGE
        }
    )
    back_pair_pool = base.build_back_pair_pool(back_ranked, state_info, last.back, hot_back, warm_back, back_neighbors)
    back_pair = back_pair_pool[0]

    note = f"{style}；2胆5拖覆盖10注单式"
    plan = DanTuoPlan(style=style, dans=dans, tuos=tuos, back=back_pair, note=note)
    return plan, front_ranked, back_ranked, front_feature_table, back_feature_table, state_info


def dantuo_hit_summary(plan: DanTuoPlan, actual: base.Draw) -> dict[str, int]:
    dan_hits = len(set(plan.dans) & set(actual.front))
    tuo_hits = len(set(plan.tuos) & set(actual.front))
    best_front_hits = dan_hits + min(tuo_hits, 3)
    back_hits = len(set(plan.back) & set(actual.back))
    return {
        "dan_hits": dan_hits,
        "tuo_hits": tuo_hits,
        "best_front_hits": best_front_hits,
        "back_hits": back_hits,
        "front_pool_hits": dan_hits + tuo_hits,
        "full_cover": int(dan_hits == 2 and tuo_hits >= 3),
    }


def rolling_dantuo_backtest(
    draws: list[base.Draw],
    window_size: int,
    *,
    adaptive: bool,
    adaptive_config: dict[str, float] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    results: list[dict[str, object]] = []
    aggregate = defaultdict(int)
    aggregate_lists: dict[str, list[float]] = defaultdict(list)
    state = base.build_state(adaptive=adaptive, **(adaptive_config or {}))

    for start in range(0, len(draws) - window_size):
        window = draws[start : start + window_size]
        actual = draws[start + window_size]
        plan, front_ranked, back_ranked, front_feature_table, back_feature_table, state_info = generate_dantuo_plan(window, state)
        hits = dantuo_hit_summary(plan, actual)
        flags = base.actual_feature_flags(window, actual, front_ranked, back_ranked)

        aggregate["windows"] += 1
        aggregate["dan_all_hit"] += int(hits["dan_hits"] == 2)
        aggregate["dan_any_hit"] += int(hits["dan_hits"] >= 1)
        aggregate["front_3plus"] += int(hits["best_front_hits"] >= 3)
        aggregate["front_4plus"] += int(hits["best_front_hits"] >= 4)
        aggregate["full_cover"] += hits["full_cover"]
        aggregate["back_1plus"] += int(hits["back_hits"] >= 1)
        aggregate["back_exact"] += int(hits["back_hits"] == 2)

        aggregate_lists["best_front_hits"].append(hits["best_front_hits"])
        aggregate_lists["front_pool_hits"].append(hits["front_pool_hits"])
        aggregate_lists["back_hits"].append(hits["back_hits"])
        aggregate_lists["front_top15"].append(flags["front_top15"])

        results.append(
            {
                "train_start": window[0].issue,
                "train_end": window[-1].issue,
                "actual_issue": actual.issue,
                "actual_front": actual.front,
                "actual_back": actual.back,
                "plan": plan,
                "hits": hits,
                "flags": flags,
                "state_info": state_info,
                "style_score": round(state.template_scores[plan.style], 3),
            }
        )

        if adaptive:
            base.update_bonus_weights(
                state.front_bonus_weights,
                front_feature_table,
                actual.front,
                base.FRONT_RANGE,
                state.front_learning_rate,
                state.front_weight_decay,
                limit=1.75,
            )
            base.update_bonus_weights(
                state.back_bonus_weights,
                back_feature_table,
                actual.back,
                base.BACK_RANGE,
                state.back_learning_rate,
                state.back_weight_decay,
                limit=1.75,
            )
            dummy_ticket = base.Ticket(plan.style, tuple(sorted(plan.dans + plan.tuos[:3])), plan.back, plan.note)
            base.update_template_scores(state, [dummy_ticket], [{"front_hits": hits["best_front_hits"], "back_hits": hits["back_hits"]}])

    summary = {
        "adaptive": adaptive,
        "windows": aggregate["windows"],
        "dan_all_hit_rate": aggregate["dan_all_hit"] / aggregate["windows"],
        "dan_any_hit_rate": aggregate["dan_any_hit"] / aggregate["windows"],
        "front_3plus_rate": aggregate["front_3plus"] / aggregate["windows"],
        "front_4plus_rate": aggregate["front_4plus"] / aggregate["windows"],
        "full_cover_rate": aggregate["full_cover"] / aggregate["windows"],
        "back_1plus_rate": aggregate["back_1plus"] / aggregate["windows"],
        "back_exact_rate": aggregate["back_exact"] / aggregate["windows"],
        "avg_best_front_hits": statistics.mean(aggregate_lists["best_front_hits"]),
        "avg_front_pool_hits": statistics.mean(aggregate_lists["front_pool_hits"]),
        "avg_back_hits": statistics.mean(aggregate_lists["back_hits"]),
        "avg_actual_front_top15": statistics.mean(aggregate_lists["front_top15"]),
        "top_back_weights": base.top_weight_items(state.back_bonus_weights),
        "top_styles": sorted(state.template_scores.items(), key=lambda item: (-item[1], item[0]))[:5],
    }
    return results, summary


def compare_summaries(fixed: dict[str, object], adaptive: dict[str, object]) -> dict[str, float]:
    return {
        "dan_all_hit_delta": adaptive["dan_all_hit_rate"] - fixed["dan_all_hit_rate"],
        "front_3plus_delta": adaptive["front_3plus_rate"] - fixed["front_3plus_rate"],
        "front_4plus_delta": adaptive["front_4plus_rate"] - fixed["front_4plus_rate"],
        "full_cover_delta": adaptive["full_cover_rate"] - fixed["full_cover_rate"],
        "back_1plus_delta": adaptive["back_1plus_rate"] - fixed["back_1plus_rate"],
        "back_exact_delta": adaptive["back_exact_rate"] - fixed["back_exact_rate"],
    }


def build_report(
    fixed_results: list[dict[str, object]],
    fixed_summary: dict[str, object],
    adaptive_results: list[dict[str, object]],
    adaptive_summary: dict[str, object],
    comparison: dict[str, float],
    window_size: int,
) -> str:
    lines: list[str] = []
    lines.append("# 大乐透2胆5拖逐期回测复盘报告")
    lines.append("")
    lines.append("## 1. 回测设定")
    lines.append(f"- 历史窗口：固定 `{window_size}` 期。")
    lines.append("- 每期输出：`前区2胆 + 5拖码 + 后区2码`，等价 `10` 注单式，不含追加金额为 `20元`。")
    lines.append("- 复盘口径：前区以胆拖隐含的 `10` 注里最优前区命中为准；后区因固定 `2` 码，所有隐含注共享同一后区结果。")
    lines.append("- 对比方式：先跑固定规则，再跑逐期在线修订规则。")
    lines.append("")
    lines.append("## 2. 固定规则 vs 逐期自适应")
    lines.append("| 指标 | 固定规则 | 逐期自适应 | 差值 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 2个胆码全中 | `{fixed_summary['dan_all_hit_rate']:.2%}` | `{adaptive_summary['dan_all_hit_rate']:.2%}` | `{comparison['dan_all_hit_delta']:+.2%}` |")
    lines.append(f"| 隐含10注至少前区3中 | `{fixed_summary['front_3plus_rate']:.2%}` | `{adaptive_summary['front_3plus_rate']:.2%}` | `{comparison['front_3plus_delta']:+.2%}` |")
    lines.append(f"| 隐含10注至少前区4中 | `{fixed_summary['front_4plus_rate']:.2%}` | `{adaptive_summary['front_4plus_rate']:.2%}` | `{comparison['front_4plus_delta']:+.2%}` |")
    lines.append(f"| 前区全覆盖5中 | `{fixed_summary['full_cover_rate']:.2%}` | `{adaptive_summary['full_cover_rate']:.2%}` | `{comparison['full_cover_delta']:+.2%}` |")
    lines.append(f"| 后区至少1中 | `{fixed_summary['back_1plus_rate']:.2%}` | `{adaptive_summary['back_1plus_rate']:.2%}` | `{comparison['back_1plus_delta']:+.2%}` |")
    lines.append(f"| 后区2中 | `{fixed_summary['back_exact_rate']:.2%}` | `{adaptive_summary['back_exact_rate']:.2%}` | `{comparison['back_exact_delta']:+.2%}` |")
    lines.append("")
    lines.append("## 3. 解读")
    lines.append(f"- 自适应最终高分风格：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['top_styles'])}`")
    lines.append(f"- 后区在线增强信号：`{', '.join(f'{name}({value:.2f})' for name, value in adaptive_summary['top_back_weights'])}`")
    lines.append(f"- 隐含10注平均前区最佳命中：`{adaptive_summary['avg_best_front_hits']:.2f}`")
    lines.append(f"- 7码池平均覆盖到的实际前区号码个数：`{adaptive_summary['avg_front_pool_hits']:.2f}`")
    lines.append("")
    lines.append("## 4. 最近12期样本")
    lines.append("| 训练区间 | 预测期 | 胆拖方案 | 实际开奖 | 命中摘要 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in adaptive_results[-12:]:
        plan: DanTuoPlan = row["plan"]
        hits = row["hits"]
        plan_text = f"`{base.format_numbers(plan.dans)} 胆 + {base.format_numbers(plan.tuos)} 拖 + {base.format_numbers(plan.back)}`"
        actual_text = f"`{base.format_numbers(row['actual_front'])} + {base.format_numbers(row['actual_back'])}`"
        hit_text = f"`胆{hits['dan_hits']}中 拖{hits['tuo_hits']}中 前区最佳{hits['best_front_hits']}中 后区{hits['back_hits']}中`"
        lines.append(f"| {row['train_start']}-{row['train_end']} | {row['actual_issue']} | {plan_text} | {actual_text} | {hit_text} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="大乐透2胆5拖逐期回测")
    parser.add_argument("--history", default="docs/大乐透历史开奖号码.md", help="历史开奖 Markdown 文件路径")
    parser.add_argument("--report", default="docs/大乐透2胆5拖逐期回测复盘报告.md", help="报告输出路径")
    parser.add_argument("--window-size", type=int, default=base.WINDOW_SIZE, help="滚动窗口大小")
    args = parser.parse_args()

    draws = base.parse_history(Path(args.history))
    if len(draws) <= args.window_size:
        raise SystemExit("历史数据不足，无法完成滚动回测。")

    fixed_results, fixed_summary = rolling_dantuo_backtest(draws, args.window_size, adaptive=False)
    adaptive_results, adaptive_summary = rolling_dantuo_backtest(draws, args.window_size, adaptive=True)
    comparison = compare_summaries(fixed_summary, adaptive_summary)
    report = build_report(fixed_results, fixed_summary, adaptive_results, adaptive_summary, comparison, args.window_size)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"已生成报告：{report_path}")
    print(f"固定规则 胆码双中率：{fixed_summary['dan_all_hit_rate']:.2%}")
    print(f"自适应规则 胆码双中率：{adaptive_summary['dan_all_hit_rate']:.2%}")
    print(f"固定规则 前区全覆盖率：{fixed_summary['full_cover_rate']:.2%}")
    print(f"自适应规则 前区全覆盖率：{adaptive_summary['full_cover_rate']:.2%}")
    print(f"固定规则 后区1码命中率：{fixed_summary['back_1plus_rate']:.2%}")
    print(f"自适应规则 后区1码命中率：{adaptive_summary['back_1plus_rate']:.2%}")


if __name__ == "__main__":
    main()
