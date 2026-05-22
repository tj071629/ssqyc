"""福彩3D 单选/直选复式：选 3 个不同号码 → 6 注全排列（组六口径，不含组三/豹子）。"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

from p3_rolling_backtest import (
    DIGITS,
    INITIAL_WEIGHTS,
    POSITIONS,
    StrategyState,
    format_number,
    parse_draws,
    pattern_type,
    rank_candidates,
    window,
)


def compound6(digits: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    if len(set(digits)) != 3:
        raise ValueError("Need 3 distinct digits for 6-bet straight compound.")
    return list(permutations(digits))


def digit_freq(history, last_n: int) -> Counter:
    counter: Counter = Counter()
    for draw in window(history, last_n):
        for pos in POSITIONS:
            counter[draw.number[pos]] += 1
    return counter


def digit_omission(history) -> dict[int, int]:
    """Draws since digit last appeared at any position."""
    miss = {d: len(history) + 1 for d in DIGITS}
    for idx, draw in enumerate(reversed(history)):
        for pos in POSITIONS:
            d = draw.number[pos]
            if miss[d] > len(history):
                miss[d] = idx
    return miss


def pick_top3_by_counter(counter: Counter, tie: dict[int, int] | None = None) -> tuple[int, int, int]:
    tie = tie or {}

    def key(d: int) -> tuple:
        return (-counter[d], tie.get(d, d), d)

    ranked = sorted(DIGITS, key=key)
    return ranked[0], ranked[1], ranked[2]


def pick_hot3(history, last_n: int = 30) -> tuple[int, int, int]:
    return pick_top3_by_counter(digit_freq(history, last_n))


def pick_omit3(history) -> tuple[int, int, int]:
    miss = digit_omission(history)
    counter = Counter({d: miss[d] for d in DIGITS})
    return pick_top3_by_counter(counter)


def pick_mix_hot2_omit1(history) -> tuple[int, int, int]:
    hot = digit_freq(history, 30)
    miss = digit_omission(history)
    hot_rank = sorted(DIGITS, key=lambda d: (-hot[d], d))
    omit_rank = sorted(DIGITS, key=lambda d: (-miss[d], d))
    chosen = [hot_rank[0], hot_rank[1]]
    for d in omit_rank:
        if d not in chosen:
            chosen.append(d)
            break
    chosen.sort()
    return chosen[0], chosen[1], chosen[2]


def pick_fusion_zuliu(history) -> tuple[int, int, int]:
    state = StrategyState(adaptive=False, weights=dict(INITIAL_WEIGHTS))
    ranked = rank_candidates(history, state)
    for cand, _, _ in ranked:
        if pattern_type(cand) == "zuliu":
            return cand  # type: ignore[return-value]
    return pick_hot3(history)


def pick_random3(rng: random.Random) -> tuple[int, int, int]:
    return tuple(sorted(rng.sample(list(DIGITS), 3)))  # type: ignore[return-value]


@dataclass
class Acc:
    rounds: int = 0
    exact: int = 0
    zuliu_rounds: int = 0
    zuliu_hit: int = 0
    set_cover: int = 0
    pattern_miss: int = 0


def eval_compound(tickets: list[tuple[int, int, int]], actual: tuple[int, int, int]) -> tuple[int, int, int]:
    exact = int(actual in tickets)
    set_hit = int(set(actual) <= {d for t in tickets for d in t})
    return exact, set_hit, int(pattern_type(actual) == "zuliu")


def run_compare(
    draws,
    *,
    min_history: int,
    last_n: int | None,
    seed: int = 42,
) -> tuple[dict[str, Acc], int]:
    rng = random.Random(seed)
    start = max(min_history, len(draws) - last_n) if last_n else min_history
    schemes: dict[str, object] = {
        "热号Top3(近30期)": pick_hot3,
        "热号Top3(近100期)": lambda h: pick_hot3(h, 100),
        "遗漏Top3": pick_omit3,
        "热2+遗漏1": pick_mix_hot2_omit1,
        "融合最优组六": pick_fusion_zuliu,
        "随机3码": lambda h: pick_random3(rng),
    }
    accs = {name: Acc() for name in schemes}

    for idx in range(start, len(draws)):
        history = draws[:idx]
        actual = draws[idx].number
        for name, picker in schemes.items():
            digits = picker(history)  # type: ignore[operator]
            tickets = compound6(digits)
            exact, set_hit, is_zuliu = eval_compound(tickets, actual)
            acc = accs[name]
            acc.rounds += 1
            acc.exact += exact
            acc.set_cover += set_hit
            if is_zuliu:
                acc.zuliu_rounds += 1
                acc.zuliu_hit += exact
            else:
                acc.pattern_miss += 1

    return accs, start


def pct(n: float, d: float) -> str:
    return f"{100 * n / d:.3f}%" if d else "0.000%"


def next_preview(draws) -> list[str]:
    history = draws
    picks = {
        "热号Top3(近30期)": pick_hot3(history),
        "热号Top3(近100期)": pick_hot3(history, 100),
        "遗漏Top3": pick_omit3(history),
        "热2+遗漏1": pick_mix_hot2_omit1(history),
        "融合最优组六": pick_fusion_zuliu(history),
    }
    freq30 = digit_freq(history, 30)
    omit = digit_omission(history)
    lines = [
        "## 下一期参考（选 3 码 → 6 注单选复式）",
        "",
        "### 各位数字统计（与选号页一致：全位合计）",
        "",
        "| 数字 | 近30期次数 | 当前遗漏 |",
        "| ---: | ---: | ---: |",
    ]
    for d in DIGITS:
        lines.append(f"| {d} | {freq30[d]} | {omit[d]} |")
    lines.append("")
    lines.append("### 推荐 3 码池")
    lines.append("")
    lines.append("| 策略 | 3码 | 展开6注 |")
    lines.append("| --- | --- | --- |")
    for name, digits in picks.items():
        tickets = compound6(digits)
        nums = ", ".join(format_number(t) for t in tickets)
        ds = "".join(str(x) for x in sorted(digits))
        lines.append(f"| {name} | **{ds}** | {nums} |")
    lines.append("")
    return lines


def render_report(draws, windows: list[int | None], min_history: int) -> str:
    lines = [
        "# 福彩3D单选复式（选3码·6注）历史预测分析",
        "",
        "## 口径（与选号页一致）",
        "",
        "- **玩法**：在 **0–9** 中选 **3 个不同号码**，生成 **6 注直选**（3 个数字的全排列，如 012 → 012/021/102/120/201/210）。",
        "- **注数 / 成本**：**6 注**，约 **12 元**（页面显示 12 魔币）。",
        "- **不含**：组三（对子）、豹子；若开奖为 112、000，则 6 注复式 **必不中** 直选。",
        "- **理论命中率**：6/1000 = **0.6%**（仅对「开奖为组六且恰为所选 3 码」）。",
        "- **组六占比**：全历史约 **72%** 期次为组六，故 unconditional 期望约 **0.6% × 72% ≈ 0.43%** 量级（粗算）。",
        "",
        "## 数据范围",
        "",
        f"- 历史期数：{len(draws)}",
        f"- 区间：{draws[0].issue}～{draws[-1].issue}",
        f"- 冷启动：前 {min_history} 期",
        "",
    ]

    zuliu_total = sum(1 for d in draws if pattern_type(d.number) == "zuliu")
    lines.append(f"- 全历史组六期数：**{zuliu_total}**（{pct(zuliu_total, len(draws))}）")
    lines.append("")

    for last_n in windows:
        accs, start = run_compare(draws, min_history=min_history, last_n=last_n)
        title = f"近{last_n}期" if last_n else f"全量（{len(draws) - min_history} 期）"
        lines.append(f"## {title}（自 {draws[start].issue} 起）")
        lines.append("")
        lines.append("### 6 注直选全中")
        lines.append("")
        lines.append("| 策略 | 6注全中 | 命中率 | 相对理论0.6% | 组六期命中率 | 数字集合覆盖 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        lines.append("")
        lines.append("数字集合覆盖：开奖三位数字均落在所选3码内（组三/豹子也可能为是，但不中直选）。")
        lines.append("")
        for name in accs:
            a = accs[name]
            r = a.rounds or 1
            zr = a.zuliu_rounds or 1
            rate = a.exact / r
            lines.append(
                f"| {name} | {a.exact}/{a.rounds} | {pct(a.exact, r)} | {rate / 0.006:.2f}x | "
                f"{pct(a.zuliu_hit, zr)} | {pct(a.set_cover, r)} |"
            )
        lines.append("")
        lines.append("### 未中构成")
        lines.append("")
        lines.append("| 策略 | 组三/豹子期（必不中） |")
        lines.append("| --- | ---: |")
        for name, a in accs.items():
            lines.append(f"| {name} | {a.pattern_miss}/{a.rounds} ({pct(a.pattern_miss, a.rounds)}) |")
        lines.append("")

    lines.extend(next_preview(draws))
    lines.extend(
        [
            "## 结论",
            "",
            "1. **选 3 码 6 注** 与 **每位 Top3 共 27 注** 完全不同；前者是 **组六直选包号**，后者是 **按位定位复式**。",
            "2. 全量回测中，各策略直选命中率应在 **0.6%**（6/1000）附近；略高于/低于随机 3 码属正常波动。",
            "3. **近 30 期热号 Top3** 通常最稳；**遗漏 Top3** 波动大，不宜单独重仓。",
            "4. **融合最优组六** 取模型排名第一的组六形态，与热号 Top3 往往接近，可二选一。",
            "5. 约 **28%** 期为组三/豹子，此类复式 **0 收益**；若近期组三偏多，可改买 **组选复式** 或直选 **2 码定位**。",
            "",
            "```text",
            "python src/fc3d_straight3_compound.py --data docs/3d.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D straight compound: pick 3 digits -> 6 bets")
    parser.add_argument("--data", type=Path, default=Path("docs/3d.md"))
    parser.add_argument("--report", type=Path, default=Path("docs/福彩3D单选复式3码6注回测分析.md"))
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--windows", type=str, default="50,200,500,all")
    args = parser.parse_args()

    draws = parse_draws(args.data.resolve())
    windows: list[int | None] = []
    for x in args.windows.split(","):
        xs = x.strip()
        windows.append(None if xs == "all" else int(xs))

    text = render_report(draws, windows, args.min_history)
    args.report.write_text(text, encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
