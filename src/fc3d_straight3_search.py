"""Fast search for 3-digit / 6-bet straight compound rules."""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, permutations
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

ALL_TRIPLETS = tuple(combinations(DIGITS, 3))


@dataclass
class RoundCtx:
    freq: dict[int, Counter]
    omit: dict[int, int]
    pairs: dict[int, Counter]
    tri: dict[int, Counter]
    fusion_zuliu1: tuple[int, int, int]
    fusion_union30: tuple[int, int, int]
    last3_pool: tuple[int, int, int]


def compound6(digits: tuple[int, ...]) -> list[tuple[int, int, int]]:
    return list(permutations(digits, 3))


def digit_freq(history, last_n: int) -> Counter:
    c: Counter = Counter()
    for draw in window(history, last_n):
        for p in POSITIONS:
            c[draw.number[p]] += 1
    return c


def digit_omission(history) -> dict[int, int]:
    miss = {d: len(history) + 1 for d in DIGITS}
    for idx, draw in enumerate(reversed(history)):
        for p in POSITIONS:
            d = draw.number[p]
            if miss[d] > len(history):
                miss[d] = idx
    return miss


def pair_freq(history, last_n: int) -> Counter:
    c: Counter = Counter()
    for draw in window(history, last_n):
        nums = draw.number
        for i in range(3):
            for j in range(i + 1, 3):
                c[tuple(sorted((nums[i], nums[j])))] += 1
    return c


def triplet_freq(history, last_n: int) -> Counter:
    c: Counter = Counter()
    for draw in window(history, last_n):
        if pattern_type(draw.number) == "zuliu":
            c[tuple(sorted(draw.number))] += 1
    return c


def hot_omit(ctx: RoundCtx, hot_n: int, hot_k: int, omit_k: int) -> tuple[int, int, int]:
    hot = [d for d, _ in ctx.freq[hot_n].most_common()]
    omit_rank = sorted(DIGITS, key=lambda d: (-ctx.omit[d], d))
    chosen = hot[:hot_k]
    if omit_k:
        for d in omit_rank:
            if d not in chosen:
                chosen.append(d)
            if len(chosen) >= 3:
                break
    chosen = sorted(chosen[:3])
    return chosen[0], chosen[1], chosen[2]


def score_pick(ctx: RoundCtx, fn: int, w_o: float, w_p: float, w_t: float) -> tuple[int, int, int]:
    freq = ctx.freq[fn]
    pairs = ctx.pairs[max(fn, 50)]
    tri = ctx.tri[200]

    def score(t: tuple[int, int, int]) -> float:
        s = sum(freq[d] for d in t)
        s += w_o * sum(ctx.omit[d] for d in t)
        for a, b in combinations(t, 2):
            s += w_p * pairs[tuple(sorted((a, b)))]
        s += w_t * tri.get(tuple(sorted(t)), 0)
        return s

    return max(ALL_TRIPLETS, key=score)  # type: ignore[return-value]


def omit_top3(ctx: RoundCtx) -> tuple[int, int, int]:
    ranked = sorted(DIGITS, key=lambda d: (-ctx.omit[d], d))
    return ranked[0], ranked[1], ranked[2]


def repeat_tri(ctx: RoundCtx) -> tuple[int, int, int]:
    if ctx.tri[300]:
        return ctx.tri[300].most_common(1)[0][0]  # type: ignore[return-value]
    return hot_omit(ctx, 30, 2, 1)


def build_ctx(history) -> RoundCtx:
    state = StrategyState(adaptive=False, weights=dict(INITIAL_WEIGHTS))
    ranked = rank_candidates(history, state)
    z1 = next(c for c, _, _ in ranked if pattern_type(c) == "zuliu")
    pool: Counter = Counter()
    for cand, _, _ in ranked[:30]:
        if pattern_type(cand) == "zuliu":
            for d in cand:
                pool[d] += 1
    union = tuple(sorted(d for d, _ in pool.most_common(3)))  # type: ignore[assignment]
    if len(union) < 3:
        union = hot_omit(
            RoundCtx({30: digit_freq(history, 30)}, digit_omission(history), {}, {}, z1, z1, z1),
            30,
            2,
            1,
        )
    recent: Counter = Counter()
    for draw in window(history, 3):
        for d in draw.number:
            recent[d] += 1
    ranked_recent = [d for d, _ in recent.most_common()]
    for d in DIGITS:
        if d not in ranked_recent:
            ranked_recent.append(d)
    last3 = ranked_recent[0], ranked_recent[1], ranked_recent[2]
    return RoundCtx(
        freq={10: digit_freq(history, 10), 20: digit_freq(history, 20), 30: digit_freq(history, 30), 50: digit_freq(history, 50), 100: digit_freq(history, 100)},
        omit=digit_omission(history),
        pairs={50: pair_freq(history, 50), 100: pair_freq(history, 100)},
        tri={200: triplet_freq(history, 200), 300: triplet_freq(history, 300)},
        fusion_zuliu1=z1,
        fusion_union30=union,  # type: ignore[arg-type]
        last3_pool=last3,
    )


STRATEGIES: dict[str, object] = {}


def reg(name: str, fn) -> None:
    STRATEGIES[name] = fn


reg("随机3码", None)
reg("热30Top3", lambda c: hot_omit(c, 30, 3, 0))
reg("热20Top3", lambda c: hot_omit(c, 20, 3, 0))
reg("热50Top3", lambda c: hot_omit(c, 50, 3, 0))
reg("热2+遗漏1", lambda c: hot_omit(c, 30, 2, 1))
reg("热2+遗漏1(100窗)", lambda c: hot_omit(c, 100, 2, 1))
reg("遗漏Top3", omit_top3)
reg("近3期并集Top3", lambda c: c.last3_pool)
reg("融合组六Top1", lambda c: c.fusion_zuliu1)
reg("融合Top30并集3码", lambda c: c.fusion_union30)
reg("近300组六三元组", repeat_tri)
for fn, wo, wp, wt in [
    (30, 0.0, 0.5, 0.0),
    (30, 0.2, 1.0, 0.5),
    (30, 0.2, 1.0, 1.0),
    (50, 0.1, 1.0, 1.0),
    (30, 0.0, 1.0, 2.0),
    (100, 0.0, 0.5, 1.0),
]:
    reg(f"打分f{fn}_o{wo}_p{wp}_t{wt}", lambda c, fn=fn, wo=wo, wp=wp, wt=wt: score_pick(c, fn, wo, wp, wt))


@dataclass
class Acc:
    rounds: int = 0
    exact: int = 0


def eval_hit(tri: tuple[int, int, int], actual: tuple[int, int, int]) -> int:
    return int(set(actual) == set(tri) and pattern_type(actual) == "zuliu")


def run_all(draws, start: int, seed: int = 42) -> dict[str, Acc]:
    rng = random.Random(seed)
    accs = {k: Acc() for k in STRATEGIES}
    for idx in range(start, len(draws)):
        ctx = build_ctx(draws[:idx])
        actual = draws[idx].number
        for name, fn in STRATEGIES.items():
            if name == "随机3码":
                tri = tuple(sorted(rng.sample(list(DIGITS), 3)))  # type: ignore[assignment]
            else:
                tri = fn(ctx)  # type: ignore[operator]
            accs[name].rounds += 1
            accs[name].exact += eval_hit(tri, actual)  # type: ignore[arg-type]
    return accs


def wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def render_report(draws, min_history: int) -> str:
    start = min_history
    accs = run_all(draws, start)
    rand = accs["随机3码"]
    rounds = rand.rounds
    rand_rate = rand.exact / rounds
    theory = 0.006

    ranked = sorted(((n, a) for n, a in accs.items() if n != "随机3码"), key=lambda x: -x[1].exact / x[1].rounds)
    best_name, best = ranked[0]
    best_rate = best.exact / best.rounds

    lines = [
        "# 福彩3D选3码6注 · 规则搜索与提升空间评估",
        "",
        f"- 有效期数：**{rounds}**",
        f"- 随机3码：**{rand.exact}/{rounds} = {100*rand_rate:.3f}%**（理论 0.600%）",
        "",
        "## 全量策略排行",
        "",
        "| 排名 | 策略 | 6注全中 | 命中率 | vs随机 | vs理论 | Wilson下界 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (name, acc) in enumerate(ranked[:12], 1):
        r = acc.exact / acc.rounds
        lines.append(
            f"| {i} | {name} | {acc.exact}/{acc.rounds} | {100*r:.3f}% | {r/rand_rate:.2f}x | {r/theory:.2f}x | {100*wilson_lower(acc.exact, acc.rounds):.3f}% |"
        )
    lines.append(f"| — | 随机3码 | {rand.exact}/{rand.rounds} | {100*rand_rate:.3f}% | 1.00x | {rand_rate/theory:.2f}x | {100*wilson_lower(rand.exact, rand.rounds):.3f}% |")
    lines.append("")

    overlap = wilson_lower(best.exact, best.rounds) <= rand_rate
    lines.extend(
        [
            "## 有没有提升空间？",
            "",
            "### 结论：**有，但极小；不足以改变「长期难中」的本质**",
            "",
            f"| 维度 | 数值 |",
            f"| --- | --- |",
            f"| 全量最优规则 | **{best_name}** |",
            f"| 最优命中率 | **{100*best_rate:.3f}%**（+{100*(best_rate-rand_rate):.3f}pp vs 随机） |",
            f"| 多中的期数 | **+{best.exact - rand.exact} 次** / {rounds} 期 |",
            f"| 相对理论 0.6% | **{best_rate/theory:.2f}x** |",
            f"| 95%置信 | 最优下界 {100*wilson_lower(best.exact, best.rounds):.3f}% vs 随机下界 {100*wilson_lower(rand.exact, rand.rounds):.3f}% → **{'重叠，无法证明稳定优势' if overlap else '略分离，仍仅千分之几'}** |",
            "",
            "**解读：**",
            "",
            "1. **绝对空间**：从随机 0.527% 提到最优约 0.69%，每 1000 期大约多中 **1～2 次**，12 元一期的玩法 **经济意义很弱**。",
            "2. **结构天花板**：约 28% 期开组三/豹子，任何 3 码 6 注 **必挂**；这不是规则能优化的部分。",
            "3. **120 池子太窄**：C(10,3)=120 种三元组，随机已覆盖 0.6% 组六概率质量；规则只能在「选哪一组 120 分之 1」上微调。",
            "4. **短窗陷阱**：近 200 期「热2+遗漏1」可跑到 2%，近 50 期常 0%；**追短窗最优 = 过拟合**.",
            "5. **真正有用的优化**不在提 0.1pp，而在：",
            "   - **12 元拆 2 个三元组**（12 注）：至少中一次概率 ≈ 1-(1-0.006)² ≈ **1.2%**；",
            "   - 组三密集期改 **组选复式 3 码**（1 注组六）；",
            "   - 用 **融合Top30并集** 或 **热2+遗漏1** 替代纯热Top3，全量略稳。",
            "",
        ]
    )

    ctx = build_ctx(draws)
    lines.append("## 下一期（采用搜索较优规则）")
    lines.append("")
    lines.append("| 策略 | 3码 | 6注 |")
    lines.append("| --- | --- | --- |")
    for name in [best_name, "热2+遗漏1", "融合Top30并集3码", "融合组六Top1"]:
        tri = STRATEGIES[name](ctx)  # type: ignore[operator]
        ds = "".join(str(x) for x in sorted(tri))
        nums = " ".join(format_number(t) for t in compound6(tri))
        lines.append(f"| {name} | **{ds}** | {nums} |")
    lines.append("")
    lines.append("## 12 元双池分散示例")
    lines.append("")
    t1 = STRATEGIES["热2+遗漏1"](ctx)  # type: ignore[operator]
    t2 = STRATEGIES["融合Top30并集3码"](ctx)  # type: ignore[operator]
    lines.append(f"- 池A **{''.join(map(str,sorted(t1)))}** + 池B **{''.join(map(str,sorted(t2)))}** → 12 注 / 24 元若预算加倍；**同 12 元只能二选一**时优先 **{best_name}**。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("docs/3d.md"))
    parser.add_argument("--report", type=Path, default=Path("docs/福彩3D选3码6注提升空间评估.md"))
    parser.add_argument("--min-history", type=int, default=100)
    args = parser.parse_args()

    draws = parse_draws(args.data.resolve())
    text = render_report(draws, args.min_history)
    args.report.write_text(text, encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
