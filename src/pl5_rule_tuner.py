"""
Holdout hyper-parameter search for PL5 adaptive model.

This does NOT discover new arbitrary "rules" from thin air: it only compares a small
grid of (learning_rate, decay) on the same fixed feature set, scored on the last
`holdout` draws after training only on earlier history (walk-forward on holdout).

Tuning on random lottery series easily overfits noise; use holdout + small grids,
and treat gains as weak evidence unless replicated on a fresh time slice.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pl5_rolling_backtest as pl5  # noqa: E402


def _tail_draws(draws: list[pl5.Draw], tail: int) -> list[pl5.Draw]:
    if tail <= 0 or tail >= len(draws):
        return draws
    return draws[-tail:]


def run_train_then_holdout(
    draws: list[pl5.Draw],
    *,
    holdout: int,
    min_history: int,
    learning_rate: float,
    decay: float,
    ticket_counts: tuple[int, ...],
) -> tuple[list[int], pl5.StrategyState]:
    if holdout <= 0:
        raise ValueError("holdout must be positive.")
    if len(draws) <= min_history + holdout:
        raise ValueError(
            f"Need len(draws) > min_history + holdout, got {len(draws)} vs {min_history}+{holdout}."
        )
    train = draws[:-holdout]
    _, state, _ = pl5.rolling_backtest(
        train,
        min_history=min_history,
        ticket_counts=ticket_counts,
        adaptive=True,
        learning_rate=learning_rate,
        decay=decay,
    )
    ranks: list[int] = []
    for idx in range(len(train), len(draws)):
        rank, _ = pl5.one_roll_step(draws, idx, state, ticket_counts)
        ranks.append(rank)
    return ranks, state


def summarize_ranks(ranks: list[int]) -> dict[str, float | int]:
    n = len(ranks)
    if n == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "top100_rate": 0.0,
            "top1000_rate": 0.0,
            "rank1_hits": 0,
            "min_rank": 0,
        }
    return {
        "mean": statistics.fmean(ranks),
        "median": statistics.median(ranks),
        "top100_rate": sum(1 for r in ranks if r <= 100) / n,
        "top1000_rate": sum(1 for r in ranks if r <= 1000) / n,
        "rank1_hits": sum(1 for r in ranks if r == 1),
        "min_rank": min(ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PL5 holdout grid search (lr, decay).")
    parser.add_argument("--data", type=Path, default=Path("docs/p5.md"))
    parser.add_argument("--holdout", type=int, default=300, help="Last N draws used only for scoring.")
    parser.add_argument("--min-history", type=int, default=pl5.DEFAULT_MIN_HISTORY)
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="If >0, only use the last N draws from the file (faster approximate tuning).",
    )
    parser.add_argument("--quick", action="store_true", help="Smaller candidate grid.")
    parser.add_argument("--out", type=Path, default=None, help="Optional UTF-8 markdown summary path.")
    args = parser.parse_args()

    draws_all = pl5.parse_draws(args.data)
    draws = _tail_draws(draws_all, args.tail)

    if args.quick:
        grid: list[tuple[str, float, float]] = [
            ("baseline", 0.035, 0.997),
            ("lr_high", 0.055, 0.997),
            ("decay_loose", 0.035, 0.999),
        ]
    else:
        grid = []
        for lr in (0.02, 0.035, 0.05):
            for decay in (0.995, 0.997, 0.999):
                grid.append((f"lr{lr}_dc{decay}", lr, decay))

    ticket_counts = (1,)
    rows: list[dict[str, object]] = []
    for label, lr, decay in grid:
        ranks, _ = run_train_then_holdout(
            draws,
            holdout=args.holdout,
            min_history=args.min_history,
            learning_rate=lr,
            decay=decay,
            ticket_counts=ticket_counts,
        )
        stats = summarize_ranks(ranks)
        rows.append(
            {
                "label": label,
                "learning_rate": lr,
                "decay": decay,
                **stats,
            }
        )

    rows.sort(key=lambda r: (r["mean"], r["median"]))  # type: ignore[arg-type]
    best = rows[0]

    lines: list[str] = []
    lines.append("# 排列5 留出段超参搜索（非自动生成新特征）")
    lines.append("")
    lines.append("## 口径说明")
    lines.append(
        "- 仅在**固定特征集**上搜索 `learning_rate` 与 `decay`；不会从数据中自动发明新规则字段。"
    )
    lines.append(
        "- 训练段：`0 .. len-holdout-1` 逐期自适应；留出段：对最后 `holdout` 期做**继续 walk-forward**（每期仍更新权重），仅用留出段的排名指标做比较。"
    )
    lines.append("- 随机序列上指标改善可能是噪声；若某组参数明显更优，应在**另一段未参与调参的留出期**上复核。")
    lines.append("")
    lines.append("## 数据与切分")
    lines.append(f"- 文件：`{args.data}`")
    lines.append(f"- 使用期数：{len(draws)}（全文件 {len(draws_all)}；tail={args.tail or 'all'}）")
    lines.append(f"- holdout：{args.holdout}；min_history：{args.min_history}；grid：{'quick' if args.quick else 'full'}")
    lines.append("")
    lines.append("## 留出段直选一等奖口径（Top1）")
    lines.append(
        "- 当前脚本在留出段每期只取模型**排名第 1** 的一注作为虚拟投注对象；若该期开奖号码的得分排名 **= 1**，"
        "则等价于「买这一注中直选一等奖」（五位全中）。"
    )
    lines.append("- 下表「一等奖次数」即留出期内 `rank == 1` 的期数；「最佳排名」为留出期内开奖号的最靠前名次。")
    lines.append("")
    lines.append("## 结果（按留出段平均排名升序）")
    lines.append("")
    lines.append(
        "| 方案 | lr | decay | 留出均值排名 | 留出中位 | 前100占比 | 前1000占比 | 一等奖次数 | 最佳排名 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['learning_rate']} | {r['decay']} | "
            f"{r['mean']:.1f} | {r['median']:.1f} | {r['top100_rate']*100:.2f}% | {r['top1000_rate']*100:.2f}% | "
            f"{r['rank1_hits']} | {r['min_rank']} |"
        )
    lines.append("")
    lines.append("## 当前推荐（仅相对本切分）")
    lines.append(
        f"- 参数：`learning_rate={best['learning_rate']}`, `decay={best['decay']}`（标签 `{best['label']}`）"
    )
    lines.append(
        f"- 留出段平均排名 **{best['mean']:.1f}** / 100000（越小表示开奖号在模型排序中越靠前，仍不等于提高真实中奖率）。"
    )
    lines.append(
        f"- 留出段直选一等奖（Top1）次数：**{best['rank1_hits']}** / {args.holdout}；最佳排名：**{best['min_rank']}**。"
    )
    lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
