"""福彩3D rolling backtest — reuses P3 engine (same 000-999 direct play)."""

from __future__ import annotations

import argparse
from pathlib import Path

from p3_rolling_backtest import (
    DEFAULT_MIN_HISTORY,
    DEFAULT_TICKET_COUNTS,
    INITIAL_WEIGHTS,
    group_report,
    parse_draws,
    report,
    rolling_backtest,
)


def fc3d_report(
    draws,
    metrics,
    state,
    next_ranked,
    *,
    ticket_counts=DEFAULT_TICKET_COUNTS,
) -> str:
    text = report(draws, metrics, state, next_ranked, ticket_counts=ticket_counts)
    return (
        text.replace("# 排列3逐期自适应滚动回测报告", "# 福彩3D逐期自适应滚动回测报告")
        .replace("# 排列3固定权重滚动回测报告", "# 福彩3D固定权重滚动回测报告")
        .replace("排列3直选理论中奖率", "福彩3D直选理论中奖率")
    )


def fc3d_group_report(draws, metrics, *, ticket_counts=DEFAULT_TICKET_COUNTS, recent_n: int = 50) -> str:
    text = group_report(draws, metrics, ticket_counts=ticket_counts, recent_n=recent_n)
    return text.replace("# 排列3组选滚动回测报告", "# 福彩3D组选滚动回测报告")


def main() -> None:
    parser = argparse.ArgumentParser(description="福彩3D rolling backtest.")
    parser.add_argument("--data", type=Path, default=Path("docs/3d.md"))
    parser.add_argument("--report", type=Path, default=Path("docs/福彩3D固定权重滚动回测报告.md"))
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument("--fixed", action="store_true", help="Disable online adaptive weight updates.")
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--group-report", type=Path, default=Path("docs/福彩3D组选滚动回测报告.md"))
    args = parser.parse_args()

    draws = parse_draws(args.data)
    metrics, state, next_ranked = rolling_backtest(
        draws,
        min_history=args.min_history,
        adaptive=not args.fixed,
        learning_rate=args.learning_rate,
    )
    text = fc3d_report(draws, metrics, state, next_ranked)
    args.report.write_text(text, encoding="utf-8")
    args.group_report.write_text(fc3d_group_report(draws, metrics), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
