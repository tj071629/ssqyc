"""Replay PL5 model for recent issues (single walk-forward pass)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import pl5_rolling_backtest as m  # noqa: E402

TARGETS = ("26124", "26125", "26126", "26127", "26128", "26129", "26130")


def main() -> None:
    draws = m.parse_draws(Path("docs/p5.md"))
    state = m.StrategyState(
        adaptive=True,
        weights=dict(m.INITIAL_WEIGHTS),
        learning_rate=0.035,
        decay=0.997,
    )
    rows: list[dict[str, object]] = []

    for idx in range(100, len(draws)):
        issue = draws[idx].issue
        if issue not in TARGETS:
            m.one_roll_step(draws, idx, state, (1,))
            continue

        history = draws[:idx]
        actual = draws[idx]
        order, _, scores = m.rank_for_round(history, state)
        ai = m._ticket_index(actual.number)
        rank = int(np.where(order == ai)[0][0]) + 1
        top1_i = int(order[0])
        top1 = m.format_number(tuple(int(m._DIGIT_MATRIX[top1_i, p]) for p in m.POSITIONS))
        st = m.stratified_three_entries(order, scores, pool=100)
        st_nums = [str(r["number"]) for r in st]
        rows.append(
            {
                "issue": issue,
                "date": actual.date,
                "actual": actual.text,
                "rank": rank,
                "top1": top1,
                "in100": rank <= 100,
                "hit3": actual.text in st_nums,
                "strat3": ",".join(st_nums),
            }
        )
        m.one_roll_step(draws, idx, state, (1,))

    print("=== 排列5 最近7期 · 当期模型得分排序复盘 ===")
    print("issue | date | actual | rank | top1 | 进前100 | 分段3注中 | 当时分段3注")
    for r in rows:
        print(
            f"{r['issue']} | {r['date']} | {r['actual']} | {r['rank']} | {r['top1']} | "
            f"{'是' if r['in100'] else '否'} | {'是' if r['hit3'] else '否'} | {r['strat3']}"
        )

    hit100 = sum(1 for r in rows if r["in100"])
    hit3 = sum(1 for r in rows if r["hit3"])
    print()
    print(f"7期内：开奖进模型前100 → {hit100}/7；分段3注全中 → {hit3}/7；直选全中(rank=1) → "
          f"{sum(1 for r in rows if r['rank'] == 1)}/7")


if __name__ == "__main__":
    main()
