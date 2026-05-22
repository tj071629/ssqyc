"""Export PL5 model top-100 ranked tickets to docs/排列5模型前100候选.md."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import pl5_rolling_backtest as m  # noqa: E402

OUT = ROOT / "docs" / "排列5模型前100候选.md"
TOP_N = 100


def main() -> None:
    draws = m.parse_draws(ROOT / "docs" / "p5.md")
    _, state, _ = m.rolling_backtest(draws, ticket_counts=(1,))
    order, _, scores = m.rank_for_round(draws, state)
    last = draws[-1]

    lines: list[str] = [
        "# 排列5 模型得分前100注",
        "",
        "口径：全历史逐期自适应学习后的权重，对截至末期的全部历史重算得分并排序（与 `pl5_rolling_backtest.py` 一致）。",
        "",
        f"- 数据末条：{last.issue}（{last.date}），开奖 **{last.text}**",
        f"- 导出时间：按脚本运行时刻生成",
        "",
        "| 排名 | 号码 | 分数 |",
        "| ---: | --- | ---: |",
    ]
    for rank in range(TOP_N):
        idx = int(order[rank])
        num = m.format_number(tuple(int(m._DIGIT_MATRIX[idx, p]) for p in m.POSITIONS))
        lines.append(f"| {rank + 1} | {num} | {float(scores[idx]):.4f} |")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {TOP_N} lines to {OUT}")


if __name__ == "__main__":
    main()
