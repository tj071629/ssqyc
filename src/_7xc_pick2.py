"""Generate 2 picks with exactly 4/7 historical position match."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qxc_rolling_backtest import (
    INITIAL_WEIGHTS,
    build_position_scores,
    generate_candidates,
    parse_draws,
    pos_hot30_ticket,
    ticket_score,
)


def match_count(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(x == y for x, y in zip(a, b))


def fmt(t: tuple[int, ...]) -> str:
    return " ".join(str(x) for x in t[:6]) + f" +{t[6]:02d}"


def best_hist_match(ticket: tuple[int, ...], draws) -> tuple[int, object]:
    best = (0, draws[0])
    for d in draws:
        mc = match_count(ticket, d.number)
        if mc > best[0]:
            best = (mc, d)
    return best


def main() -> None:
    draws = parse_draws(Path("docs/7xc.md"))
    weights = dict(INITIAL_WEIGHTS)
    ps, bs = build_position_scores(draws, weights)
    cands = generate_candidates(ps, bs, top_per_pos=4, top_back=5)
    hot = pos_hot30_ticket(draws)
    last = draws[-1]

    pool: list[tuple[float, tuple[int, ...], str, str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()

    def add(ticket: tuple[int, ...]) -> None:
        if ticket in seen:
            return
        seen.add(ticket)
        mc, ref = best_hist_match(ticket, draws)
        if mc != 4:
            return
        sc = ticket_score(ticket[:6], ticket[6], ps, bs)
        pool.append((sc, ticket, ref.issue, ref.date, ref.number))

    for c in cands[:800]:
        add(c[0] + (c[1],))

    for ref in draws[-40:]:
        refn = ref.number
        for change_pos in itertools.combinations(range(7), 3):
            t = list(refn)
            ok = True
            for p in change_pos:
                if p < 6:
                    order = sorted(range(10), key=lambda d: -ps[p][d])
                else:
                    order = sorted(range(15), key=lambda d: -bs[d])
                replaced = False
                for d in order:
                    if d != refn[p]:
                        t[p] = d
                        replaced = True
                        break
                if not replaced:
                    ok = False
                    break
            if ok:
                add(tuple(t))

    pool.sort(key=lambda x: -x[0])
    picked: list[tuple[float, tuple[int, ...], str, str, tuple[int, ...]]] = []
    for row in pool:
        t = row[1]
        if any(match_count(t, p[1]) >= 6 for p in picked):
            continue
        picked.append(row)
        if len(picked) == 2:
            break

    print(f"上期 {last.issue} ({last.date}): {fmt(last.number)}")
    print(f"热号30: {fmt(hot)}")
    top = cands[0][0] + (cands[0][1],)
    print(f"模型Top1: {fmt(top)} score={cands[0][2]:.3f}")
    print()
    for i, (sc, t, iss, dt, rn) in enumerate(picked, 1):
        same = [str(j + 1) for j in range(7) if t[j] == rn[j]]
        print(f"推荐第{i}注: {fmt(t)}")
        print(f"  模型分 {sc:.3f} | 与 {iss}期({dt}) 同位重复4码: 位置 {','.join(same)}")
        print(f"  参照开奖: {fmt(rn)}")
        print()


if __name__ == "__main__":
    main()
