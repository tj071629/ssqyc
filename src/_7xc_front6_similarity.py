"""Analyze front-6 position similarity for 7xc historical draws."""
import re
from collections import Counter
from math import comb

path = r"D:\001dfgx\ssqyc\docs\7xc.md"
rows = []
with open(path, encoding="utf-8") as f:
    for line in f:
        m = re.match(r"\| (\d+) \| ([\d-]+) \| (.+?) \|", line.strip())
        if m:
            nums = [int(x) for x in m.group(3).split()]
            rows.append(
                {
                    "issue": m.group(1),
                    "date": m.group(2),
                    "all7": nums,
                    "front6": nums[:6],
                }
            )


def match_count(a, b):
    return sum(x == y for x, y in zip(a, b))


def fmt(nums):
    return " ".join(map(str, nums))


best_match_dist = Counter()
exact6_count = 0
recent_examples = []

for i in range(1, len(rows)):
    cur = rows[i]
    best = 0
    best_prior = None
    for j in range(i):
        mc = match_count(cur["front6"], rows[j]["front6"])
        if mc > best:
            best = mc
            best_prior = rows[j]
    best_match_dist[best] += 1
    if best == 6:
        exact6_count += 1
    if i >= len(rows) - 5:
        recent_examples.append((best, cur, best_prior))

win50 = Counter()
for i in range(1, len(rows)):
    cur = rows[i]["front6"]
    start = max(0, i - 50)
    best = max(match_count(cur, rows[j]["front6"]) for j in range(start, i))
    win50[best] += 1

print("=== 七星彩前6位：每期 vs 之前全部期次的最高相似度分布 ===")
print("(相似度 = 相同位置相同数字的个数，范围 0~6)")
print("总期数(可比较):", len(rows) - 1)
print("前6位完全相同(6/6)出现次数:", exact6_count)
print()
for k in range(6, -1, -1):
    c = best_match_dist[k]
    pct = c / (len(rows) - 1) * 100
    bar = "#" * int(pct / 2)
    print(f"  {k}/6: {c:4d} 期 ({pct:5.2f}%) {bar}")

print()
print("=== 近50期内最高相似度分布 ===")
for k in range(6, -1, -1):
    c = win50[k]
    pct = c / (len(rows) - 1) * 100
    print(f"  {k}/6: {c:4d} 期 ({pct:5.2f}%)")

print()
print("=== 最近5期：与历史最高相似的前6位对照 ===")
for best, cur, prior in recent_examples:
    print(
        f"  {cur['issue']} {fmt(cur['front6'])}"
        f"  <-最高{best}/6->"
        f"  {prior['issue']}({prior['date']}) {fmt(prior['front6'])}"
    )

target = [2, 6, 0, 3, 0, 1]
print()
print("=== 示例：260301(前6位) 与历史各期相似度 ===")
matches = []
for r in rows:
    mc = match_count(target, r["front6"])
    if mc >= 4:
        matches.append((mc, r["issue"], r["date"], r["front6"]))
matches.sort(reverse=True)
print(">=4/6 的期数:", len(matches))
for mc, iss, dt, nums in matches[:8]:
    print(f"  {mc}/6  {iss} {dt}  {fmt(nums)}")

print()
print("=== 随机基准(独立均匀0-9，每位匹配概率1/10) ===")
for k in range(6, -1, -1):
    p = comb(6, k) * (0.1**k) * (0.9 ** (6 - k))
    print(f"  恰好{k}/6相同: {p * 100:5.2f}%")
