import re
from pathlib import Path

target = (2, 6, 0, 3, 0, 0, 11)
rows = []
for line in Path("docs/7xc.md").read_text(encoding="utf-8").splitlines():
    m = re.match(r"\| (\d+) \| ([\d-]+) \| (.+?) \|", line.strip())
    if m:
        rows.append((m.group(1), m.group(2), tuple(int(x) for x in m.group(3).split())))


def mc(a, b):
    return sum(x == y for x, y in zip(a, b))


print("号码:", " ".join(map(str, target[:6])), f"+{target[6]:02d}")
print("7位完全相同:", sum(1 for r in rows if r[2] == target), "次")
print("前6位完全相同:", sum(1 for r in rows if r[2][:6] == target[:6]), "次")
best = sorted(((mc(target, n), iss, dt, n) for iss, dt, n in rows), reverse=True)
print("历史最高同位重复:", best[0][0], "/7")
print("\n最接近的期次:")
for c, iss, dt, n in best[:10]:
    pos = [str(i + 1) for i in range(7) if target[i] == n[i]]
    nums = " ".join(map(str, n))
    print(f"  {c}/7  {iss} ({dt})  {nums}  重复位置: {','.join(pos) if pos else '无'}")
