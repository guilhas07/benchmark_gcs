import glob
from collections import defaultdict
import json


graal = "GraalVM"
oracle = "HotSpot"
d = defaultdict(set)
for f in glob.glob("*.json"):
    if graal in f:
        d[graal].add(f.replace(graal, ""))
    elif oracle in f:
        d[oracle].add(f.replace(oracle, ""))


print(f"{len(d[graal])} == {len(d[oracle])}")
# print(f"{json.dumps(d[graal])} {len(d[graal])}")
# print(d[oracle])

assert len(d[graal]) == len(d[oracle])
assert d[graal] == d[oracle]
