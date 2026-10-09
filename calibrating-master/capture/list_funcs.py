import re

path = r"E:\MVviewer\MV Viewer\Development\Samples\Python\IMV\MVSDK\IMVApi.py"

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

print("=== 所有 IMV_Set 开头的函数 ===")
for line in content.splitlines():
    line = line.strip()
    if line.startswith("def IMV_Set"):
        print(line)

print("\n=== 所有 IMV_Get 开头的函数 ===")
for line in content.splitlines():
    line = line.strip()
    if line.startswith("def IMV_Get"):
        print(line)