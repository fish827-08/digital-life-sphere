"""R98 纪律：**非 ASCII print 全域扫描** —— 找出会在 GBK 控制台炸掉 `print` 的字符。

背景（F-R15 / F-R10 / F-R14 同族）
--------------------------------
`print()` 里只要出现 **GBK 无法编码**的字符（如 `↻` U+21BB、emoji ✅❌🔴），
在中文 Windows 默认 GBK 控制台下就抛 `UnicodeEncodeError` ⇒ 脚本 **rc=1**
⇒ 批次状态/退出码被污染（**外围把真结果搅坏**，而数据其实无损）。
已发生：`a4_verify_capacity.py` 续跑路径 `print("  ↻ 从快照续跑…")`（内评代修）。

口径
----
判据 = **GBK 不可编码**（不是"非 ASCII"）：中文在 GBK 里没问题，不必报。
另外区分两类：
  - **受保护**：文件入口已 `sys.stdout.reconfigure(encoding="utf-8", …)`（如 a4）⇒ 运行期安全；
  - **未保护**：会真的炸。
用法
----
    python experiments/scan_nonascii_print.py            # 全受控目录
    python experiments/scan_nonascii_print.py --dirs simulation experiments
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = ("simulation", "observatory", "experiments", "world", "core", "tests",
                "persistence")
PROTECT_MARK = 'reconfigure(encoding="utf-8"'


def _print_strings(tree: ast.AST):
    """产出 (行号, 字符串) —— 所有 `print(...)` 调用里的字符串常量（含 f-string 片段）。"""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                yield getattr(sub, "lineno", node.lineno), sub.value


def scan_file(p: Path) -> list[tuple[int, str, str]]:
    """返回 [(行号, 字符, 说明)] —— 该文件 `print` 里 **GBK 不可编码**的字符。"""
    try:
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [(0, "?", f"解析失败：{exc}")]
    protected = PROTECT_MARK in src
    out: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str]] = set()
    for lineno, s in _print_strings(tree):
        for ch in s:
            try:
                ch.encode("gbk")
            except UnicodeEncodeError:
                key = (lineno, ch)
                if key in seen:
                    continue
                seen.add(key)
                out.append((lineno, ch, f"U+{ord(ch):04X}"
                            + ("（有 reconfigure ⇒ 运行期受保护）" if protected else "⚠️ 未保护")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=list(DEFAULT_DIRS))
    args = ap.parse_args()
    n_files = n_off = n_prot = 0
    offenders: list[tuple[str, int, str, str]] = []
    for d in args.dirs:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            n_files += 1
            bad = scan_file(p)
            if not bad:
                continue
            rel = p.relative_to(ROOT).as_posix()
            prot = PROTECT_MARK in p.read_text(encoding="utf-8")
            if prot:
                n_prot += 1
            else:
                n_off += 1
            for lineno, ch, note in bad:
                offenders.append((rel, lineno, ch, note))
    print("=" * 88)
    print(f"非 ASCII print 扫描：{n_files} 个 .py")
    print(f"  含 GBK 不可编码 print 的文件：{n_off} 个**未保护** / {n_prot} 个已 reconfigure")
    print("=" * 88)
    if not offenders:
        print("✅ 无非 ASCII print 隐患")
    for rel, lineno, ch, note in offenders:
        print(f"  {rel}:{lineno}  {ch!r} {note}")
    print("-" * 88)
    print("建议：① 受保护文件（入口 reconfigure）无需改；② 未保护文件二选一："
          "换成 ASCII 字符，或入口加同样的 reconfigure 兜底。")
    return 1 if n_off else 0


if __name__ == "__main__":
    raise SystemExit(main())
