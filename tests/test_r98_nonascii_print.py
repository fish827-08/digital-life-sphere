"""R98 纪律：**非 ASCII print 全域守卫**（F-R10 / F-R14 / F-R15 同族）。

背景：中文 Windows 默认 GBK 控制台下，`print()` 里出现 **GBK 无法编码**的字符
（`↻` U+21BB、emoji ✅❌🔴、`⇒` U+21D2 …）会抛 `UnicodeEncodeError` ⇒ 脚本 **rc=1**
⇒ 批次状态/退出码被污染（**外围把真结果搅坏**，而数据其实无损）。
已发生实例：`a4_verify_capacity.py` 续跑路径 `print("  ↻ …")`（F-R15，内评代修）。

本测试把"扫描"变成**不可回退的纪律**：受控目录下不允许存在
「print 含 GBK 不可编码字符」**且**「未做 UTF-8 兜底」的脚本。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_scanner():
    path = ROOT / "experiments" / "scan_nonascii_print.py"
    spec = importlib.util.spec_from_file_location("scan_nonascii_print", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scan_nonascii_print"] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load_scanner()


def test_no_unprotected_non_gbk_print_in_controlled_tree():
    offenders = []
    for d in sc.DEFAULT_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            src = p.read_text(encoding="utf-8")
            if sc.PROTECT_MARK in src:      # 入口已 reconfigure ⇒ 运行期安全
                continue
            bad = sc.scan_file(p)
            if bad:
                offenders.append((p.relative_to(ROOT).as_posix(),
                                  sorted({c for _, c, _ in bad})))
    assert not offenders, (
        "以下脚本在 GBK 控制台下会 rc=1 假失败（print 含 GBK 不可编码字符且无兜底）：\n"
        + "\n".join(f"  {f}: {chars}" for f, chars in offenders)
        + "\n修法：入口加 sys.stdout.reconfigure(encoding='utf-8', errors='replace')"
    )


def test_scanner_detects_gbk_unencodable_and_ignores_chinese(tmp_path):
    """扫描器自身的行为：中文（GBK 可编码）不得报，emoji/箭头必须报。"""
    good = tmp_path / "good.py"
    good.write_text('print("中文没问题 ⇒ 这个不行")\n', encoding="utf-8")
    assert any(c == "⇒" for _, c, _ in sc.scan_file(good))
    assert not any(c == "中" for _, c, _ in sc.scan_file(good))

    ok = tmp_path / "ok.py"
    ok.write_text('print("纯中文与 ASCII 都安全")\n', encoding="utf-8")
    assert sc.scan_file(ok) == []
