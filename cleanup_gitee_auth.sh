#!/usr/bin/env bash
# cleanup_gitee_auth.sh — 实验完成后清理：删除凭证 + 设回私有
# 用法: ./cleanup_gitee_auth.sh <解密密码>

set -e

if [ -z "$1" ]; then
    echo "用法: $0 <解密密码>"
    exit 1
fi

PASSWORD="$1"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENC_FILE="$REPO_DIR/.gitee_token.enc"

# 解密 token（用于调用 API 设回私有）
TOKEN=$(cat "$ENC_FILE" | openssl enc -d -aes-256-cbc -pbkdf2 -pass pass:"$PASSWORD" -base64 2>/dev/null)
if [ -z "$TOKEN" ]; then
    echo "错误: 解密失败"
    exit 1
fi

cd "$REPO_DIR"

echo "1/3 把仓库设回私有..."
curl -s -X PUT "https://gitee.com/api/v5/repos/little-fishy/digital-life-sphere" \
    -H "Content-Type: application/json" \
    -d "{\"access_token\":\"${TOKEN}\",\"private\":true}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  仓库状态: private={d.get(\"private\")}')" 2>/dev/null || echo "  (API调用失败，请手动设回私有)"

echo "2/3 删除加密凭证文件..."
rm -f "$ENC_FILE"
git rm --cached "$ENC_FILE" 2>/dev/null || true
echo "  已删除 .gitee_token.enc"

echo "3/3 清除 remote 中的 token..."
git remote set-url gitee "https://gitee.com/little-fishy/digital-life-sphere.git"
echo "  已清除"

echo ""
echo "✅ 清理完成"
echo "⚠️  请确认: gitee 仓库已设为私有，.gitee_token.enc 已从仓库删除"
