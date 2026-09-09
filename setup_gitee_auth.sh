#!/usr/bin/env bash
# setup_gitee_auth.sh — 解密 gitee token 并配置 remote 推送权限
# 用法: ./setup_gitee_auth.sh <解密密码>
# 注意: 用完后立即执行 ./cleanup_gitee_auth.sh 删除凭证并设回私有

set -e

if [ -z "$1" ]; then
    echo "用法: $0 <解密密码>"
    exit 1
fi

PASSWORD="$1"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENC_FILE="$REPO_DIR/.gitee_token.enc"

if [ ! -f "$ENC_FILE" ]; then
    echo "错误: 找不到加密文件 $ENC_FILE"
    exit 1
fi

# 解密 token
TOKEN=$(cat "$ENC_FILE" | openssl enc -d -aes-256-cbc -pbkdf2 -pass pass:"$PASSWORD" -base64 2>/dev/null)
if [ -z "$TOKEN" ]; then
    echo "错误: 解密失败，密码不正确"
    exit 1
fi

echo "解密成功，配置 gitee remote..."

cd "$REPO_DIR"

# 配置 gitee remote（带 token）
git remote set-url gitee "https://little-fishy:${TOKEN}@gitee.com/little-fishy/digital-life-sphere.git"

# 验证配置（不显示 token）
git remote -v | sed "s/${TOKEN}/***TOKEN***隐藏***/g"

echo ""
echo "✅ gitee 推送权限已配置"
echo "⚠️  实验完成后，请立即执行: ./cleanup_gitee_auth.sh"
echo "   （删除凭证文件 + 把仓库设回私有）"
