#!/bin/bash

# ヘルプメッセージ表示関数
show_help() {
    echo "使用法: $0 <テキスト>"
    echo "AIモデルでテキスト生成します"
    exit 1
}

# 引数チェック
if [ $# -ne 1 ]; then
    show_help
fi

# 入力テキスト
INPUT_TEXT="$1"

# APIエンドポイント
URL="http://0.0.0.0:8000/generate/"

# POSTリクエスト送信
response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$INPUT_TEXT\"}" \
    $URL)

# レスポンス表示
echo "$response"
