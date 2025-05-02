FROM python:3.10-slim
WORKDIR /app

# 必要なライブラリをインストール
COPY app/script/requirements.txt ./script/requirements.txt
RUN pip install --no-cache-dir -r ./script/requirements.txt

# カスタマイズしたモデルファイルをコピー
COPY app/model/02_trained ./model/02_trained

# APIサーバーのコードをコピー
COPY app/script/03_app.py ./script/03_app.py

# ポートを公開
EXPOSE 8000

# APIサーバーを起動
CMD ["uvicorn", "script.03_app:app", "--host", "0.0.0.0", "--port", "8000"]
