FROM python:3.10-slim
WORKDIR ${APP_DIR}

# 環境変数定義ファイルをコピー
COPY .env .env

# 必要なライブラリをインストール
COPY resources/requirements.txt script/requirements.txt
RUN pip install --no-cache-dir -r script/requirements.txt

# カスタマイズしたモデルファイルをコピー
COPY resources/model/01_download resources/model/01_download
COPY resources/model/02_trained resources/model/02_trained

# APIサーバーのコードをコピー
COPY script/main/03_app.py script/main/03_app.py

# ポートを公開
EXPOSE 8000

# APIサーバーを起動
CMD ["python3", "script/main/03_app.py"]
