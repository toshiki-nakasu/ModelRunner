# modelRunnerSample

## 実行可能なタスク

- 事前学習済みモデルをダウンロード
- モデルを転移学習
- FastAPIアプリ実行 もしくは モデル推論用コンテナの実行 で学習したモデルを起動

詳細は`.vscode/tasks.json`を参照すること

## 開発環境

devcontainerで開発環境を構築することが可能になっている

### 開発コンテナを使用しない場合

基本的には必要なものはないがGPUを使う場合、以下が必要 ([開発コンテナを使用する場合](#開発コンテナを使用する場合)はイメージタグ指定のみでOK)

1. CUDA Toolkit
    **今のToolkitはCUDAを含む**
    1. [WSLの対応](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2)
    2. [インストール](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
    3. 環境変数を追加

        ```bash
        cat <<"EOF" >> ~/.bashrc
        export PATH=${PATH}:/usr/local/cuda/bin
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
        EOF
        source ~/.bashrc
        ```

    4. インストールできたか確認
        `nvcc -V`

2. cuDNN (必須ではないが、ディープラーニングに活用可能)
    [archive](https://developer.nvidia.com/rdp/cudnn-archive): アカウントログインが必要

3. 現在GPUやcuDNNが使えるかを確認するスクリプトを用意しています
    `python3 test/check_gpu/check_gpu.py`

- インストールしたものをリセットしたい場合

    ```bash
    sudo apt-get purge -y nvidia-*
    sudo apt-get purge -y cuda-*
    ```

### 開発コンテナを使用する場合

開発コンテナは`nvidia/cuda`を使用するため、ホストマシンに以下のインストールが必要

- Nvidia Container Toolkit
    [リファレンス](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 動作手順

3.以降の手順はそれぞれ`.vscode/tasks.json`に登録済み

1. 事前学習済みモデルを指定
    1. `cp .env.sample .env`
    2. `.env`ファイルの`MODEL_NAME`変数を[huggingface](https://huggingface.co/models)から任意のモデル名にする
        リポジトリには含まれません
2. 転移学習データセットを用意
    1. `app/dataset/*.jsonl`を作成
        リポジトリには含まれません
    2. `sample.jsonl`はサンプル用なので不要になれば削除すること
3. Pythonスクリプト実行に必要なライブラリをインストールする (devcontainerの場合はコンテナ起動時にインストール済み)
    `pip3 install -r app/script/requirements.txt`
4. 事前学習済みモデルをローカルにダウンロード
    1. `python3 app/script/main/01_download.py`
    2. `app/model/01_download`に保存される
        リポジトリには含まれません
5. 転移学習を実行
    1. `python3 app/script/main/02_train.py`
    2. `app/model/02_trained`に保存される
        リポジトリには含まれません
6. 学習したモデルをFastAPIアプリとして実行
    1. `uvicorn app.script.main.03_app:app --reload --host 0.0.0.0 --port 8000`
    2. スクリプトの引数にこちらからのテキストを入力し、POSTリクエスト
        `bash test/model_post/model_post.sh test-text`
    3. 推論結果がresponseで返される
7. モデルが起動するDockerイメージをビルド
    1. `docker build -f app/Dockerfile -t my-custom-llm .`
    2. ローカルのDockerHubにイメージがpushされる
8. モデル推論用コンテナの実行
    1. `docker run -p 8000:8000 my-custom-llm`
    2. ポートは同じなので6.のテスト用スクリプトで動作確認が可能

---

## メモ

### バージョンの確認方法

よくあるCUDA, Toolkit, PyTorchなどのバージョン確認方法をメモ

1. CUDA
    [wiki](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
    1. `Compute capability, GPU semiconductors and Nvidia GPU board products`の表から`Micro-architecture`を特定
    2, `Compute capability (CUDA SDK support vs. microarchitecture)`の表から`CUDA SDK version(s)`を特定 (緑の範囲が対応範囲)

2. CUDA Toolkit
    [Release](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
    特定したCUDAのバージョンからNvidia Driverの必要なバージョンを特定

### 類似のfeaturesについて

devcontainerを活用する際に`ghcr.io/devcontainers/features/nvidia-cuda:1`が候補に挙がるが、現在はNVIDIAがチューニングしている`NVIDIA Docker`を使う必要がある。

通常のDockerイメージだと、ホストからクライアントにGPUの引き渡しができない。(`--gpus=all`がエラーになる)
