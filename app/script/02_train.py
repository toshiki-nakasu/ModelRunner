from datasets import load_dataset
import os
import pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def check_gpu():
    """
    GPUの利用可能性と情報を確認して表示します

    Returns:
        bool: GPUが利用可能かどうか
    """
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"利用可能なGPU数: {gpu_count}")

        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} メモリ: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

        if hasattr(torch.backends, 'cudnn'):
            print(f"CuDNN version: {torch.backends.cudnn.version()}")
            print(f"CuDNN enabled: {torch.backends.cudnn.enabled}")

            # cuDNNの設定を最適化
            torch.backends.cudnn.benchmark = True
            print(f"CuDNN benchmark mode: {torch.backends.cudnn.benchmark}")

            # 非決定的アルゴリズムを許可（より高速だが、実行ごとに結果が微妙に異なる可能性あり）
            torch.backends.cudnn.deterministic = False
            print(f"CuDNN deterministic mode: {torch.backends.cudnn.deterministic}")

            # 自動チューニングを有効化（利用可能な最も効率的なアルゴリズムを選択）
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                print(f"CuDNN TF32 allowed: {torch.backends.cudnn.allow_tf32}")

            if hasattr(torch, 'set_float32_matmul_precision'):
                # Ampere以降のGPUで高精度行列乗算を有効化
                torch.set_float32_matmul_precision('high')
                print("Float32 matmul precision: high")

    return cuda_available


def prepare_dataset(tokenizer, dataset_path=None, dataset_name=None):
    """
    データセットを準備し、トークン化します

    Args:
        tokenizer: 使用するトークナイザー
        dataset_path: カスタムデータセットのディレクトリパス（優先される）
        dataset_name: Hugging Faceの組み込みデータセット名

    Returns:
        tokenized_datasets: トークン化されたデータセット
    """
    if dataset_path:
        print(f"カスタムデータセットディレクトリ {dataset_path} を探索しています...")
        # ディレクトリ内のすべてのJSONLファイルを検索
        import glob
        jsonl_files = glob.glob(os.path.join(dataset_path, "*.jsonl"))

        if not jsonl_files:
            raise ValueError(f"{dataset_path} にJSONLファイルが見つかりませんでした")

        print(f"見つかったJSONLファイル: {len(jsonl_files)}個")
        for file in jsonl_files:
            print(f" - {os.path.basename(file)}")

        # 複数のJSONLファイルをデータセットとして読み込む
        dataset = load_dataset('json', data_files=jsonl_files)

        # データセットの分割（トレーニングセットのみの場合）
        if 'train' not in dataset:
            dataset = dataset['train'].train_test_split(test_size=0.1)
    elif dataset_name:
        print(f"Hugging Faceデータセット {dataset_name} を準備しています...")
        dataset = load_dataset(dataset_name)
    else:
        raise ValueError("dataset_pathまたはdataset_nameのいずれかを指定してください")

    def tokenize_function(examples):
        # 入力をトークン化
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        # labelsとして同じ入力を設定（言語モデリングのため）
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    print("データセットをトークン化しています...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=3
    )
    return tokenized_datasets


def train_model(model, tokenizer, train_dataset, results_dir, logs_dir, gpu_available=False):
    """
    モデルの学習を実行します

    Args:
        model: 学習するモデル
        tokenizer: 使用するトークナイザー
        train_dataset: 学習用データセット
        results_dir: 結果保存先ディレクトリ
        logs_dir: ログ保存先ディレクトリ
        gpu_available: GPUが利用可能かどうか

    Returns:
        model: 学習済みのモデル
    """
    print("モデルの学習を開始します...")

    # ディレクトリがなければ作成
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # GPUの数を取得
    n_gpus = torch.cuda.device_count() if gpu_available else 0

    # GPUに合わせてバッチサイズを調整
    batch_size = 4 if gpu_available else 2

    # トレーニングの設定
    training_args = TrainingArguments(
        output_dir=results_dir,          # チェックポイント保存先
        num_train_epochs=3,              # エポック数
        per_device_train_batch_size=batch_size,   # バッチサイズ
        gradient_accumulation_steps=4,   # 勾配蓄積ステップを増加
        save_steps=1000,                 # チェックポイント保存頻度
        save_total_limit=2,              # 保存するチェックポイントの数
        logging_dir=logs_dir,            # ログディレクトリ
        fp16=True if gpu_available else False,    # GPU使用時のみ16ビット精度を使用
        dataloader_num_workers=2,        # データローダーの並列処理
        gradient_checkpointing=True,     # メモリ効率化のためのチェックポイント
        report_to="tensorboard",         # テンソルボードでの可視化
        logging_steps=50,                # ログ出力の頻度
        warmup_steps=100,                # ウォームアップステップ数
        optim="adamw_torch",             # オプティマイザーの選択
        learning_rate=5e-5,              # 学習率
        ddp_find_unused_parameters=False if n_gpus > 1 else None,  # 分散学習の設定
        no_cuda=not gpu_available,       # GPUが利用可能でない場合はCUDAを無効化
    )

    # モデルがすでにdevice_mapを持っている場合の対応
    use_cuda = gpu_available and torch.cuda.is_available()
    print(f"CUDA使用状態: {'有効' if use_cuda else '無効'}")

    # トレーナーの初期化と学習実行
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    return model


def prepare_tokenizer_and_model(model_path):
    # メモリ効率を改善
    print(f"モデルを {model_path} から読み込んでいます...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # GPUの利用可能性を確認
    use_gpu = torch.cuda.is_available()
    print(f"モデルロード時のGPU使用: {'可能' if use_gpu else '不可'}")

    # メモリ効率の良い設定でモデルを読み込む
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if use_gpu else None,  # 利用可能なデバイスに自動的に分散
        # torch_dtype=torch.float16 if use_gpu else torch.float32,  # GPUではfloat16を使用
        low_cpu_mem_usage=True,   # CPU メモリ使用量を抑える
        use_cache=False,  # 勾配チェックポイントと互換性を持たせるために無効化
    )

    # トークナイザーの設定（必要に応じて）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def save_model(model, tokenizer, save_path):
    """
    モデルとトークナイザーを保存します

    Args:
        model: 保存するモデル
        tokenizer: 保存するトークナイザー
        save_path: 保存先パス
    """
    # ディレクトリがなければ作成
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"モデルを {save_path} に保存しました")

    # 保存されたファイルを表示
    print("\n保存されたファイル:")
    for file in os.listdir(save_path):
        file_size = os.path.getsize(os.path.join(save_path, file)) / (1024 * 1024)  # サイズをMB単位で
        print(f" - {file} ({file_size:.2f} MB)")


if __name__ == "__main__":
    # GPUの確認
    gpu_available = check_gpu()
    if gpu_available:
        print("GPUを使用して学習を行います")
    else:
        print("GPUが利用できません。CPUで学習を実行します")

    # スクリプトの場所を基準に相対パスを計算する
    SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
    APP_DIR = SCRIPT_DIR.parent  # app ディレクトリまで遡る

    # パスの設定
    MODEL_PATH = str(APP_DIR / "model" / "01_download")
    RESULTS_DIR = str(SCRIPT_DIR / "temp" / "results")
    LOGS_DIR = str(SCRIPT_DIR / "temp" / "logs")
    MODEL_SAVE_PATH = str(APP_DIR / "model" / "02_trained")

    # カスタムデータセットのディレクトリパスを設定（ファイルではなくディレクトリを指定）
    DATASET_PATH = str(APP_DIR / "dataset")

    # 1. モデルとトークナイザーの準備
    tokenizer, model = prepare_tokenizer_and_model(MODEL_PATH)

    # 2. カスタムデータセットの準備
    tokenized_datasets = prepare_dataset(tokenizer, dataset_path=DATASET_PATH)

    # 3. モデルの学習（GPUの利用可能性を渡す）
    trained_model = train_model(model, tokenizer, tokenized_datasets["train"], RESULTS_DIR, LOGS_DIR, gpu_available)

    # 4. モデルの保存
    save_model(trained_model, tokenizer, MODEL_SAVE_PATH)
