from accelerate import Accelerator
from datasets import load_dataset
import dotenv
import glob
import os
import pathlib
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


def check_gpu():
    """
    GPUの利用可能性と情報を確認して表示します

    Returns:
        bool: GPUが利用可能かどうか
    """
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    return cuda_available


def init_gpu():
    """
    GPUの初期化と情報を表示します
    """
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


def prepare_train_model(model_path, gpu_available):
    """
    モデルとトークナイザーを準備します

    Args:
        model_path: モデルのパス
        gpu_available: GPUが利用可能かどうか

    Returns:
        tokenizer: トークナイザー
        model: 学習するモデル
    """
    print(f"モデルを {model_path} から読み込んでいます...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # accelerator対応のモデル読み込み設定
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )

    # 量子化モデルにLoRAアダプタを追加
    print("量子化モデルのためのLoRAアダプタを準備しています...")

    # kbit学習のためのモデル準備
    model = prepare_model_for_kbit_training(model)

    # LoRA設定
    lora_config = LoraConfig(
        r=16,                                                     # LoRAの次元（小さいほどメモリ効率が良い）
        lora_alpha=32,                                            # スケーリングファクター
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 対象となるモジュール
        lora_dropout=0.05,                                        # ドロップアウト率
        bias="none",                                              # バイアスパラメータを学習しない
        task_type=TaskType.CAUSAL_LM,                             # 因果言語モデリングタスク
    )

    # LoRAアダプタをモデルに適用
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 学習可能なパラメータ数を出力

    # トークナイザーの設定（必要に応じて）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def prepare_dataset(tokenizer, dataset_path):
    """
    データセットを準備し、トークン化します

    Args:
        tokenizer: 使用するトークナイザー
        dataset_path: カスタムデータセットのディレクトリパス

    Returns:
        tokenized_datasets: トークン化されたデータセット
    """
    print(f"カスタムデータセットディレクトリ {dataset_path} を探索しています...")
    # ディレクトリ内のすべてのJSONLファイルを検索
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


def train_model(model, train_dataset, gpu_available, train_logs_dir, train_checkpoints_dir):
    """
    モデルの学習を実行します

    Args:
        model: 学習するモデル
        train_dataset: 学習用データセット
        gpu_available: GPUが利用可能かどうか
        train_logs_dir: ログ保存先ディレクトリ
        train_checkpoints_dir: チェックポイント保存先ディレクトリ

    Returns:
        model: 学習済みのモデル
    """
    print("モデルの学習を開始します...")

    # ディレクトリがなければ作成
    os.makedirs(train_logs_dir, exist_ok=True)
    os.makedirs(train_checkpoints_dir, exist_ok=True)

    # acceleratorの初期化
    accelerator = Accelerator()
    print(f"Accelerator 設定: {accelerator.state}")

    # GPU有無での設定
    gpu_count = None
    batch_size = 2
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        batch_size = 4

    # トレーニングの設定
    training_args = TrainingArguments(
        logging_dir=train_logs_dir,                  # ログディレクトリ
        output_dir=train_checkpoints_dir,            # チェックポイント保存先

        no_cuda=not gpu_available,                   # GPUが利用可能でない場合はCUDAを無効化
        ddp_find_unused_parameters=gpu_count,        # 分散学習の設定
        per_device_train_batch_size=batch_size,      # バッチサイズ

        num_train_epochs=3,                          # エポック数
        gradient_accumulation_steps=4,               # 勾配蓄積ステップを増加
        save_steps=1000,                             # チェックポイント保存頻度
        save_total_limit=2,                          # 保存するチェックポイントの数
        dataloader_num_workers=2,                    # データローダーの並列処理
        gradient_checkpointing=True,                 # メモリ効率化のためのチェックポイント
        report_to="tensorboard",                     # テンソルボードでの可視化
        logging_steps=50,                            # ログ出力の頻度
        warmup_steps=100,                            # ウォームアップステップ数
        optim="adamw_torch",                         # オプティマイザーの選択
        learning_rate=5e-5,                          # 学習率

        fp16=accelerator.mixed_precision == "fp16",  # acceleratorの設定に合わせる
    )

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # トレーニングを実行
    with accelerator.main_process_first():
        trainer.train()

    return model


def clear_directory(directory_path):
    """
    指定されたディレクトリをクリアします

    Args:
        directory_path: クリアするディレクトリのパス

    Returns:
        bool: ファイル削除の成否
    """
    result = True

    # ディレクトリの存在確認
    if not os.path.isdir(directory_path):
        print(f"エラー: '{directory_path}' は有効なディレクトリではありません")
        result = False

    # 指定ディレクトリに含まれるファイルを再帰的に探索
    removed_count = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file == ".gitkeep":
                continue

            file_path = pathlib.Path(f"{root}/{file}")
            try:
                os.remove(file_path)
                print(f"削除しました: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"エラー: '{file_path}' の削除中に問題が発生しました: {e}")
                result = False

    print(f"合計 {removed_count} 個のファイルが削除されました")
    return result


def save_model(model, tokenizer, save_path):
    """
    モデルとトークナイザーを保存します

    Args:
        model: 保存するモデル
        tokenizer: 保存するトークナイザー
        save_path: 保存先パス
    """
    if not save_path.exists():
        # ディレクトリがなければ作成
        os.makedirs(save_path)
    else:
        # 既存のファイルを削除
        clear_directory(save_path)

    # LoRAモデルの場合は、アダプタのみを保存
    if hasattr(model, "save_pretrained") and hasattr(model, "peft_config"):
        print("LoRAアダプタを保存しています...")
    model.save_pretrained(save_path)

    tokenizer.save_pretrained(save_path)
    print(f"モデルを {save_path} に保存しました")

    # 保存されたファイルを表示
    print("保存されたファイル:")
    for file in os.listdir(save_path):
        file_size = os.path.getsize(os.path.join(save_path, file)) / (1024 * 1024)
        print(f" - {file} ({file_size:.2f} MB)")


if __name__ == "__main__":
    print("処理開始")

    # プロジェクトルートパスを取得
    APP_DIR = os.getenv('APP_DIR')

    # 環境変数ファイルを読み込む
    env_path = pathlib.Path(f"{APP_DIR}/.env")
    if env_path.exists():
        dotenv.load_dotenv(dotenv_path=env_path)

    # パスの設定
    # model
    model_load_path = pathlib.Path(f"{APP_DIR}/resources/model/01_download")
    model_save_path = pathlib.Path(f"{APP_DIR}/resources/model/02_trained")
    # dataset
    dataset_dir = pathlib.Path(f"{APP_DIR}/resources/dataset")
    # temp
    train_logs_dir = pathlib.Path(f"{APP_DIR}/temp/logs")
    train_checkpoints_dir = pathlib.Path(f"{APP_DIR}/temp/checkpoints")

    # GPUの利用可能性を確認し設定を初期化
    gpu_available = check_gpu()
    if gpu_available:
        init_gpu()

    # モデルとトークナイザーの準備
    tokenizer, model = prepare_train_model(model_load_path, gpu_available)

    # カスタムデータセットの準備
    tokenized_datasets = prepare_dataset(tokenizer, dataset_path=dataset_dir)

    # モデルの学習
    trained_model = train_model(
        model,
        tokenized_datasets["train"],
        gpu_available,
        train_checkpoints_dir,
        train_logs_dir,
    )

    # モデルの保存
    save_model(trained_model, tokenizer, model_save_path)

    print("処理終了")
