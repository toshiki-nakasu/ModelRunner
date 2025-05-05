from accelerate import Accelerator
import dotenv
import os
import pathlib
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def download_model(model_name, save_path):
    """
    指定されたモデルとそのトークナイザーをダウンロードして保存します

    Args:
        model_name: Hugging Faceのモデル名 (例: "gpt2", "facebook/opt-350m")
        save_path: モデルを保存するディレクトリパス
    """
    print(f"モデル {model_name} をダウンロード中...")

    if not save_path.exists():
        # ディレクトリがなければ作成
        os.makedirs(save_path)
    else:
        # 既存のファイルを削除
        clear_directory(save_path)

    try:
        # トークナイザーだけをダウンロード
        print("トークナイザーをダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print(f"トークナイザーを {save_path} に保存しました")

        # モデルをダウンロード
        print("モデルをダウンロード中...")
        # メモリ効率のための8bit量子化設定
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,  # if gpu_available else torch.float32,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
        model.save_pretrained(
            save_path,
            max_shard_size="500MB",
            safe_serialization=True
        )
        print(f"モデルを {save_path} に保存しました")

        # 後処理: ダウンロードされたファイルを表示
        print("ダウンロードされたファイル:")
        for file in os.listdir(save_path):
            file_size = os.path.getsize(os.path.join(save_path, file)) / (1024 * 1024)
            print(f" - {file} ({file_size:.2f} MB)")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise e


if __name__ == "__main__":
    print("処理開始")

    # プロジェクトルートパスを取得
    APP_DIR = os.getenv('APP_DIR')

    # 環境変数ファイルを読み込む
    env_path = pathlib.Path(f"{APP_DIR}/.env")
    if env_path.exists():
        dotenv.load_dotenv(dotenv_path=env_path)

    # GPUの利用可能性を確認し設定を初期化
    # gpu_available = check_gpu()
    # if gpu_available:
    #     init_gpu()

    try:
        MODEL_NAME = os.getenv('MODEL_NAME', "llm-jp/llm-jp-3-1.8b-instruct")
        save_path = pathlib.Path(f"{APP_DIR}/resources/model/01_download")
        download_model(MODEL_NAME, save_path)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

    print("処理終了")
