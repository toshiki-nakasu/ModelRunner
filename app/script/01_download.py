from dotenv import load_dotenv
import os
import pathlib
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_name, save_path):
    """
    指定されたモデルとそのトークナイザーをダウンロードして保存します

    Args:
        model_name: Hugging Faceのモデル名 (例: "gpt2", "facebook/opt-350m")
        save_path: モデルを保存するディレクトリパス
    """
    print(f"モデル {model_name} をダウンロード中...")

    # ディレクトリがなければ作成
    os.makedirs(save_path, exist_ok=True)

    try:
        # まずトークナイザーだけをダウンロード
        print("トークナイザーをダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print(f"トークナイザーを {save_path} に保存しました")

        # 次にモデルをダウンロード（メモリ効率の良い設定で）
        print("モデルをダウンロード中...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # 利用可能なデバイスに自動的に分散
            torch_dtype=torch.bfloat16,  # メモリ使用量削減のため半精度を使用
            low_cpu_mem_usage=True       # CPU メモリ使用量を抑える
        )

        # ローカルに保存（シャードサイズを小さく設定し、safetensors形式で保存）
        print("モデルを保存中（小さいシャードサイズで分割保存）...")
        model.save_pretrained(
            save_path,
            max_shard_size="500MB",  # シャードサイズを小さく設定
            safe_serialization=True   # safetensors形式で保存
        )
        print(f"モデル {model_name} を {save_path} に保存しました")

        # ダウンロードされたファイルを表示
        print("\nダウンロードされたファイル:")
        for file in os.listdir(save_path):
            file_size = os.path.getsize(os.path.join(save_path, file)) / (1024 * 1024)  # サイズをMB単位で
            print(f" - {file} ({file_size:.2f} MB)")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # スクリプトの場所を基準に相対パスを計算する
    SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
    APP_DIR = SCRIPT_DIR.parent  # app ディレクトリまで遡る

    # 環境変数ファイルを読み込む
    env_path = APP_DIR.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    # 環境変数からモデル名を取得（ない場合はデフォルト値を使用）
    MODEL_NAME = os.getenv("MODEL_NAME")
    SAVE_PATH = str(APP_DIR / "model" / "01_download")

    print(f"使用するモデル: {MODEL_NAME}")
    download_model(MODEL_NAME, SAVE_PATH)
