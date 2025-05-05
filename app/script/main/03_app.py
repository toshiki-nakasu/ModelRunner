import dotenv
from fastapi import FastAPI, HTTPException
import os
import pathlib
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import uvicorn


def create_app():
    """
    FastAPIアプリケーションを作成し、設定します

    Returns:
        app: 設定済みのFastAPIアプリケーション
    """
    app = FastAPI(title="Custom LLM API")

    # プロジェクトルートパスを取得
    APP_DIR = os.getenv('APP_DIR')

    # 環境変数ファイルを読み込む
    env_path = pathlib.Path(f"{APP_DIR}/.env")
    if env_path.exists():
        dotenv.load_dotenv(dotenv_path=env_path)

    # スクリプトの場所を基準に相対パスを計算する
    model_path = pathlib.Path(f"{APP_DIR}/resources/model/02_trained")

    # GPUの利用可能性を確認し設定を初期化
    # gpu_available = check_gpu()
    # if gpu_available:
    #     init_gpu()

    print(f"モデルを {model_path} から読み込んでいます...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # メモリ効率のための8bit量子化設定
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # モデルのロード
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        torch_dtype=torch.float16  # if gpu_available else torch.float32,
    )

    class QueryInput(BaseModel):
        text: str
        max_length: int = 256

    @app.post("/generate/")
    async def generate_text(query: QueryInput):
        try:
            inputs = tokenizer(query.text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=query.max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"generated_text": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


# アプリケーションのインスタンスを作成
app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
