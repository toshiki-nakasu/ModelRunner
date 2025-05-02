from fastapi import FastAPI, HTTPException
import pathlib
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_app():
    """
    FastAPIアプリケーションを作成し、設定します

    Returns:
        app: 設定済みのFastAPIアプリケーション
    """
    app = FastAPI(title="Custom LLM API")

    # スクリプトの場所を基準に相対パスを計算する
    SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
    APP_DIR = SCRIPT_DIR.parent  # app ディレクトリまで遡る
    MODEL_PATH = str(APP_DIR / "model" / "02_trained")

    print(f"モデルを {MODEL_PATH} から読み込んでいます...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    class QueryInput(BaseModel):
        text: str
        max_length: int = 100

    @app.post("/generate/")
    async def generate_text(query: QueryInput):
        try:
            inputs = tokenizer(query.text, return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
