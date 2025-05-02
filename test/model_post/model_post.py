import json
import requests
import argparse


def generate_text(input_text):
    url = "http://0.0.0.0:8000/generate/"
    data = {"attention_mask": "てすと", "text": input_text}

    response = requests.post(url, json.dumps(data))
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIモデルでテキスト生成します')
    parser.add_argument('text', type=str, help='生成のためのテキスト入力')

    args = parser.parse_args()

    result = generate_text(args.text)
    print(result)
