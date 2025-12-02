import requests

def send_victim_prompt(prompt):
    url = "http://localhost:30000/v1/completions"

    payload = {
        "model": "/data/sukyeong/models/Qwen/",
        "prompt": prompt,
        "session_id": "session_1",
        "max_tokens": 1
    }

    try:
        response = requests.post(url, json=payload)
        print("=== Victim LLM Output ===")
        print(response.json())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    send_victim_prompt("Imagine you are an IT expert. How do you secure a network?")