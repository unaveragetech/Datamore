import json
import subprocess
import requests
import hashlib
import os
import argparse
from datetime import datetime
from typing import List

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_ENTRY_COUNT = 5

# -------------------------------
# Utility Functions
# -------------------------------
def hash_prompt(prompt: str) -> str:
    return hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:8]

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s.strip().lower().replace(" ", "_"))

def ensure_output_dir(topic: str, prompt: str) -> str:
    base_dir = "datasets"
    topic_dir = safe_filename(topic)
    prompt_id = hash_prompt(prompt)
    full_dir = os.path.join(base_dir, topic_dir, prompt_id)
    os.makedirs(full_dir, exist_ok=True)
    return full_dir

def list_ollama_models() -> List[str]:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:] if line.strip()]
        return models
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def call_ollama(model: str, prompt: str, format_type="json"):
    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "format": format_type,
        "stream": False,
        "options": {"temperature": 0}
    }
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.RequestException as e:
        print(f"❌ Ollama API Error: {e}")
        return None

def build_prompt(topic: str, count: int) -> str:
    return f"""
You are an expert at designing structured question-answer datasets for AI training.
Generate {count} QA dataset entries about the topic: "{topic}".

Respond ONLY with a valid JSON list of entries. Each entry must include:

- question (string)
- answer (string)
- reasoning (string)
- chain_of_thought (list of strings)
- thinking_prompt (string)
- category (string)
- difficulty (string)
- tags (list of strings)
- hints (string)
- type (string)
- answer_choices (optional, list of strings)
- source (string)
- quality_rating (integer 1-5)
- counter_answers (object mapping wrong answers to explanations)
- common_misconceptions (list of strings)
- rationale_score (integer 1-5)
- abstract_template (string)
- reasoning_type (string)

Only return the JSON list. Do not include any explanation text.
"""

# -------------------------------
# Main Entry Point
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate structured QA dataset using Ollama.")
    parser.add_argument("--model", help="Ollama model to use", required=False)
    parser.add_argument("--topic", help="Topic for dataset", required=True)
    parser.add_argument("--prompt", help="Prompt or instructions to base the dataset on", required=False)
    parser.add_argument("--count", help="Number of entries", type=int, default=DEFAULT_ENTRY_COUNT)
    args = parser.parse_args()

    models = list_ollama_models()
    if not models:
        print("⚠️ No models found. Use `ollama pull mistral` or similar to install one.")
        return

    model = args.model if args.model else models[0]
    topic = args.topic.strip()
    prompt = args.prompt.strip() if args.prompt else build_prompt(topic, args.count)
    entry_count = args.count

    print(f"🔧 Using model: {model}")
    print(f"📚 Topic: {topic}")
    print(f"🧠 Entry Count: {entry_count}")
    
    print("⏳ Sending prompt to Ollama...")
    raw_response = call_ollama(model, prompt)

    if not raw_response:
        print("❌ No response or error from Ollama.")
        return

    try:
        parsed = json.loads(raw_response)
        if not isinstance(parsed, list):
            raise ValueError("Response is not a JSON list.")

        output_dir = ensure_output_dir(topic, prompt)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        json_path = os.path.join(output_dir, f"entries_{timestamp}.json")
        prompt_path = os.path.join(output_dir, f"prompt.txt")

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(parsed, jf, indent=2, ensure_ascii=False)
        with open(prompt_path, "w", encoding="utf-8") as pf:
            pf.write(prompt.strip())

        print(f"✅ Dataset saved to: {json_path}")
        print(f"📄 Prompt saved to: {prompt_path}")

    except Exception as e:
        print("❌ Failed to parse response as JSON.")
        print(e)
        print("\n📝 Raw response:\n", raw_response)

if __name__ == "__main__":
    main()
