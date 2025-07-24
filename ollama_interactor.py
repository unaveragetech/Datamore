import os
import subprocess
import json
import requests
import sys

# Full model command mapping
model_commands = {
    "llama 3": "ollama run llama3",
    "llama 3 (70b)": "ollama run llama3:70b",
    "llama 3 (8b)": "ollama run llama3:8b",
    "phi 3 mini": "ollama run phi3",
    "phi 3 medium": "ollama run phi3:medium",
    "gemma (2b)": "ollama run gemma:2b",
    "gemma (7b)": "ollama run gemma:7b",
    "mistral": "ollama run mistral",
    "moondream 2": "ollama run moondream",
    "neural chat": "ollama run neural-chat",
    "starling": "ollama run starling-lm",
    "code llama": "ollama run codellama",
    "llama 2 uncensored": "ollama run llama2-uncensored",
    "llava": "ollama run llava",
    "solar": "ollama run solar",
    "macro-1o": "ollama run marco-o1:7b-fp16",
    "minslayer_v2_basic": "ollama run Beelzebub4883/Mineslayer_V2_basic",
    "ibm granite (2b)": "ollama run granite:2b",
    "deepseek r1 (1.5b)": "ollama run deepseek-r1:1.5b",
    "llama 2 (7b)": "ollama run llama2:7b",
    "phi 3 (7b)": "ollama run phi3:7b",
    "chatgpt (3.5b)": "ollama run chatgpt:3.5b",
    "gpt-neo (2.7b)": "ollama run gpt-neo:2.7b",
    "gpt-j (6b)": "ollama run gpt-j:6b",
    "t5 (3b)": "ollama run t5:3b",
    "bert (large)": "ollama run bert:large",
    "xlnet (large)": "ollama run xlnet:large",
    "roberta (large)": "ollama run roberta:large",
    "distilbert (base)": "ollama run distilbert:base",
    "transformer-xl (large)": "ollama run transformer-xl:large",
    "ernie (large)": "ollama run ernie:large",
    "electra (large)": "ollama run electra:large",
    "albert (xlarge)": "ollama run albert:xlarge",
    "reformer (large)": "ollama run reformer:large",
    "funnel-transformer (large)": "ollama run funnel-transformer:large"
}

def load_config():
    """Load config.json or derive repo info from environment."""
    if os.path.exists("config.json"):
        cfg = json.load(open("config.json"))
        if "repo_owner" not in cfg or "repo_name" not in cfg:
            raise Exception("config.json needs repo_owner and repo_name")
        return cfg
    else:
        owner, repo = os.getenv("GITHUB_REPOSITORY", ":").split(":")[0].split("/")
        return {"repo_owner": owner, "repo_name": repo}

def get_issue_data(issue_number):
    cfg = load_config()
    url = f"https://api.github.com/repos/{cfg['repo_owner']}/{cfg['repo_name']}/issues/{issue_number}"
    resp = requests.get(url, headers={
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3+json"
    })
    resp.raise_for_status()
    issue = resp.json()
    return issue["title"].strip(), issue["body"].strip()

def run_model(model_name, prompt):
    key = model_name.strip().lower()
    cmd = model_commands.get(key)
    if not cmd:
        print(f"❌ Unknown model: {model_name}")
        return None
    proc = subprocess.Popen(cmd.split() + [prompt],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if proc.returncode != 0:
        print(f"Ollama error:{err}")
        return None
    return out.strip()

def comment_issue(issue_number, text):
    cfg = load_config()
    url = f"https://api.github.com/repos/{cfg['repo_owner']}/{cfg['repo_name']}/issues/{issue_number}/comments"
    resp = requests.post(url, headers={
        "Authorization": f"token {os.getenv('GH_PAT')}",
        "Accept": "application/vnd.github.v3+json"
    }, json={"body": text})
    resp.raise_for_status()

def save_dataset(issue_number, model_name, prompt, response):
    safe = model_name.replace(" ", "_").lower()
    dir_path = f"datasets/{safe}"
    os.makedirs(dir_path, exist_ok=True)
    path = f"{dir_path}/issue_{issue_number}.json"
    json.dump({"prompt": prompt, "response": response}, open(path, "w"), indent=2)
    print(f"✅ Saved dataset: {path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python ollama_interactor.py <issue_number>")
        sys.exit(1)
    issue = sys.argv[1]
    model, prompt = get_issue_data(issue)
    print(f"Issue #{issue} → model: {model}")
    res = run_model(model, prompt)
    if not res:
        comment_issue(issue, f"❌ Failed to run model `{model}`.")
        return
    save_dataset(issue, model, prompt, res)
    comment_issue(issue, f"✅ Output from `{model}`:\n```\n{res[:1900]}\n```")

if __name__ == "__main__":
    main()
