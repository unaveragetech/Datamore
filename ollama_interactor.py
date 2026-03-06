import os
import json
import zipfile
import requests
import sys
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_API = "http://localhost:11434/api"

# Normalise common model name variants to Ollama model tags
MODEL_ALIASES = {
    "llama 3": "llama3",
    "llama3": "llama3",
    "llama 3 (70b)": "llama3:70b",
    "llama 3 (8b)": "llama3:8b",
    "llama 2": "llama2",
    "llama2": "llama2",
    "llama 2 (7b)": "llama2:7b",
    "llama 2 uncensored": "llama2-uncensored",
    "phi 3 mini": "phi3",
    "phi3": "phi3",
    "phi 3 medium": "phi3:medium",
    "phi 3 (7b)": "phi3:7b",
    "gemma (2b)": "gemma:2b",
    "gemma (7b)": "gemma:7b",
    "gemma": "gemma",
    "mistral": "mistral",
    "moondream 2": "moondream",
    "neural chat": "neural-chat",
    "starling": "starling-lm",
    "code llama": "codellama",
    "codellama": "codellama",
    "llava": "llava",
    "solar": "solar",
    "marco-o1": "marco-o1:7b-fp16",
    "macro-1o": "marco-o1:7b-fp16",
    "ibm granite (2b)": "granite:2b",
    "deepseek r1 (1.5b)": "deepseek-r1:1.5b",
    "deepseek-r1": "deepseek-r1",
}

DATASETS_DIR = Path("datasets")
INDEX_FILE = DATASETS_DIR / "index.json"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    """Return repo_owner / repo_name from config.json or env."""
    if os.path.exists("config.json"):
        cfg = json.loads(Path("config.json").read_text())
        if "repo_owner" in cfg and "repo_name" in cfg:
            return cfg
    repo = os.getenv("GITHUB_REPOSITORY", "/")
    parts = repo.split("/", 1)
    return {"repo_owner": parts[0], "repo_name": parts[1] if len(parts) > 1 else ""}


def _gh_headers():
    return {
        "Authorization": f"token {os.getenv('GH_PAT', '')}",
        "Accept": "application/vnd.github.v3+json",
    }


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def get_issue(issue_number):
    cfg = load_config()
    url = (
        f"https://api.github.com/repos/"
        f"{cfg['repo_owner']}/{cfg['repo_name']}/issues/{issue_number}"
    )
    resp = requests.get(url, headers=_gh_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def comment_issue(issue_number, text):
    cfg = load_config()
    url = (
        f"https://api.github.com/repos/"
        f"{cfg['repo_owner']}/{cfg['repo_name']}/issues/{issue_number}/comments"
    )
    resp = requests.post(url, headers=_gh_headers(), json={"body": text}, timeout=30)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Issue body parsing
# ---------------------------------------------------------------------------

def parse_issue_body(body):
    """
    Parse a structured or plain-text issue body.

    Structured format (key: value lines):
        Topic: World History
        Entries: 100
        Format: qa
        Description: Major events since 1900
        Model: mistral          ← optional model override

    Plain format (body is just the topic string):
        History and dates
    """
    body = (body or "").strip()
    params = {
        "topic": None,
        "entries": 50,
        "format": "qa",
        "description": "",
        "model_override": None,
    }

    lines = [l for l in body.split("\n") if l.strip()]
    structured = any(":" in l for l in lines)

    if structured:
        for line in lines:
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip()
                if key == "topic":
                    params["topic"] = val
                elif key in ("entries", "count", "size"):
                    try:
                        # Cap at 500 to stay within Ollama context-window limits
                        params["entries"] = max(1, min(500, int(val)))
                    except ValueError:
                        pass
                elif key == "format":
                    params["format"] = val.lower()
                elif key == "description":
                    params["description"] = val
                elif key in ("model", "model_override"):
                    params["model_override"] = val
        if not params["topic"]:
            params["topic"] = lines[0] if lines else "general"
    else:
        params["topic"] = body or "general"

    return params


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def resolve_model(raw_name):
    return MODEL_ALIASES.get(raw_name.strip().lower(), raw_name.strip().lower())


def pull_model(model):
    """Ask Ollama to pull the model (no-op if already present)."""
    print(f"📥 Pulling model {model} …")
    try:
        resp = requests.post(
            f"{OLLAMA_API}/pull",
            json={"name": model, "stream": False},
            timeout=600,
        )
        resp.raise_for_status()
        print(f"✅ Model {model} ready")
    except Exception as exc:
        print(f"⚠️  Could not pull {model}: {exc}")


def build_prompt(topic, entries, fmt):
    if fmt == "facts":
        return (
            f"Generate {entries} interesting facts about \"{topic}\". "
            "Return ONLY a JSON array of objects with \"fact\" and \"explanation\" fields. "
            "No extra text outside the JSON array."
        )
    if fmt == "story":
        return (
            f"Generate {entries} short story prompts and completions about \"{topic}\". "
            "Return ONLY a JSON array of objects with \"prompt\" and \"story\" fields. "
            "No extra text outside the JSON array."
        )
    # default: qa
    return (
        f"Generate a dataset of {entries} question-answer pairs about \"{topic}\". "
        "Return ONLY a JSON array where each object has an \"instruction\" key (the question) "
        "and an \"output\" key (a detailed answer). "
        "No extra text outside the JSON array."
    )


def call_ollama(model, prompt):
    try:
        resp = requests.post(
            f"{OLLAMA_API}/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        print(f"❌ Ollama API error: {exc}")
        return None


def extract_json(text):
    """Extract the first top-level JSON array from a model response.

    Scans for matching bracket pairs to avoid mis-parsing responses that
    contain explanatory text before or after the array.
    """
    # Walk character-by-character to find a balanced top-level array
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try the next array if this one fails
                    start = None
    # Last resort: try parsing the entire response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

MAX_QUALITY_SAMPLES = 5  # entries shown to the model for scoring


def build_quality_prompt(fmt, samples):
    """Build a prompt asking the model to score a list of dataset entries."""
    samples_json = json.dumps(samples, indent=2)
    field_hint = {
        "qa": '"instruction" (question) and "output" (answer)',
        "facts": '"fact" and "explanation"',
        "story": '"prompt" and "story"',
    }.get(fmt, "the fields present")
    return (
        f"You are a dataset quality evaluator. Below are {len(samples)} sample entries "
        f"from a '{fmt}' dataset. Each entry has {field_hint}.\n\n"
        f"Samples:\n{samples_json}\n\n"
        "Rate each entry on the following criteria (score 1-10):\n"
        "  - Accuracy: Is the content factually correct?\n"
        "  - Clarity: Is the writing clear and well-structured?\n"
        "  - Completeness: Does the entry fully address the topic?\n\n"
        "Return ONLY a JSON array with one object per entry containing:\n"
        '  {"index": <0-based index>, "accuracy": <1-10>, "clarity": <1-10>, '
        '"completeness": <1-10>, "overall": <1-10>, "note": "<one sentence>"}\n'
        "No extra text outside the JSON array."
    )


def assess_dataset_quality(model, fmt, data):
    """
    Ask the model to score up to MAX_QUALITY_SAMPLES entries.

    Returns a dict with keys:
      scores   – list of per-entry score dicts
      average  – dict of average scores per criterion
    """
    if not data:
        return {"scores": [], "average": {}}

    samples = data[:MAX_QUALITY_SAMPLES]
    prompt = build_quality_prompt(fmt, samples)
    print(f"📊 Assessing dataset quality ({len(samples)} samples) …")
    raw = call_ollama(model, prompt)
    if not raw:
        print("⚠️  Quality assessment skipped – no response from model")
        return {"scores": [], "average": {}}

    scores = extract_json(raw)
    if not scores or not isinstance(scores, list):
        print("⚠️  Quality assessment skipped – could not parse scores")
        return {"scores": [], "average": {}}

    # Compute averages across all returned scores
    criteria = ["accuracy", "clarity", "completeness", "overall"]
    totals = {c: 0 for c in criteria}
    count = 0
    for s in scores:
        if isinstance(s, dict):
            for c in criteria:
                totals[c] += s.get(c, 0)
            count += 1
    average = {c: round(totals[c] / count, 1) for c in criteria} if count else {}
    print(f"✅ Quality averages: {average}")
    return {"scores": scores, "average": average}


def _quality_chart_html(quality):
    """Render a compact CSS bar chart for the quality averages."""
    avg = quality.get("average", {})
    if not avg:
        return ""
    criteria_labels = {
        "accuracy": "Accuracy",
        "clarity": "Clarity",
        "completeness": "Completeness",
        "overall": "Overall",
    }
    bars = ""
    for key, label in criteria_labels.items():
        score = avg.get(key, 0)
        pct = score * 10  # score is 1-10, map to 0-100%
        color = "#3fb950" if score >= 7 else "#d29922" if score >= 4 else "#f85149"
        bars += (
            f'<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem">'
            f'<span style="width:100px;color:#8b949e;font-size:.8rem">{label}</span>'
            f'<div style="flex:1;background:#21262d;border-radius:4px;height:14px">'
            f'<div style="width:{pct}%;background:{color};height:14px;border-radius:4px"></div>'
            f'</div>'
            f'<span style="width:24px;text-align:right;font-size:.8rem;color:#e6edf3">{score}</span>'
            f'</div>'
        )
    scores_list = quality.get("scores", [])
    notes_html = ""
    if scores_list:
        notes = [
            f'<li style="font-size:.8rem;color:#8b949e">Entry {s.get("index",i)}: {s.get("note","")}</li>'
            for i, s in enumerate(scores_list) if isinstance(s, dict) and s.get("note")
        ]
        if notes:
            notes_html = '<ul style="padding-left:1rem;margin-top:.5rem">' + "".join(notes) + "</ul>"
    return (
        f'<div style="margin-top:.75rem">'
        f'<p style="color:#8b949e;font-size:.8rem;margin-bottom:.5rem">Quality Assessment '
        f'(sampled {len(scores_list)} entr{"y" if len(scores_list)==1 else "ies"})</p>'
        f'{bars}{notes_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Dataset persistence
# ---------------------------------------------------------------------------

def save_dataset_zip(issue_number, model, topic, entries_requested, fmt, data, raw_response,
                     quality=None):
    """Write a zip containing dataset.json, dataset.jsonl, quality.json and README.md."""
    DATASETS_DIR.mkdir(exist_ok=True)
    safe_topic = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in topic.lower()
    )[:50]
    safe_model = model.replace(":", "-").replace("/", "-")
    zip_name = f"issue_{issue_number}_{safe_model}_{safe_topic}.zip"
    zip_path = DATASETS_DIR / zip_name

    generated_at = datetime.now(timezone.utc).isoformat()
    dataset_obj = {
        "metadata": {
            "issue_number": issue_number,
            "model": model,
            "topic": topic,
            "format": fmt,
            "entries_requested": entries_requested,
            "entries_generated": len(data) if data else 0,
            "generated_at": generated_at,
            "quality": quality or {},
        },
        "data": data or [],
        "raw_response": raw_response or "",
    }

    avg = (quality or {}).get("average", {})
    quality_section = ""
    if avg:
        quality_section = (
            "\n## Quality Assessment\n\n"
            "| Criterion | Score (1–10) |\n|---|---|\n"
            + "".join(f"| {k.capitalize()} | {v} |\n" for k, v in avg.items())
        )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset.json", json.dumps(dataset_obj, indent=2))
        if data:
            zf.writestr(
                "dataset.jsonl",
                "\n".join(json.dumps(row) for row in data),
            )
        if quality:
            zf.writestr("quality.json", json.dumps(quality, indent=2))
        readme = (
            f"# Dataset: {topic}\n\n"
            f"Generated by [Datamore](https://github.com/unaveragetech/Datamore) "
            f"from GitHub Issue #{issue_number}\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Model | `{model}` |\n"
            f"| Topic | {topic} |\n"
            f"| Format | {fmt} |\n"
            f"| Entries | {len(data) if data else 0} |\n"
            f"| Generated at | {generated_at} |\n"
            f"{quality_section}"
        )
        zf.writestr("README.md", readme)

    print(f"✅ Saved: {zip_path}")
    return zip_name, zip_path


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def load_index():
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text())
        except Exception:
            pass
    return {"datasets": []}


def update_index(issue_number, model, topic, fmt, entries, zip_name, zip_path, issue_url,
                 issue_title, quality=None):
    idx = load_index()
    # Remove any previous entry for this issue
    idx["datasets"] = [d for d in idx["datasets"] if d.get("issue_number") != issue_number]
    idx["datasets"].insert(0, {
        "issue_number": issue_number,
        "issue_title": issue_title,
        "issue_url": issue_url,
        "model": model,
        "topic": topic,
        "format": fmt,
        "entries": entries,
        "zip_file": f"datasets/{zip_name}",
        "file_size": os.path.getsize(zip_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality": quality or {},
    })
    DATASETS_DIR.mkdir(exist_ok=True)
    INDEX_FILE.write_text(json.dumps(idx, indent=2))
    print("✅ Updated datasets/index.json")
    return idx


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _card(d):
    quality_html = _quality_chart_html(d.get("quality", {}))
    return f"""
        <div class="card">
          <h3>{d.get('issue_title', d.get('topic', 'Dataset'))}</h3>
          <p><strong>Topic:</strong> {d.get('topic', '')}</p>
          <p><strong>Model:</strong> <code>{d.get('model', '')}</code></p>
          <p><strong>Format:</strong> {d.get('format', 'qa').upper()}</p>
          <p><strong>Entries:</strong> {d.get('entries', '?')}</p>
          <p><strong>Created:</strong> {str(d.get('created_at', ''))[:10]}</p>
          {quality_html}
          <a href="{d.get('zip_file', '#')}" class="download-btn">&#x2B07; Download</a>
          <a href="{d.get('issue_url', '#')}" target="_blank" class="issue-link">View Issue</a>
        </div>"""


def _quality_badge(quality):
    avg = (quality or {}).get("average", {})
    overall = avg.get("overall")
    if overall is None:
        return ""
    color = "#3fb950" if overall >= 7 else "#d29922" if overall >= 4 else "#f85149"
    return (
        f'<span style="background:{color}20;color:{color};border:1px solid {color}50;'
        f'padding:1px 6px;border-radius:10px;font-size:.75rem;margin-left:.25rem">'
        f'★ {overall}</span>'
    )


def _row(d):
    size_kb = d.get("file_size", 0) // 1024
    quality_badge = _quality_badge(d.get("quality"))
    return (
        f"<tr>"
        f"<td><a href=\"{d.get('issue_url','#')}\" target=\"_blank\">#{d.get('issue_number')}</a></td>"
        f"<td>{d.get('issue_title','')}</td>"
        f"<td>{d.get('topic','')}</td>"
        f"<td><code>{d.get('model','')}</code></td>"
        f"<td>{d.get('format','qa').upper()}</td>"
        f"<td>{d.get('entries','?')}</td>"
        f"<td>{size_kb} KB</td>"
        f"<td>{str(d.get('created_at',''))[:10]}{quality_badge}</td>"
        f"<td><a href=\"{d.get('zip_file','#')}\" class=\"download-btn\">&#x2B07; Download</a></td>"
        f"</tr>"
    )


def generate_index_html(idx):
    datasets = idx.get("datasets", [])
    total = len(datasets)
    total_size_kb = sum(d.get("file_size", 0) for d in datasets) // 1024
    models_used = len({d.get("model", "") for d in datasets})

    recent_cards = "".join(_card(d) for d in datasets[:3]) or (
        '<p class="empty">No datasets yet. '
        '<a href="../../issues/new">Create an issue</a> to generate one!</p>'
    )
    rows = "".join(_row(d) for d in datasets) or (
        '<tr><td colspan="9" style="text-align:center;color:#8b949e">No datasets yet.</td></tr>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Datamore &#8211; AI Dataset Generator</title>
  <style>
    :root {{
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
      --green: #3fb950; --purple: #bc8cff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ background: var(--surface); border-bottom: 1px solid var(--border);
              padding: 1.5rem 2rem; }}
    header h1 {{ font-size: 1.8rem; color: var(--accent); }}
    header p {{ color: var(--muted); margin-top: .25rem; }}
    .badge {{ background: rgba(188,140,255,.15); color: var(--purple);
              border: 1px solid rgba(188,140,255,.3); padding: 2px 8px;
              border-radius: 12px; font-size: .75rem; margin-left: .5rem; }}
    .hero-stats {{ background: var(--surface); border-bottom: 1px solid var(--border);
                   padding: 1rem 2rem; display: flex; gap: 2rem; flex-wrap: wrap; }}
    .stat .value {{ font-size: 2rem; font-weight: 700; color: var(--green); }}
    .stat .label {{ color: var(--muted); font-size: .85rem; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
    h2 {{ font-size: 1.3rem; margin-bottom: 1rem; border-bottom: 1px solid var(--border);
          padding-bottom: .5rem; }}
    .how-to {{ background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; padding: 1.5rem; margin-bottom: 2.5rem; }}
    .how-to ol {{ padding-left: 1.5rem; color: var(--muted); line-height: 1.9; }}
    .how-to ol strong {{ color: var(--text); }}
    pre {{ background: #0d1117; border: 1px solid var(--border); border-radius: 6px;
            padding: 1rem; margin-top: .75rem; color: var(--muted);
            font-size: .85rem; overflow-x: auto; }}
    .recent-grid {{ display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 1rem; margin-bottom: 2.5rem; }}
    .card {{ background: var(--surface); border: 1px solid var(--border);
              border-radius: 8px; padding: 1.25rem;
              transition: border-color .2s; }}
    .card:hover {{ border-color: var(--accent); }}
    .card h3 {{ font-size: 1rem; color: var(--accent); margin-bottom: .75rem; }}
    .card p {{ font-size: .875rem; color: var(--muted); margin-bottom: .4rem; }}
    .card p strong {{ color: var(--text); }}
    code {{ background: rgba(88,166,255,.1); color: var(--accent);
             padding: 2px 6px; border-radius: 4px; font-size: .8rem; }}
    .download-btn {{ display: inline-block; margin-top: .75rem; margin-right: .5rem;
                      background: var(--green); color: #000; padding: .4rem .9rem;
                      border-radius: 6px; text-decoration: none; font-size: .85rem;
                      font-weight: 600; transition: opacity .2s; }}
    .download-btn:hover {{ opacity: .8; }}
    .issue-link {{ display: inline-block; margin-top: .75rem; color: var(--accent);
                   text-decoration: none; font-size: .85rem; padding: .4rem .9rem;
                   border: 1px solid var(--accent); border-radius: 6px;
                   transition: background .2s; }}
    .issue-link:hover {{ background: rgba(88,166,255,.1); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .875rem; }}
    th {{ background: var(--surface); color: var(--muted); font-weight: 600;
           text-align: left; padding: .65rem .75rem;
           border-bottom: 2px solid var(--border); white-space: nowrap; }}
    td {{ padding: .65rem .75rem; border-bottom: 1px solid var(--border); }}
    tr:hover td {{ background: rgba(255,255,255,.03); }}
    td a {{ color: var(--accent); text-decoration: none; }}
    td a:hover {{ text-decoration: underline; }}
    td .download-btn {{ margin-top: 0; padding: .3rem .7rem; font-size: .8rem; }}
    .empty {{ color: var(--muted); padding: 1rem 0; }}
    footer {{ border-top: 1px solid var(--border); padding: 1.5rem 2rem;
               text-align: center; color: var(--muted); font-size: .85rem; margin-top: 3rem; }}
    footer a {{ color: var(--accent); text-decoration: none; }}
  </style>
</head>
<body>
  <header>
    <h1>&#x1F916; Datamore <span class="badge">AI Datasets</span></h1>
    <p>Generate AI training datasets from GitHub Issues using Ollama</p>
  </header>

  <div class="hero-stats">
    <div class="stat">
      <div class="value">{total}</div>
      <div class="label">Datasets Generated</div>
    </div>
    <div class="stat">
      <div class="value">{total_size_kb} KB</div>
      <div class="label">Total Size</div>
    </div>
    <div class="stat">
      <div class="value">{models_used}</div>
      <div class="label">Models Used</div>
    </div>
  </div>

  <main>
    <div class="how-to">
      <h2>&#x1F4CB; How to Generate a Dataset</h2>
      <ol>
        <li><strong>Create a new issue</strong> in this repository.</li>
        <li><strong>Set the title</strong> to the Ollama model name
            (e.g. <code>Llama 3</code>, <code>Mistral</code>, <code>deepseek-r1</code>).</li>
        <li><strong>Set the body</strong> to your topic
            (e.g. <code>History and dates</code>) or use the structured format below.</li>
      </ol>
      <pre>Topic: World History
Entries: 100
Format: qa
Description: Questions about major world history events</pre>
      <h3 style="margin-top:1.25rem;margin-bottom:.5rem;font-size:1rem">&#x1F4C4; Dataset Formats</h3>
      <table style="margin-bottom:.75rem">
        <thead><tr><th>Format</th><th>Fields</th><th>Best for</th></tr></thead>
        <tbody>
          <tr><td><code>qa</code></td><td><code>instruction</code>, <code>output</code></td><td>Q&amp;A pairs, instruction-tuning, chatbots</td></tr>
          <tr><td><code>facts</code></td><td><code>fact</code>, <code>explanation</code></td><td>Knowledge bases, trivia, encyclopedic data</td></tr>
          <tr><td><code>story</code></td><td><code>prompt</code>, <code>story</code></td><td>Creative writing, narrative generation</td></tr>
        </tbody>
      </table>
      <h3 style="margin-top:1rem;margin-bottom:.5rem;font-size:1rem">&#x1F522; How Many Entries?</h3>
      <p style="color:var(--muted);font-size:.875rem;line-height:1.7">
        The <code>Entries</code> field controls how many dataset rows are generated (1&ndash;500).
        Recommended starting points:
      </p>
      <ul style="padding-left:1.5rem;color:var(--muted);font-size:.875rem;line-height:1.9;margin-top:.4rem">
        <li><strong style="color:var(--text)">Quick test</strong> &mdash; <code>Entries: 10</code> (fast, good for validating a topic)</li>
        <li><strong style="color:var(--text)">Small dataset</strong> &mdash; <code>Entries: 50</code> (default, suitable for fine-tuning experiments)</li>
        <li><strong style="color:var(--text)">Medium dataset</strong> &mdash; <code>Entries: 200</code> (good coverage of a topic)</li>
        <li><strong style="color:var(--text)">Large dataset</strong> &mdash; <code>Entries: 500</code> (maximum; may take several minutes)</li>
      </ul>
      <p style="color:var(--muted);margin-top:.75rem;font-size:.875rem">
        Each generated dataset is automatically <strong style="color:var(--text)">zipped</strong>
        (containing <code>dataset.json</code>, <code>dataset.jsonl</code>,
        <code>quality.json</code>, and <code>README.md</code>) and assessed for quality.
        The &#x2605; score shown on each card is the model&rsquo;s own overall quality rating (1&ndash;10).
      </p>
    </div>

    <h2>&#x2728; Recent Datasets</h2>
    <div class="recent-grid">
      {recent_cards}
    </div>

    <h2>&#x1F4E6; All Datasets</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Issue</th><th>Title</th><th>Topic</th><th>Model</th>
            <th>Format</th><th>Entries</th><th>Size</th><th>Created</th><th>Download</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </main>

  <footer>
    <p>
      Powered by <a href="https://ollama.com" target="_blank">Ollama</a> &amp;
      <a href="https://github.com/unaveragetech/Datamore" target="_blank">Datamore</a>
      &middot; Datasets are generated automatically from GitHub Issues
    </p>
  </footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python ollama_interactor.py <issue_number>")
        sys.exit(1)

    issue_number = int(sys.argv[1])
    print(f"🔍 Processing issue #{issue_number} …")

    issue = get_issue(issue_number)
    issue_title = issue["title"].strip()
    issue_body = (issue.get("body") or "").strip()
    issue_url = issue["html_url"]

    params = parse_issue_body(issue_body)
    raw_model = params.get("model_override") or issue_title
    model = resolve_model(raw_model)
    topic = params["topic"]
    entries = params["entries"]
    fmt = params["format"]

    print(f"  Model  : {model}")
    print(f"  Topic  : {topic}")
    print(f"  Entries: {entries}")
    print(f"  Format : {fmt}")

    pull_model(model)

    prompt = build_prompt(topic, entries, fmt)
    print(f"📤 Sending prompt to {model} …")
    raw_response = call_ollama(model, prompt)
    if not raw_response:
        comment_issue(
            issue_number,
            f"❌ Failed to get a response from model `{model}`.\n"
            "Check that Ollama is running and the model name is correct.",
        )
        sys.exit(1)

    data = extract_json(raw_response)
    if not data:
        print("⚠️  Could not parse JSON from response – wrapping raw text")
        # Preserve schema keys matching the requested format so downstream
        # consumers always see consistent field names.
        fallback_key = {"qa": "instruction", "facts": "fact", "story": "prompt"}.get(fmt, "raw")
        data = [{fallback_key: raw_response, "_parse_error": True}]
    actual_entries = len(data)
    print(f"✅ Got {actual_entries} entries")

    quality = assess_dataset_quality(model, fmt, data)

    zip_name, zip_path = save_dataset_zip(
        issue_number, model, topic, entries, fmt, data, raw_response, quality
    )

    idx = update_index(
        issue_number, model, topic, fmt, actual_entries,
        zip_name, zip_path, issue_url, issue_title, quality,
    )

    html = generate_index_html(idx)
    Path("index.html").write_text(html, encoding="utf-8")
    print("✅ Updated index.html")

    cfg = load_config()
    pages_url = f"https://{cfg['repo_owner']}.github.io/{cfg['repo_name']}/"

    avg = quality.get("average", {})
    quality_comment = ""
    if avg:
        quality_comment = (
            "\n\n**Quality Assessment** (sampled entries scored by the model):\n\n"
            "| Criterion | Score (1–10) |\n|---|---|\n"
            + "".join(f"| {k.capitalize()} | {v} |\n" for k, v in avg.items())
        )

    comment_issue(
        issue_number,
        f"## ✅ Dataset Generated\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Model | `{model}` |\n"
        f"| Topic | {topic} |\n"
        f"| Format | {fmt.upper()} |\n"
        f"| Entries | {actual_entries} |\n"
        f"| Download | [{zip_name}](../../raw/main/datasets/{zip_name}) |\n"
        f"{quality_comment}\n"
        f"📊 View all datasets: [{pages_url}]({pages_url})",
    )


if __name__ == "__main__":
    main()
