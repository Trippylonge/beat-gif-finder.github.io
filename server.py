"""
Flask API server for Beat GIF Finder.
"""
from __future__ import annotations
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import analyzer
import sources

HERE = Path(__file__).parent
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="")
CORS(app)  # Allow iframe embedding from any origin (Blogger, etc.)

@app.after_request
def allow_iframe(response):
    # Remove X-Frame-Options so Blogger can embed via iframe
    response.headers.pop("X-Frame-Options", None)
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return response


def load_config() -> dict:
    cfg = HERE / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f)
    return {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/analyze")
def api_analyze():
    duration = float(request.form.get("duration", 30))
    audio_file = request.files.get("audio")

    if audio_file and audio_file.filename:
        suffix = Path(audio_file.filename).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        try:
            feats = analyzer.analyze(tmp_path, duration=duration)
        except Exception as e:
            os.unlink(tmp_path)
            return jsonify({"error": str(e)}), 422
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    else:
        feats = analyzer.analyze(None)

    return jsonify(dataclasses.asdict(feats))


@app.post("/api/search")
def api_search():
    body  = request.get_json(force=True) or {}
    queries = body.get("queries", [])
    limit   = int(body.get("limit", 5))

    if not queries:
        return jsonify({}), 400

    config = load_config()
    results = sources.fetch_all(queries, config, limit_per_query=limit)

    return jsonify({
        src: [dataclasses.asdict(r) for r in items]
        for src, items in results.items()
    })


@app.get("/api/config-status")
def api_config_status():
    """All 3 sources are always active."""
    return jsonify({
        "reddit": True,
        "giphy":  True,
        "tenor":  True,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Beat GIF Finder running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
