"""
Beat GIF Finder
===============
Analyzes a music track's BPM, instruments, and mood, then searches
GIPHY, Tenor, YouTube, and Twitter/X for matching GIF/video suggestions.

Usage:
  python main.py                          # demo mode (no audio)
  python main.py track.mp3               # analyze track
  python main.py track.mp3 --limit 8     # 8 results per query per source
  python main.py track.mp3 --json        # output raw JSON
  python main.py track.mp3 --save out/   # save result HTML report

Setup:
  1. Copy config.example.json to config.json
  2. Add your API keys (at least one of GIPHY or Tenor for GIFs)
  3. Run!

API Keys (all free unless noted):
  GIPHY   https://developers.giphy.com
  Tenor   https://developers.google.com/tenor/guides/quickstart
  YouTube https://console.cloud.google.com  (YouTube Data API v3)
  Twitter Requires Basic plan ($100/mo) — optional
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
import analyzer
import sources

console = Console()
HERE = Path(__file__).parent


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = HERE / "config.json"
    example  = HERE / "config.example.json"

    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)

    # Auto-create from example if missing
    if example.exists():
        import shutil
        shutil.copy(example, cfg_path)
        console.print(
            "[yellow]config.json created from example. "
            "Add your API keys to enable search.[/yellow]\n"
        )
        with open(cfg_path) as f:
            return json.load(f)

    return {}


# ── Display ────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"

def print_track_features(feats: analyzer.TrackFeatures):
    console.rule("[bold cyan]Track Analysis[/bold cyan]")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()

    grid.add_row("BPM",          f"[bold green]{feats.bpm:.1f}[/bold green]  ({feats.bpm_label})")
    grid.add_row("Key",          f"{feats.key}  (confidence: {feats.key_confidence:.0%})")
    grid.add_row("Energy",       f"{_bar(feats.energy)} {feats.energy:.0%}")
    grid.add_row("Valence",      f"{_bar(feats.valence)} {feats.valence:.0%}  "
                                 f"({'bright' if feats.valence > 0.5 else 'dark'})")
    grid.add_row("Danceability", f"{_bar(feats.danceability)} {feats.danceability:.0%}")
    grid.add_row("Bass weight",  f"{_bar(feats.bass_weight)} {feats.bass_weight:.0%}")
    grid.add_row("Harmonic",     f"{_bar(feats.harmonic_weight)} {feats.harmonic_weight:.0%}")
    grid.add_row("Percussive",   f"{_bar(feats.percussive_weight)} {feats.percussive_weight:.0%}")
    grid.add_row("Instruments",  ", ".join(feats.instruments_detected) or "unknown")

    console.print(grid)
    console.print()

    console.print("[bold]Mood tags:[/bold]   " + "  ".join(
        f"[cyan]{t}[/cyan]" for t in feats.mood_tags[:8]
    ))
    console.print("[bold]Scene tags:[/bold]  " + "  ".join(
        f"[magenta]{t}[/magenta]" for t in feats.scene_tags
    ))
    console.print()

    console.print("[bold]Search queries generated:[/bold]")
    for i, q in enumerate(feats.search_queries[:10], 1):
        console.print(f"  [dim]{i:2}.[/dim] {q}")
    console.print()


def print_results(all_results: dict[str, list[sources.GifResult]], top_n: int = 5):
    total = sum(len(v) for v in all_results.values())
    if total == 0:
        console.print(
            Panel(
                "[yellow]No results returned.\n"
                "Make sure you have at least one API key configured in config.json[/yellow]",
                title="Results"
            )
        )
        return

    for source_name, items in all_results.items():
        if not items:
            continue

        console.rule(f"[bold]{source_name.upper()}[/bold]  ({len(items)} results)")
        shown = items[:top_n]

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        table.add_column("#",      width=3,  style="dim")
        table.add_column("Title",  width=40)
        table.add_column("Query",  width=28, style="cyan")
        table.add_column("Size",   width=12, style="dim")
        table.add_column("URL",    style="blue underline")

        for i, item in enumerate(shown, 1):
            size = f"{item.width}x{item.height}" if item.width else "--"
            title = textwrap.shorten(item.title, 38) if item.title else "(no title)"
            table.add_row(
                str(i),
                title,
                textwrap.shorten(item.query_used, 26),
                size,
                item.page_url or item.url,
            )

        console.print(table)


# ── JSON output ────────────────────────────────────────────────────────────────

def to_json(feats: analyzer.TrackFeatures,
            all_results: dict[str, list[sources.GifResult]]) -> dict:
    import dataclasses
    return {
        "track": dataclasses.asdict(feats),
        "results": {
            src: [dataclasses.asdict(r) for r in items]
            for src, items in all_results.items()
        },
    }


# ── HTML report ───────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Beat GIF Finder — Results</title>
<style>
  body {{ background:#0d0f18; color:#ccc; font-family:monospace; padding:2rem; }}
  h1   {{ color:#7ecfff; }}
  h2   {{ color:#b56cff; border-bottom:1px solid #333; padding-bottom:.3em; }}
  .meta {{ color:#888; font-size:.85em; margin-bottom:1.5rem; }}
  .tags span {{ background:#1e2235; border-radius:4px; padding:2px 8px;
               margin:2px; display:inline-block; font-size:.8em; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:1rem; margin-bottom:2rem; }}
  .card {{ background:#141720; border-radius:8px; padding:.8rem;
           width:220px; overflow:hidden; }}
  .card img  {{ width:100%; border-radius:4px; height:130px; object-fit:cover; }}
  .card a    {{ color:#7ecfff; font-size:.75em; word-break:break-all; }}
  .card .src {{ font-size:.7em; color:#888; margin-top:.3em; }}
  .card .q   {{ font-size:.72em; color:#b56cff; }}
  .bars .row {{ display:flex; align-items:center; gap:.5rem; margin:.2rem 0; }}
  .bars .label {{ width:120px; color:#aaa; font-size:.85em; }}
  .bars .bar  {{ height:8px; background:#7ecfff; border-radius:2px; }}
  .bars .val  {{ color:#888; font-size:.8em; }}
</style>
</head>
<body>
<h1>Beat GIF Finder</h1>
<div class="meta">
  <strong>BPM:</strong> {bpm:.1f} ({bpm_label}) &nbsp;|&nbsp;
  <strong>Key:</strong> {key} &nbsp;|&nbsp;
  <strong>Instruments:</strong> {instruments} &nbsp;|&nbsp;
  <strong>Source:</strong> {source}
</div>

<h2>Audio Features</h2>
<div class="bars">
{bars}
</div>

<h2>Mood Tags</h2>
<div class="tags">{mood_tags}</div>

<h2>Scene Tags</h2>
<div class="tags">{scene_tags}</div>

<h2>Search Queries Used</h2>
<div class="tags">{queries}</div>

{sections}
</body>
</html>
"""

def _bar_html(label: str, value: float) -> str:
    pct = int(value * 100)
    w   = int(value * 200)
    return (f'<div class="row"><span class="label">{label}</span>'
            f'<div class="bar" style="width:{w}px"></div>'
            f'<span class="val">{pct}%</span></div>')

def save_html(feats: analyzer.TrackFeatures,
              all_results: dict[str, list[sources.GifResult]],
              out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.html")

    bars_html = "\n".join([
        _bar_html("Energy",       feats.energy),
        _bar_html("Valence",      feats.valence),
        _bar_html("Danceability", feats.danceability),
        _bar_html("Bass weight",  feats.bass_weight),
        _bar_html("Harmonic",     feats.harmonic_weight),
        _bar_html("Percussive",   feats.percussive_weight),
    ])

    def tag_span(t: str) -> str:
        return f'<span>{t}</span>'

    sections_html = ""
    for src, items in all_results.items():
        if not items:
            continue
        cards = ""
        for item in items[:12]:
            preview = item.preview or item.url
            link    = item.page_url or item.url
            title   = textwrap.shorten(item.title or "(no title)", 50)
            cards += (
                f'<div class="card">'
                f'<img src="{preview}" alt="{title}" loading="lazy">'
                f'<div class="q">query: {item.query_used}</div>'
                f'<a href="{link}" target="_blank">{title}</a>'
                f'<div class="src">{src}</div>'
                f'</div>'
            )
        sections_html += f"<h2>{src.upper()}</h2><div class='grid'>{cards}</div>\n"

    html = HTML_TEMPLATE.format(
        bpm       = feats.bpm,
        bpm_label = feats.bpm_label,
        key       = feats.key,
        instruments = ", ".join(feats.instruments_detected),
        source    = feats.source,
        bars      = bars_html,
        mood_tags = "".join(tag_span(t) for t in feats.mood_tags),
        scene_tags = "".join(tag_span(t) for t in feats.scene_tags),
        queries   = "".join(tag_span(q) for q in feats.search_queries[:12]),
        sections  = sections_html,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Beat GIF Finder — find GIFs that match your track's energy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio", nargs="?",
                        help="Audio file to analyze (mp3/wav/flac/ogg). "
                             "Omit for demo mode.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Seconds of audio to analyze (default: 30)")
    parser.add_argument("--limit",   type=int, default=5,
                        help="Max GIF results per query per source (default: 5)")
    parser.add_argument("--queries", type=int, default=6,
                        help="Number of search queries to use (default: 6)")
    parser.add_argument("--json",    action="store_true",
                        help="Print raw JSON output")
    parser.add_argument("--save",    metavar="DIR",
                        help="Save an HTML report to this directory")
    args = parser.parse_args()

    t0 = time.perf_counter()
    config = load_config()

    # ── Step 1: Analyze ────────────────────────────────────────────────────────
    console.rule("[bold cyan]Beat GIF Finder[/bold cyan]")

    if args.audio:
        console.print(f"[bold]Analyzing:[/bold] {args.audio}")
        try:
            feats = analyzer.analyze(args.audio, duration=args.duration)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
    else:
        console.print("[yellow]No audio file provided — running in demo mode.[/yellow]")
        feats = analyzer.analyze(None)

    print_track_features(feats)

    # ── Step 2: Search ─────────────────────────────────────────────────────────
    queries = feats.search_queries[:args.queries]
    console.print(f"[bold]Fetching GIF suggestions[/bold] using {len(queries)} queries...\n")

    all_results = sources.fetch_all(queries, config, limit_per_query=args.limit)

    # ── Step 3: Output ─────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps(to_json(feats, all_results), indent=2))
    else:
        print_results(all_results, top_n=args.limit)

        elapsed = time.perf_counter() - t0
        total   = sum(len(v) for v in all_results.values())
        console.print(f"\n[dim]Done in {elapsed:.1f}s -- {total} total suggestions[/dim]")

    if args.save:
        path = save_html(feats, all_results, args.save)
        console.print(f"\n[green]HTML report saved -> {path}[/green]")


if __name__ == "__main__":
    main()
