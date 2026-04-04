"""
Platform search sources — Reddit, GIPHY, Tenor.

All three work out of the box with zero user configuration.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

import requests

TIMEOUT = 8  # seconds

# Built-in public keys — work without user signup
_BUILTIN_TENOR_KEY = "AIzaSyAyimkuYQYF_FXVALexPuGQctUWRURdCYQ"
_BUILTIN_GIPHY_KEY = "Gc7131jiJuvI7IdN0HZ1D7nh0ow5BU6g"


@dataclass
class GifResult:
    title:      str
    url:        str          # direct GIF / video / image URL
    preview:    str          # thumbnail or small preview URL
    source:     str          # "giphy" | "tenor" | "reddit"
    tags:       list[str] = field(default_factory=list)
    page_url:   str = ""     # link to original post/page
    width:      int = 0
    height:     int = 0
    query_used: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Reddit (no key needed)
# ═══════════════════════════════════════════════════════════════════════════════

REDDIT_SUBS = [
    "LofiHipHop", "lo_fi", "aestheticrain", "cinemagraphs",
    "perfectloops", "unixporn", "analog", "VaporwaveAesthetics",
]

def search_reddit(queries: list[str], limit_per_query: int = 5) -> list[GifResult]:
    results: list[GifResult] = []
    seen: set[str] = set()
    headers = {"User-Agent": "BeatGifFinder/1.0"}
    subs = "+".join(REDDIT_SUBS)

    for query in queries[:5]:
        url = f"https://www.reddit.com/r/{subs}/search.json"
        params = {
            "q":           query,
            "restrict_sr": "1",
            "sort":        "relevance",
            "t":           "year",
            "limit":       limit_per_query * 2,
            "type":        "link",
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            posts = r.json().get("data", {}).get("children", [])
        except Exception as e:
            print(f"  [Reddit] Error for '{query}': {e}")
            continue

        for post in posts:
            d = post.get("data", {})
            pid = d.get("id", "")
            if pid in seen:
                continue

            post_url  = d.get("url", "")
            permalink = "https://reddit.com" + d.get("permalink", "")
            title     = d.get("title", query)
            thumb     = d.get("thumbnail", "")
            if thumb in ("self", "default", "nsfw", ""):
                thumb = ""

            if post_url.endswith(".gif") or post_url.endswith(".gifv"):
                seen.add(pid)
                results.append(GifResult(
                    title=title, url=post_url, preview=thumb,
                    source="reddit", page_url=permalink, query_used=query,
                ))
                continue

            if "v.redd.it" in post_url or d.get("is_video"):
                preview_img = (d.get("preview", {})
                                .get("images", [{}])[0]
                                .get("source", {})
                                .get("url", "").replace("&amp;", "&"))
                seen.add(pid)
                results.append(GifResult(
                    title=title, url=post_url, preview=preview_img or thumb,
                    source="reddit", page_url=permalink, query_used=query,
                ))
                continue

            for domain in ("imgur.com", "gfycat.com", "redgifs.com", "i.imgur.com"):
                if domain in post_url:
                    seen.add(pid)
                    results.append(GifResult(
                        title=title, url=post_url, preview=thumb,
                        source="reddit", page_url=permalink, query_used=query,
                    ))
                    break

        time.sleep(0.15)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GIPHY (built-in key)
# ═══════════════════════════════════════════════════════════════════════════════

def search_giphy(queries: list[str], api_key: str,
                 limit_per_query: int = 5) -> list[GifResult]:
    results: list[GifResult] = []
    seen: set[str] = set()
    base = "https://api.giphy.com/v1/gifs/search"

    for query in queries:
        params = {
            "api_key": api_key,
            "q":       query,
            "limit":   limit_per_query,
            "rating":  "g",
            "lang":    "en",
        }
        try:
            r = requests.get(base, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception as e:
            print(f"  [GIPHY] Error for '{query}': {e}")
            continue

        for item in data:
            gid = item.get("id", "")
            if gid in seen:
                continue
            seen.add(gid)

            images = item.get("images", {})
            orig   = images.get("original", {})
            thumb  = images.get("fixed_width_small", images.get("preview_gif", {}))
            url    = orig.get("url", "")
            if not url:
                continue

            results.append(GifResult(
                title      = item.get("title", query),
                url        = url,
                preview    = thumb.get("url", url),
                source     = "giphy",
                tags       = item.get("title", "").lower().split()[:6],
                page_url   = item.get("url", f"https://giphy.com/gifs/{gid}"),
                width      = int(orig.get("width", 0)),
                height     = int(orig.get("height", 0)),
                query_used = query,
            ))

        time.sleep(0.1)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Tenor (built-in key)
# ═══════════════════════════════════════════════════════════════════════════════

def search_tenor(queries: list[str], api_key: str,
                 limit_per_query: int = 5) -> list[GifResult]:
    results: list[GifResult] = []
    seen: set[str] = set()
    base = "https://tenor.googleapis.com/v2/search"

    for query in queries:
        params = {
            "q":      query,
            "key":    api_key,
            "limit":  limit_per_query,
            "media_filter": "gif,tinygif",
        }
        try:
            r = requests.get(base, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            items = r.json().get("results", [])
        except Exception as e:
            print(f"  [Tenor] Error for '{query}': {e}")
            continue

        for item in items:
            tid = item.get("id", "")
            if tid in seen:
                continue
            seen.add(tid)

            media_formats = item.get("media_formats", {})
            gif_data  = media_formats.get("gif", {})
            tiny_data = media_formats.get("tinygif", {})
            url     = gif_data.get("url", "")
            preview = tiny_data.get("url", url)
            if not url:
                continue

            dims = gif_data.get("dims", [0, 0])
            results.append(GifResult(
                title      = item.get("content_description", query),
                url        = url,
                preview    = preview,
                source     = "tenor",
                tags       = item.get("tags", [])[:6],
                page_url   = item.get("itemurl", ""),
                width      = dims[0] if dims else 0,
                height     = dims[1] if dims else 0,
                query_used = query,
            ))

        time.sleep(0.1)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_all(queries: list[str], config: dict,
              limit_per_query: int = 5) -> dict[str, list[GifResult]]:
    """Run all three sources and return grouped results."""
    results: dict[str, list[GifResult]] = {}

    print("  Searching Reddit...")
    results["reddit"] = search_reddit(queries, limit_per_query)
    print(f"    -> {len(results['reddit'])} results")

    giphy_key = config.get("giphy", {}).get("api_key", "") or _BUILTIN_GIPHY_KEY
    print("  Searching GIPHY...")
    results["giphy"] = search_giphy(queries, giphy_key, limit_per_query)
    print(f"    -> {len(results['giphy'])} results")

    tenor_key = config.get("tenor", {}).get("api_key", "") or _BUILTIN_TENOR_KEY
    print("  Searching Tenor...")
    results["tenor"] = search_tenor(queries, tenor_key, limit_per_query)
    print(f"    -> {len(results['tenor'])} results")

    return results
