"""
Audio analyzer — extracts BPM, instruments, mood, and generates search tags.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ── Mood / descriptor mappings ─────────────────────────────────────────────────

BPM_DESCRIPTORS: list[tuple[float, str, list[str]]] = [
    # (max_bpm, label, aesthetic_tags)
    (60,  "glacial",    ["ambient", "drone", "meditation", "slow motion"]),
    (75,  "slow",       ["chill", "late night", "half-asleep", "drifting"]),
    (90,  "lofi",       ["lofi", "laid back", "cozy", "rainy day"]),
    (105, "mid-tempo",  ["mellow", "flowing", "smooth", "indie"]),
    (120, "upbeat",     ["upbeat", "groovy", "feel-good", "summer"]),
    (140, "energetic",  ["energetic", "hype", "bounce", "party"]),
    (180, "fast",       ["fast", "intense", "rush", "adrenaline"]),
    (999, "frantic",    ["frantic", "chaos", "rave", "breakneck"]),
]

KEY_MOODS: dict[str, list[str]] = {
    # major keys → brighter tags
    "C major":  ["bright", "clean", "open"],
    "G major":  ["warm", "natural", "folk"],
    "D major":  ["triumphant", "bold", "golden"],
    "A major":  ["joyful", "radiant", "pop"],
    "E major":  ["shiny", "electric"],
    "F major":  ["tender", "pastoral", "soft"],
    "Bb major": ["majestic", "full", "brass"],
    "Eb major": ["heroic", "cinematic"],
    # minor keys → darker tags
    "A minor":  ["melancholy", "moody", "dark"],
    "E minor":  ["sad", "somber", "introspective"],
    "D minor":  ["brooding", "heavy", "dramatic"],
    "G minor":  ["dark", "tense", "noir"],
    "C minor":  ["serious", "powerful", "intense"],
    "B minor":  ["mysterious", "haunted"],
    "F# minor": ["deep", "dark", "gothic"],
}

INSTRUMENT_TAGS: dict[str, list[str]] = {
    "piano":      ["piano keys", "classical", "concert hall"],
    "guitar":     ["guitar riff", "strings", "acoustic"],
    "bass":       ["bass drop", "sub bass", "deep bass"],
    "drums":      ["drum beat", "percussion", "rhythm"],
    "synth":      ["synthesizer", "electronic", "retro synth"],
    "strings":    ["strings", "orchestral", "cinematic"],
    "vocal":      ["vocals", "voice", "singer"],
    "brass":      ["brass", "trumpet", "jazz horn"],
    "ambient":    ["ambient texture", "atmospheric", "soundscape"],
}

SCENE_MAP: dict[str, list[str]] = {
    # (bpm_label, darkness) → scene suggestions
    ("lofi",     "dark"):   ["rain window night", "desk lamp study", "neon city rain"],
    ("lofi",     "bright"): ["cafe window day", "cozy room sunlight", "morning coffee"],
    ("slow",     "dark"):   ["city lights night", "empty street rain", "moon fog"],
    ("slow",     "bright"): ["sunset field", "golden hour", "slow clouds"],
    ("upbeat",   "bright"): ["city timelapse", "skate park", "cherry blossom"],
    ("upbeat",   "dark"):   ["underground club", "neon lights", "street art"],
    ("energetic","dark"):   ["lightning storm", "night drive fast", "crowd energy"],
    ("energetic","bright"): ["festival crowd", "sunrise run", "ocean waves crashing"],
    ("mid-tempo","dark"):   ["subway night", "dark cafe", "film noir street"],
    ("mid-tempo","bright"): ["afternoon stroll", "bookstore", "rooftop day"],
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TrackFeatures:
    bpm:              float = 90.0
    bpm_label:        str   = "lofi"
    bpm_confidence:   float = 0.0

    key:              str   = "unknown"
    key_confidence:   float = 0.0

    # 0..1 scores
    energy:           float = 0.5
    valence:          float = 0.5   # brightness / positivity
    danceability:     float = 0.5
    bass_weight:      float = 0.5
    harmonic_weight:  float = 0.5
    percussive_weight: float = 0.5

    instruments_detected: list[str] = field(default_factory=list)
    mood_tags:            list[str] = field(default_factory=list)
    scene_tags:           list[str] = field(default_factory=list)
    search_queries:       list[str] = field(default_factory=list)

    duration: float = 0.0
    source:   str   = "unknown"   # "audio" | "demo"


# ── Key detection helpers ──────────────────────────────────────────────────────

_PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

_MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,
                            2.52,5.19,2.39,3.66,2.29,2.88])
_MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,
                            2.54,4.75,3.98,2.69,3.34,3.17])

def _krumhansl_key(chroma_mean: np.ndarray) -> tuple[str, float]:
    """Estimate key via Krumhansl-Kessler tonal hierarchy."""
    best_score, best_key = -np.inf, "C major"
    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        major_r = np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1]
        minor_r = np.corrcoef(rotated, _MINOR_PROFILE)[0, 1]
        if major_r > best_score:
            best_score, best_key = major_r, f"{_PITCH_CLASSES[i]} major"
        if minor_r > best_score:
            best_score, best_key = minor_r, f"{_PITCH_CLASSES[i]} minor"
    # Confidence: 0..1 from correlation
    confidence = float(np.clip((best_score + 1) / 2, 0, 1))
    return best_key, confidence


# ── Instrument inference from spectral features ────────────────────────────────

def _infer_instruments(y: np.ndarray, sr: int,
                       bass_weight: float,
                       harmonic_weight: float,
                       percussive_weight: float,
                       spectral_centroid_mean: float,
                       spectral_bandwidth_mean: float) -> list[str]:
    instruments = []

    if percussive_weight > 0.45:
        instruments.append("drums")
    if bass_weight > 0.5:
        instruments.append("bass")
    if harmonic_weight > 0.55:
        if spectral_centroid_mean < 1500:
            instruments.append("piano")
        elif spectral_centroid_mean < 3000:
            instruments.append("guitar")
        else:
            instruments.append("synth")
    if spectral_bandwidth_mean > 3000 and harmonic_weight > 0.4:
        instruments.append("strings")
    if spectral_centroid_mean > 4000 and harmonic_weight < 0.4:
        instruments.append("synth")

    # Zero-crossing rate — high = vocal / noise-like
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    if 0.06 < zcr < 0.18 and harmonic_weight > 0.4:
        instruments.append("vocal")
    if zcr < 0.03 and harmonic_weight > 0.5:
        instruments.append("ambient")

    return instruments if instruments else ["unknown"]


# ── Tag builders ───────────────────────────────────────────────────────────────

def _bpm_label(bpm: float) -> str:
    for max_bpm, label, _ in BPM_DESCRIPTORS:
        if bpm <= max_bpm:
            return label
    return "frantic"

def _build_mood_tags(feats: TrackFeatures) -> list[str]:
    tags: list[str] = []

    # BPM aesthetic tags
    for max_bpm, label, atags in BPM_DESCRIPTORS:
        if feats.bpm <= max_bpm:
            tags.extend(atags[:2])
            break

    # Key-derived tags
    key_tags = KEY_MOODS.get(feats.key, [])
    tags.extend(key_tags[:2])

    # Instrument tags
    for inst in feats.instruments_detected[:3]:
        tags.extend(INSTRUMENT_TAGS.get(inst, [])[:1])

    # Energy / valence
    if feats.energy > 0.7:
        tags.append("intense")
    elif feats.energy < 0.35:
        tags.append("soft")

    if feats.valence > 0.65:
        tags.append("bright")
    elif feats.valence < 0.35:
        tags.append("dark")

    return list(dict.fromkeys(tags))  # deduplicate, preserve order

def _build_scene_tags(feats: TrackFeatures) -> list[str]:
    darkness = "bright" if feats.valence > 0.5 else "dark"
    key = (feats.bpm_label, darkness)
    scenes = SCENE_MAP.get(key, [])
    # Fallback: try just bpm_label
    if not scenes:
        for (bl, _), s in SCENE_MAP.items():
            if bl == feats.bpm_label:
                scenes = s
                break
    return scenes[:3] if scenes else ["city lights", "abstract motion", "nature"]

def _build_search_queries(feats: TrackFeatures) -> list[str]:
    """Generate ranked search queries for GIF/video platforms."""
    queries: list[str] = []

    bpm_range = f"{int(feats.bpm - 5)}-{int(feats.bpm + 5)} bpm"

    # Scene-first queries (best for visual search)
    for scene in feats.scene_tags[:3]:
        queries.append(scene)

    # Mood + instrument combos
    if feats.instruments_detected and feats.mood_tags:
        inst = feats.instruments_detected[0]
        mood = feats.mood_tags[0]
        queries.append(f"{mood} {inst} aesthetic")

    # Lofi specific
    if feats.bpm_label in ("lofi", "slow", "mid-tempo"):
        for scene in feats.scene_tags[:2]:
            queries.append(f"lofi {scene}")
        queries.append("lofi aesthetic gif")

    # BPM descriptor + mood
    for mood in feats.mood_tags[:3]:
        queries.append(f"{mood} aesthetic")

    # Key mood
    key_tags = KEY_MOODS.get(feats.key, [])
    if key_tags:
        queries.append(f"{key_tags[0]} vibes")

    # Instrument scene
    for inst in feats.instruments_detected[:2]:
        queries.append(f"{inst} aesthetic gif")

    return list(dict.fromkeys(queries))  # deduplicate


# ── Main analysis entry point ──────────────────────────────────────────────────

def analyze(path: str | None, duration: float = 30.0) -> TrackFeatures:
    """
    Analyze an audio file and return a TrackFeatures object.
    If path is None, returns demo features.
    """
    if path is None or not HAS_LIBROSA:
        return _demo_features()

    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    y, sr = librosa.load(path, duration=duration, mono=True)
    hop = 512

    feats = TrackFeatures(source="audio", duration=len(y) / sr)

    def _f(v) -> float:
        """Safely convert any numpy scalar / 0-d array to Python float."""
        return float(np.asarray(v).flat[0])

    # BPM ──────────────────────────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    feats.bpm = _f(tempo) or 90.0
    feats.bpm_label = _bpm_label(feats.bpm)

    # Key ──────────────────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma_mean = chroma.mean(axis=1)
    feats.key, feats.key_confidence = _krumhansl_key(chroma_mean)

    # Harmonic / percussive separation ────────────────────────────────────────
    y_harm, y_perc = librosa.effects.hpss(y)
    h_energy = _f(np.sqrt(np.mean(y_harm ** 2)))
    p_energy = _f(np.sqrt(np.mean(y_perc ** 2)))
    total = (h_energy + p_energy) or 1.0
    feats.harmonic_weight   = h_energy / total
    feats.percussive_weight = p_energy / total

    # RMS energy ───────────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    feats.energy = float(np.clip(_f(np.mean(rms)) * 10, 0, 1))

    # Spectral features ────────────────────────────────────────────────────────
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    cent_mean = _f(np.mean(cent))
    bw_mean   = _f(np.mean(bw))

    # Valence proxy: bright spectral centroid = higher valence
    feats.valence = float(np.clip(cent_mean / 6000, 0, 1))

    # Bass weight ──────────────────────────────────────────────────────────────
    S = np.abs(librosa.stft(y, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs < 200
    bass_e = _f(S[bass_mask].mean())
    full_e = _f(S.mean()) or 1.0
    feats.bass_weight = min(bass_e / full_e * 3, 1.0)

    # Danceability proxy: beat strength regularity
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    ac = librosa.autocorrelate(onset_env, max_size=sr // hop)
    beat_idx = min(int(60 * sr / hop / feats.bpm), len(ac) - 1)
    ac0 = _f(ac[0]) or 1.0
    feats.danceability = float(np.clip(_f(ac[beat_idx]) / ac0, 0, 1))

    # Instruments ──────────────────────────────────────────────────────────────
    feats.instruments_detected = _infer_instruments(
        y, sr,
        feats.bass_weight,
        feats.harmonic_weight,
        feats.percussive_weight,
        cent_mean, bw_mean
    )

    # Tags + queries ───────────────────────────────────────────────────────────
    feats.mood_tags     = _build_mood_tags(feats)
    feats.scene_tags    = _build_scene_tags(feats)
    feats.search_queries = _build_search_queries(feats)

    return feats


def _demo_features() -> TrackFeatures:
    feats = TrackFeatures(
        bpm=87.0, bpm_label="lofi",
        key="A minor", key_confidence=0.78,
        energy=0.42, valence=0.28,
        danceability=0.61,
        bass_weight=0.68, harmonic_weight=0.58, percussive_weight=0.42,
        instruments_detected=["piano", "bass", "drums"],
        duration=0.0, source="demo"
    )
    feats.mood_tags     = _build_mood_tags(feats)
    feats.scene_tags    = _build_scene_tags(feats)
    feats.search_queries = _build_search_queries(feats)
    return feats
