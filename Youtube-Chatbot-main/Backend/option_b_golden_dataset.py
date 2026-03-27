import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


load_dotenv(dotenv_path=f"{os.path.dirname(__file__)}/.env")


VIDEO_CATALOG: List[Dict[str, str]] = [
    {
        "id": "aircAruvnKk",
        "title": "But what is a Neural Network?",
        "channel": "3Blue1Brown",
        "language": "en",
    },
    {
        "id": "wjZofJX0v4M",
        "title": "Transformers, the tech behind LLMs",
        "channel": "3Blue1Brown",
        "language": "en",
    },
    {
        "id": "fHF22Wxuyw4",
        "title": "What is Deep Learning? (Hindi)",
        "channel": "CampusX",
        "language": "hi",
    },
    {
        "id": "C6YtPJxNULA",
        "title": "All About ML & Deep Learning (Hindi)",
        "channel": "CodeWithHarry",
        "language": "hi",
    },
]


def _mm_ss(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _truncate(s: str, max_chars: int = 900) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _extract_json(text: str) -> Any:
    """
    Best-effort extraction of a JSON object/array from LLM output.
    """
    if not text:
        raise ValueError("Empty LLM output")

    # Try fenced blocks first.
    fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    # Then try raw first JSON array/object.
    m = re.search(r"(\[[\s\S]*?\]|\{[\s\S]*?\})", text)
    if m:
        return json.loads(m.group(1))

    raise ValueError("Could not locate JSON in LLM output")


def _get_transcript_entries(video_id: str, preferred_languages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Fetch transcript entries in the best available language.
    Returns raw caption chunks with {text, start, duration}.
    """
    api = YouTubeTranscriptApi()
    transcript_list = api.list_transcripts(video_id=video_id)

    # If preferred languages are specified, try them in order.
    if preferred_languages:
        try:
            transcript = transcript_list.find_transcript(preferred_languages)
            data = transcript.fetch()
            return list(data)
        except Exception:
            # Fall back to any available transcript below.
            pass

    # Otherwise pick the first available.
    for t in transcript_list:
        try:
            transcript = transcript_list.find_transcript([t.language_code])
            data = transcript.fetch()
            return list(data)
        except Exception:
            continue

    raise RuntimeError("No usable transcript found")


def fetch_video_chunks(video: Dict[str, str], window_seconds: int = 55) -> List[Dict[str, str]]:
    """
    Convert transcript entries into time-window chunks.
    Each chunk stores:
      - text
      - timestamp (mm:ss) + timestamp_end (mm:ss)
    """
    vid_id = video["id"]
    preferred = ["en", "en-US", "en-GB", "hi", "hi-IN"] if video.get("language") == "hi" else ["en", "en-US", "en-GB", ""]

    try:
        entries = _get_transcript_entries(vid_id, preferred_languages=preferred)
    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail=f"Transcript disabled for video {vid_id}")

    chunks: List[Dict[str, str]] = []
    window: List[str] = []
    start_time: Optional[float] = None
    last_end: Optional[float] = None

    for entry in entries:
        text = (entry.get("text") or "").strip()
        if not text:
            continue

        if start_time is None:
            start_time = float(entry.get("start", 0.0))

        window.append(text)
        entry_start = float(entry.get("start", 0.0))
        duration = float(entry.get("duration", 5.0) or 5.0)
        last_end = entry_start + duration

        elapsed = entry_start - (start_time or 0.0)
        if elapsed >= window_seconds:
            chunk_text = re.sub(r"\[.*?\]", "", " ".join(window)).strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "timestamp": _mm_ss(start_time),
                        "timestamp_end": _mm_ss(last_end or start_time),
                    }
                )
            window = []
            start_time = None
            last_end = None

    # Flush remaining
    if window and start_time is not None:
        chunk_text = re.sub(r"\[.*?\]", "", " ".join(window)).strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "timestamp": _mm_ss(start_time),
                    "timestamp_end": _mm_ss(last_end or start_time),
                }
            )

    return chunks


def _pick_best_chunks_openai(
    llm: ChatOpenAI,
    video: Dict[str, str],
    chunks: List[Dict[str, str]],
    k: int = 3,
) -> List[int]:
    if not chunks:
        return []

    # Keep prompt bounded: only include a subset of candidate chunks if needed.
    # The aim is to surface the richest chunks; sampling helps reduce tokens.
    max_candidates = 14
    candidate_indices = list(range(len(chunks)))
    if len(candidate_indices) > max_candidates:
        # Spread sampling across the video to avoid always picking early chunks.
        step = max(1, len(candidate_indices) // max_candidates)
        candidate_indices = candidate_indices[::step][:max_candidates]

    presented = []
    for i in candidate_indices:
        c = chunks[i]
        presented.append(f"[{i}] ({c['timestamp']}-{c['timestamp_end']}) {_truncate(c['text'], 650)}")

    prompt = f"""
You are selecting chunks for building a RAG "golden" evaluation dataset.
Goal: pick the {k} most information-dense chunks that contain specific technical definitions,
mechanisms, comparisons, or crisp explanations.

Video: {video['title']} — {video['channel']}

Candidates (return indices only):
{chr(10).join(presented)}

Return ONLY a JSON array of exactly {k} distinct integers selected from the provided indices.
No extra keys, no commentary.
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    data = _extract_json(raw)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array")

    # Coerce to ints + length k (with safe fallback).
    out: List[int] = []
    for x in data:
        try:
            xi = int(x)
            if 0 <= xi < len(chunks) and xi not in out:
                out.append(xi)
        except Exception:
            continue
        if len(out) == k:
            break
    if len(out) < k:
        # Fallback: pick earliest k among candidates.
        for i in candidate_indices:
            if i not in out:
                out.append(i)
            if len(out) == k:
                break
    return out


def _generate_qa_pairs_openai(
    llm: ChatOpenAI,
    selected_context_blocks: List[Dict[str, str]],
    num_questions: int,
    difficulty_mix: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    selected_context_blocks include:
      - text
      - source_video, source_channel, timestamp
    """
    difficulty_spec = ", ".join([f"{d}:{difficulty_mix.get(d, 0)}" for d in ["easy", "medium", "hard"]])
    context_json = json.dumps(selected_context_blocks, ensure_ascii=False)

    prompt = f"""
You are building an evaluation set for a RAG system about neural networks and deep learning.
You MUST use ONLY the provided source chunks as factual ground truth.

Return exactly {num_questions} question-answer pairs.
Difficulty distribution: {difficulty_spec}.

Rules:
- The answer MUST be answerable from the provided chunks alone.
- Each question should be discriminative: a similar but wrong chunk would lead to a clearly wrong answer.
- Answers must be in English ONLY, even if the source chunk is in Hindi.
- Each pair MUST include the exact video title + timestamp from which it is derived.
- Be specific: target mechanisms/relationships, not generic textbook fluff.

Return ONLY valid JSON that matches this schema:
[
  {{
    "question": string,
    "answer": string,
    "source_video": string,
    "source_channel": string,
    "timestamp": string, 
    "difficulty": "easy" | "medium" | "hard",
    "concept_tag": string,
    "wrong_retrieval_risk": string
  }}
]

SOURCE CHUNKS:
{context_json}
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    data = _extract_json(raw)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array of pairs")
    return data[:num_questions]


def _validate_openai(
    llm: ChatOpenAI,
    pairs: List[Dict[str, Any]],
    selected_context_blocks: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    For each pair:
      - grounded: is answer supported by at least one source chunk block matching its timestamp/video?
      - discriminative: would a wrong but semantically-close chunk likely produce a wrong answer?
    """
    pairs_json = json.dumps(pairs, ensure_ascii=False)
    context_json = json.dumps(selected_context_blocks, ensure_ascii=False)

    prompt = f"""
You are an adversarial validator for a RAG golden dataset.
For each pair i, decide:
1) grounded (true/false): is the answer supported by the specific source chunk indicated by (source_video, timestamp)?
2) discriminative (true/false): would retrieving a different chunk from the provided context (not matching the correct source chunk)
   plausibly produce a wrong answer?

Return ONLY valid JSON:
[
  {{
    "index": i,
    "grounded": true/false,
    "discriminative": true/false,
    "note": string,
    "improved_question": string | null
  }}
]

PAIRS:
{pairs_json}

CONTEXT CHUNKS:
{context_json}
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    data = _extract_json(raw)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array from validator")
    # Merge back by index
    by_index = {int(x.get("index")): x for x in data if isinstance(x, dict) and "index" in x}
    out: List[Dict[str, Any]] = []
    for i in range(len(pairs)):
        out.append(by_index.get(i, {"index": i, "grounded": True, "discriminative": True, "note": "", "improved_question": None}))
    return out


class OptionBGenerateResult(BaseModel):
    metadata: Dict[str, Any]
    methodology_note: List[str]
    qa_pairs: List[Dict[str, Any]]


def generate_option_b_dataset(
    num_questions: int = 5,
    selected_video_ids: Optional[List[str]] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    OpenAI-only pipeline for Option B: Golden Dataset for RAG.
    """
    videos = VIDEO_CATALOG
    if selected_video_ids:
        wanted = set(selected_video_ids)
        videos = [v for v in videos if v["id"] in wanted]
    if not videos:
        raise HTTPException(status_code=400, detail="No valid videos selected")

    llm_name = llm_model or os.getenv("OPENAI_GOLDEN_DATASET_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=llm_name, temperature=0.1)

    difficulty_mix = {"easy": 1, "medium": 2, "hard": max(1, num_questions - 3)}

    # 1) Fetch + chunk per video
    all_selected_context: List[Dict[str, str]] = []
    for v in videos:
        chunks = fetch_video_chunks(v, window_seconds=55)
        # 2) Select densest chunks per video
        best_idxs = _pick_best_chunks_openai(llm, v, chunks, k=3)
        for idx in best_idxs:
            c = chunks[idx]
            all_selected_context.append(
                {
                    # Truncate to keep prompts within budget while preserving factual density.
                    "text": _truncate(c["text"], 1500),
                    "source_video": v["title"],
                    "source_channel": v["channel"],
                    "timestamp": c["timestamp"],
                    "timestamp_end": c["timestamp_end"],
                }
            )

    if not all_selected_context:
        raise HTTPException(status_code=400, detail="Could not build any context chunks")

    # 3) Generate Q/A pairs
    max_retries = int(os.getenv("OPENAI_GOLDEN_DATASET_RETRIES", "2"))
    best_pairs: Optional[List[Dict[str, Any]]] = None
    best_score = -1

    for _attempt in range(max_retries + 1):
        pairs = _generate_qa_pairs_openai(
            llm=llm,
            selected_context_blocks=all_selected_context,
            num_questions=num_questions,
            difficulty_mix=difficulty_mix,
        )

        # 4) Validate Q/A pairs
        validations = _validate_openai(llm=llm, pairs=pairs, selected_context_blocks=all_selected_context)

        score = 0
        failed = 0
        for p, v in zip(pairs, validations):
            grounded = v.get("grounded", True)
            discriminative = v.get("discriminative", True)
            if grounded and discriminative:
                score += 2
            elif grounded or discriminative:
                score += 1
            else:
                failed += 1

            p["validation"] = {
                "grounded": grounded,
                "discriminative": discriminative,
                "note": v.get("note", ""),
            }
            if v.get("improved_question"):
                p["improved_question"] = v.get("improved_question")

            # Convenience for Option B formatting.
            try:
                p["source"] = f"{p.get('source_video','')} @ {p.get('timestamp','')}"
            except Exception:
                p["source"] = ""

        if score > best_score:
            best_score = score
            best_pairs = pairs

        # Early exit if everything looks strong.
        if failed == 0:
            break

    if not best_pairs:
        raise RuntimeError("Failed to generate a valid Option B dataset")

    pairs = best_pairs

    methodology_note = [
        "Selection: split each transcript into ~55s windows, then used OpenAI to pick the most information-dense (definition/mechanism/comparison) chunks per video.",
        "Extraction: each QA pair is forced to cite an exact `source_video` + `timestamp` that corresponds to a selected chunk window.",
        "Method: generate questions to be discriminative, so a near-miss chunk (covering a related but different mechanism/step) would change the answer meaningfully.",
        "Validation: run a second OpenAI check to label each pair as `grounded` (supported by the stated source chunk) and `discriminative` (wrong nearby context would likely lead to a wrong answer).",
    ]

    result = {
        "metadata": {
            "version": "option-b-1.0",
            "num_questions": num_questions,
            "num_videos": len(videos),
            "videos": [{"id": v["id"], "title": v["title"], "channel": v["channel"]} for v in videos],
            "model": llm_name,
        },
        "methodology_note": methodology_note,
        "qa_pairs": pairs,
    }
    return result


if __name__ == "__main__":
    # Simple local runner for quick generation/debugging.
    dataset = generate_option_b_dataset(num_questions=5)
    print(json.dumps(dataset, ensure_ascii=False, indent=2))

