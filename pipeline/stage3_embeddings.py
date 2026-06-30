"""
Stage 3 — Dual-Track SBERT Embedding

Runs: Offline pre-computation. Output saved to artifacts/.
Model: sentence-transformers/all-MiniLM-L6-v2 (CPU, ~40MB, 384-dim)

Two embedding tracks:
  Track A: Career Description Embedding — concatenated career history
           "{title} at {company}: {description}" per entry.
  Track B: Skills Embedding — weighted skill text where required skills
           from the JD are repeated 3× to boost their semantic weight.

Also pre-computes the JD vector from jd_parsed["jd_embedding_text"].

All embeddings are L2-normalized (normalize_embeddings=True) so that
cosine similarity = dot product at ranking time.

Output files:
  artifacts/career_embeddings.npy   shape: (N, 384)  float32
  artifacts/skills_embeddings.npy   shape: (N, 384)  float32
  artifacts/jd_vector.npy           shape: (384,)    float32
  artifacts/candidate_ids.json      ordered list of candidate_ids matching
                                    rows in the embedding arrays
"""

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

# Model name — per instructions.md Section 6.1: use all-MiniLM-L6-v2 only.
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 256  # per instructions.md Section 11.3


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_career_text(candidate: Dict) -> str:
    """
    Build the career embedding input text for one candidate.

    Per architecture.md Stage 3, Track A:
        career_text = " ".join(
            f"{entry['title']} at {entry['company']}: {entry['description']}"
            for entry in career_history
        )

    Per instructions.md Section 6.4: embed career descriptions, not just
    skill tags. The primary embedding target is career_history[*].description
    concatenated with title and company context.
    """
    career_history = candidate.get("career_history", [])
    parts = []
    for entry in career_history:
        title = entry.get("title", "")
        company = entry.get("company", "")
        description = entry.get("description", "")
        parts.append(f"{title} at {company}: {description}")
    return " ".join(parts) if parts else ""


def build_skills_text(candidate: Dict, required_skills_lower: set) -> str:
    """
    Build the skills embedding input text for one candidate.

    Per architecture.md Stage 3, Track B:
        skill_text = " ".join(
            ([skill.name] * 3 if skill.name.lower() in required_skills_lower
             else [skill.name])
            for skill in skills
        )

    Required skills are repeated 3× to boost their semantic weight
    in the embedding space.
    """
    skills = candidate.get("skills", [])
    tokens = []
    for skill in skills:
        name = skill.get("name", "")
        if not name:
            continue
        if name.lower() in required_skills_lower:
            tokens.extend([name] * 3)
        else:
            tokens.append(name)
    return " ".join(tokens) if tokens else ""


# ---------------------------------------------------------------------------
# Core embedding logic
# ---------------------------------------------------------------------------

def compute_embeddings(
    candidates: List[Dict],
    jd_parsed: Dict,
    artifacts_dir: str = "artifacts",
    model_cache_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Compute dual-track SBERT embeddings for all candidates + JD vector.

    Args:
        candidates: List of candidate dicts (post Stage-1 filtering).
        jd_parsed: Parsed JD dict from Stage 0 (must contain
                   "required_skills" and "jd_embedding_text").
        artifacts_dir: Directory to save .npy outputs.
        model_cache_dir: Optional directory for caching the SBERT model.

    Returns:
        Tuple of (career_embeddings, skills_embeddings, jd_vector, candidate_ids)
        where:
          career_embeddings: np.ndarray shape (N, 384)
          skills_embeddings: np.ndarray shape (N, 384)
          jd_vector:         np.ndarray shape (384,)
          candidate_ids:     list of N candidate_id strings
    """
    # Import here to avoid import-time cost when this module is just
    # being inspected or when the model isn't needed.
    from sentence_transformers import SentenceTransformer

    n = len(candidates)
    print(f"[{time.strftime('%H:%M:%S')}] Stage 3: Loading SBERT model "
          f"'{SBERT_MODEL_NAME}'...")

    load_kwargs = {}
    if model_cache_dir:
        load_kwargs["cache_folder"] = model_cache_dir

    model = SentenceTransformer(SBERT_MODEL_NAME, **load_kwargs)
    print(f"[{time.strftime('%H:%M:%S')}] Stage 3: Model loaded. "
          f"Embedding {n} candidates (batch_size={BATCH_SIZE})...")

    # --- Build required skills set for skill weighting ---
    required_skills_lower = {
        s.lower() for s in jd_parsed.get("required_skills", [])
    }

    # --- Build text inputs ---
    candidate_ids: List[str] = []
    career_texts: List[str] = []
    skills_texts: List[str] = []

    for candidate in candidates:
        candidate_ids.append(candidate["candidate_id"])
        career_texts.append(build_career_text(candidate))
        skills_texts.append(build_skills_text(candidate, required_skills_lower))

    # --- Track A: Career Description Embeddings (batched) ---
    # Per instructions.md Section 6.5: normalize all embeddings.
    # Per instructions.md Section 11.3: use batch processing.
    print(f"[{time.strftime('%H:%M:%S')}] Stage 3: Track A — "
          f"Career description embeddings ({n} texts)...")
    career_embeddings = model.encode(
        career_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    # Ensure float32 ndarray
    career_embeddings = np.asarray(career_embeddings, dtype=np.float32)
    assert career_embeddings.shape == (n, EMBEDDING_DIM), (
        f"career_embeddings shape mismatch: expected ({n}, {EMBEDDING_DIM}), "
        f"got {career_embeddings.shape}"
    )

    # --- Track B: Skills Embeddings (batched) ---
    print(f"[{time.strftime('%H:%M:%S')}] Stage 3: Track B — "
          f"Skills embeddings ({n} texts)...")
    skills_embeddings = model.encode(
        skills_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    skills_embeddings = np.asarray(skills_embeddings, dtype=np.float32)
    assert skills_embeddings.shape == (n, EMBEDDING_DIM), (
        f"skills_embeddings shape mismatch: expected ({n}, {EMBEDDING_DIM}), "
        f"got {skills_embeddings.shape}"
    )

    # --- JD Vector (single text) ---
    print(f"[{time.strftime('%H:%M:%S')}] Stage 3: Computing JD vector...")
    jd_embedding_text = jd_parsed.get("jd_embedding_text", "")
    if not jd_embedding_text:
        raise ValueError(
            "jd_parsed['jd_embedding_text'] is empty. "
            "Run Stage 0 (JD Parsing) first."
        )
    jd_vector = model.encode(
        jd_embedding_text,
        normalize_embeddings=True,
    )
    jd_vector = np.asarray(jd_vector, dtype=np.float32)
    assert jd_vector.shape == (EMBEDDING_DIM,), (
        f"jd_vector shape mismatch: expected ({EMBEDDING_DIM},), "
        f"got {jd_vector.shape}"
    )

    print(f"[{time.strftime('%H:%M:%S')}] Stage 3: All embeddings computed.")
    return career_embeddings, skills_embeddings, jd_vector, candidate_ids


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------

def save_embeddings(
    career_embeddings: np.ndarray,
    skills_embeddings: np.ndarray,
    jd_vector: np.ndarray,
    candidate_ids: List[str],
    artifacts_dir: str = "artifacts",
) -> None:
    """
    Save all embedding artifacts to disk.

    Files created:
      artifacts/career_embeddings.npy   (N, 384)
      artifacts/skills_embeddings.npy   (N, 384)
      artifacts/jd_vector.npy           (384,)
      artifacts/candidate_ids.json      ordered list of candidate_id strings
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
    skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
    jd_path = os.path.join(artifacts_dir, "jd_vector.npy")
    ids_path = os.path.join(artifacts_dir, "candidate_ids.json")

    np.save(career_path, career_embeddings)
    print(f"[Stage 3] Saved career_embeddings.npy -> {career_path} "
          f"(shape: {career_embeddings.shape})")

    np.save(skills_path, skills_embeddings)
    print(f"[Stage 3] Saved skills_embeddings.npy -> {skills_path} "
          f"(shape: {skills_embeddings.shape})")

    np.save(jd_path, jd_vector)
    print(f"[Stage 3] Saved jd_vector.npy -> {jd_path} "
          f"(shape: {jd_vector.shape})")

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)
    print(f"[Stage 3] Saved candidate_ids.json -> {ids_path} "
          f"({len(candidate_ids)} entries)")


def load_embeddings(
    artifacts_dir: str = "artifacts",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load pre-computed embedding artifacts from disk.

    Returns:
        Tuple of (career_embeddings, skills_embeddings, jd_vector, candidate_ids)

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
    skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
    jd_path = os.path.join(artifacts_dir, "jd_vector.npy")
    ids_path = os.path.join(artifacts_dir, "candidate_ids.json")

    for path in [career_path, skills_path, jd_path, ids_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing embedding artifact: {path}. "
                "Run precompute.py (Stage 3) first."
            )

    career_embeddings = np.load(career_path)
    skills_embeddings = np.load(skills_path)
    jd_vector = np.load(jd_path)

    with open(ids_path, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)

    assert career_embeddings.shape[1] == EMBEDDING_DIM
    assert skills_embeddings.shape[1] == EMBEDDING_DIM
    assert jd_vector.shape == (EMBEDDING_DIM,)
    assert len(candidate_ids) == career_embeddings.shape[0]

    return career_embeddings, skills_embeddings, jd_vector, candidate_ids


def compute_embedding_scores(
    career_embeddings: np.ndarray,
    skills_embeddings: np.ndarray,
    jd_vector: np.ndarray,
) -> np.ndarray:
    """
    Compute combined embedding similarity scores for all candidates.

    Per architecture.md Stage 3:
        career_sim = cosine_similarity(career_vector, jd_vector)
        skills_sim = cosine_similarity(skills_vector, jd_vector)
        embedding_score = 0.65 * career_sim + 0.35 * skills_sim

    Since all vectors are L2-normalized, cosine similarity = dot product.

    Args:
        career_embeddings: (N, 384) normalized career vectors
        skills_embeddings: (N, 384) normalized skill vectors
        jd_vector:         (384,)   normalized JD vector

    Returns:
        np.ndarray of shape (N,) with combined embedding scores.
    """
    # Dot product with normalized vectors = cosine similarity
    career_sim = career_embeddings @ jd_vector   # (N,)
    skills_sim = skills_embeddings @ jd_vector   # (N,)

    # Per architecture.md: embedding_score = 0.65 * career_sim + 0.35 * skills_sim
    # Per instructions.md Section 6.3: this formula is fixed.
    embedding_scores = 0.65 * career_sim + 0.35 * skills_sim

    return embedding_scores


# ---------------------------------------------------------------------------
# Standalone execution: run Stage 3 on sample_candidates.json
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.stage0_jd_parser import parse_jd
    from pipeline.stage1_filters import apply_hard_filters, apply_honeypot_gate

    print(f"[{time.strftime('%H:%M:%S')}] Stage 3: Dual-Track SBERT Embedding — start")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(project_root, "sample_candidates.json")
    artifacts_dir = os.path.join(project_root, "artifacts")

    # --- Stage 0: parse JD ---
    jd_parsed = parse_jd()

    # --- Load candidates ---
    if not os.path.exists(sample_path):
        print(f"ERROR: {sample_path} not found. Cannot run standalone test.")
        exit(1)

    with open(sample_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  Loaded {len(candidates)} candidates from sample_candidates.json")

    # --- Stage 1: filter ---
    filtered = apply_hard_filters(candidates, jd_parsed)
    filtered = apply_honeypot_gate(filtered)
    # Only embed non-honeypot candidates (per architecture: honeypot gate
    # before embedding — no wasted SBERT compute on zero-score candidates)
    valid = [c for c in filtered if not c.get("honeypot_flag", False)]
    print(f"  After Stage 1: {len(valid)} valid candidates "
          f"(filtered from {len(candidates)})")

    # --- Stage 3: embeddings ---
    career_emb, skills_emb, jd_vec, cids = compute_embeddings(
        valid, jd_parsed, artifacts_dir=artifacts_dir
    )

    save_embeddings(career_emb, skills_emb, jd_vec, cids, artifacts_dir=artifacts_dir)

    # --- Quick sanity check ---
    scores = compute_embedding_scores(career_emb, skills_emb, jd_vec)
    print(f"\n  Embedding score stats:")
    print(f"    min:  {scores.min():.4f}")
    print(f"    max:  {scores.max():.4f}")
    print(f"    mean: {scores.mean():.4f}")
    print(f"    std:  {scores.std():.4f}")

    # Top-5 by embedding score
    top5_idx = np.argsort(scores)[::-1][:5]
    print(f"\n  Top 5 candidates by embedding score:")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. {cids[idx]}  score={scores[idx]:.4f}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 3: Dual-Track SBERT Embedding — complete")
