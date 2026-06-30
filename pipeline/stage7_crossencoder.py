"""
Stage 7 — Cross-Encoder Reranking (Top 500 → Top 100)

Runs: Inline in rank.py. Operates ONLY on the top 500 candidates from Stage 6.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (primary)
       BAAI/bge-reranker-base             (fallback)

Why a cross-encoder here and not earlier?
  Cross-encoders score (query, document) pairs jointly — they are accurate
  but expensive (O(N) forward passes with full attention across both).
  Running this on 100K or 28K candidates would take ~hours on CPU.
  Restricting to top 500 keeps this under 2 minutes (instructions.md §11.1).

Key rules (instructions.md §7.*):
  §7.1  Run only on top 500 candidates — never on the full set.
  §7.2  Primary model: cross-encoder/ms-marco-MiniLM-L-6-v2.
        Fallback:      BAAI/bge-reranker-base. Both must be pre-cached.
  §7.3  Normalize raw logits to [0,1] before blending:
          cross_norm = (scores - min) / (max - min + 1e-9)
  §7.4  Fixed blend formula:
          final_score = 0.55 * final_base_score + 0.45 * cross_norm

Output:
  pd.DataFrame with exactly 100 rows, sorted by final_score descending.
  'rank' column is 1–100.

Cross-encoder input dependency:
  Needs raw career text strings (not embedding vectors) from
  artifacts/career_texts.json — {candidate_id: career_text}.
  Stage 3 save_embeddings() creates this file during precompute.py.
  If it is missing, a FileNotFoundError is raised with exact fix instructions.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from pipeline.stage3_embeddings import load_career_texts  # canonical definition in Stage 3

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

PRIMARY_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FALLBACK_MODEL = "BAAI/bge-reranker-base"
BATCH_SIZE = 32  # instructions.md §7.1 / architecture.md Stage 7


# ---------------------------------------------------------------------------
# Model loader with fallback
# ---------------------------------------------------------------------------

def load_cross_encoder(model_cache_dir: Optional[str] = None):
    """
    Load the cross-encoder model, falling back to BAAI/bge-reranker-base
    if the primary model cannot be loaded.

    Per instructions.md §7.2: both models must be pre-cached locally.
    If neither can be loaded, raises a RuntimeError with clear instructions.

    Args:
        model_cache_dir: Optional local directory where models are cached.

    Returns:
        Tuple of (CrossEncoder instance, model_name_used).
    """
    from sentence_transformers import CrossEncoder

    for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            kwargs: Dict = {}
            if model_cache_dir:
                kwargs["cache_dir"] = model_cache_dir

            print(
                f"[{time.strftime('%H:%M:%S')}] Stage 7: "
                f"Loading cross-encoder '{model_name}'..."
            )
            model = CrossEncoder(model_name, **kwargs)
            print(
                f"[{time.strftime('%H:%M:%S')}] Stage 7: "
                f"Cross-encoder '{model_name}' loaded successfully."
            )
            return model, model_name

        except Exception as exc:
            print(
                f"[{time.strftime('%H:%M:%S')}] Stage 7: WARNING — "
                f"Could not load '{model_name}': {exc}. "
                f"{'Trying fallback...' if model_name == PRIMARY_MODEL else ''}"
            )

    raise RuntimeError(
        "Stage 7: ABORT — Could not load either cross-encoder model.\n"
        f"  Primary  : {PRIMARY_MODEL}\n"
        f"  Fallback : {FALLBACK_MODEL}\n"
        "Both models must be downloaded before running rank.py.\n"
        "Fix: run precompute.py which caches all required models offline."
    )


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------
# Note: load_career_texts is imported from pipeline.stage3_embeddings above.
# Stage 3 is the canonical owner of career_texts.json.


def load_jd_text(artifacts_dir: str = "artifacts") -> str:
    """
    Load the JD embedding text from artifacts/jd_parsed.json.
    This is the query side of every (jd_text, candidate_career_text) pair.
    """
    jd_path = os.path.join(artifacts_dir, "jd_parsed.json")
    if not os.path.exists(jd_path):
        raise FileNotFoundError(
            f"Stage 7: Missing artifact '{jd_path}'.\n"
            "Fix: run precompute.py (Stage 0 — JD parsing) first."
        )
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_parsed = json.load(f)

    jd_text = jd_parsed.get("jd_embedding_text", "")
    if not jd_text:
        raise ValueError(
            "Stage 7: jd_parsed['jd_embedding_text'] is empty. "
            "Re-run Stage 0 to regenerate jd_parsed.json."
        )
    return jd_text


# ---------------------------------------------------------------------------
# Core reranking function
# ---------------------------------------------------------------------------

def rerank_top_100(
    top_500: pd.DataFrame,
    career_texts: Dict[str, str],
    jd_text: str,
    model_cache_dir: Optional[str] = None,
    n_output: int = 100,
) -> pd.DataFrame:
    """
    Run cross-encoder over top 500 candidates and return the top 100.

    Per architecture.md Stage 7:
      1. Build input pairs: (jd_text, candidate_career_text) for each cid.
      2. Score all 500 pairs jointly with batch_size=32.
      3. Normalize to [0,1]:
           cross_norm = (scores - min) / (max - min + 1e-9)
      4. Blend: final_score = 0.55 * final_base_score + 0.45 * cross_norm
      5. Sort descending (tiebreak: candidate_id ascending for reproducibility).
      6. Head(100). Assign rank = 1..100.

    Args:
        top_500: DataFrame from Stage 6 with 'candidate_id' and
                 'final_base_score' columns.
        career_texts: Dict {candidate_id: career_text} from Stage 3 artifact.
        jd_text: The JD embedding text string (query side of each pair).
        model_cache_dir: Optional local path for cached models.
        n_output: Number of candidates to output (default 100, per spec).

    Returns:
        pd.DataFrame with exactly n_output rows, sorted by final_score
        descending. Columns added: 'cross_score', 'cross_norm', 'final_score',
        'rank'.

    Raises:
        AssertionError: On any shape/count invariant violation (fail loudly).
        RuntimeError: If neither cross-encoder model can be loaded.
        KeyError: If candidate IDs in top_500 have no matching career text.
    """
    t_start = time.time()

    # --- Input validation ---
    required_cols = {"candidate_id", "final_base_score"}
    missing_cols = required_cols - set(top_500.columns)
    assert not missing_cols, (
        f"Stage 7: top_500 DataFrame is missing columns: {missing_cols}. "
        "Ensure Stage 6 output is passed here."
    )

    n_in = len(top_500)
    assert n_in > 0, (
        "Stage 7: top_500 is empty. Stage 6 must have produced at least "
        "1 candidate. Check Stage 6 (retrieve_top_500)."
    )
    assert n_in >= n_output, (
        f"Stage 7: top_500 has only {n_in} candidates, "
        f"but n_output={n_output} was requested. "
        "Check Stage 6 — it may not have enough valid candidates."
    )

    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 7: Cross-Encoder Reranking — "
        f"scoring {n_in} candidates (batch_size={BATCH_SIZE})..."
    )

    # --- Build input pairs: (jd_text, candidate_career_text) ---
    cids: List[str] = top_500["candidate_id"].tolist()

    missing_texts = [cid for cid in cids if cid not in career_texts]
    if missing_texts:
        raise KeyError(
            f"Stage 7: {len(missing_texts)} candidate(s) in top_500 have no "
            f"matching career text in career_texts.json. "
            f"First few missing: {missing_texts[:5]}. "
            "This can happen if precompute.py was run on a different candidate "
            "set than rank.py. Re-run precompute.py on the full dataset."
        )

    pairs: List[Tuple[str, str]] = [
        (jd_text, career_texts[cid]) for cid in cids
    ]

    # --- Load cross-encoder model ---
    model, model_name_used = load_cross_encoder(model_cache_dir=model_cache_dir)

    # --- Score all pairs jointly ---
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 7: "
        f"Running {len(pairs)} pairs through '{model_name_used}'..."
    )
    t_score = time.time()

    cross_scores = model.predict(
        pairs, batch_size=BATCH_SIZE, show_progress_bar=True
    )
    cross_scores = np.asarray(cross_scores, dtype=np.float32)

    score_elapsed = time.time() - t_score
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 7: "
        f"Cross-encoder scoring done in {score_elapsed:.1f}s. "
        f"Raw score range: [{cross_scores.min():.4f}, {cross_scores.max():.4f}]."
    )

    # Sanity check — one score per candidate
    assert len(cross_scores) == n_in, (
        f"Stage 7: Expected {n_in} cross-encoder scores, "
        f"got {len(cross_scores)}. This is a bug in the model predict call."
    )

    # --- Normalize to [0,1] per instructions.md §7.3 ---
    # Raw logits from cross-encoders are not on a [0,1] scale.
    # Must normalize before blending with final_base_score.
    cross_norm = (cross_scores - cross_scores.min()) / (
        cross_scores.max() - cross_scores.min() + 1e-9
    )

    # --- Build result DataFrame ---
    result_df = top_500.copy()
    result_df = result_df.reset_index(drop=True)
    result_df["cross_score"] = cross_scores
    result_df["cross_norm"] = cross_norm

    # --- Fixed blend formula per instructions.md §7.4 ---
    # final_score = 0.55 * final_base_score + 0.45 * cross_norm
    result_df["final_score"] = (
        0.55 * result_df["final_base_score"] + 0.45 * result_df["cross_norm"]
    )

    # --- Sort descending; tiebreak by candidate_id ascending ---
    # Per instructions.md §9.3: "Break ties by candidate_id to ensure
    # reproducibility."
    result_df = result_df.sort_values(
        ["final_score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # --- Take top n_output ---
    top_out = result_df.head(n_output).copy()

    # --- Assign rank 1..100 ---
    top_out["rank"] = range(1, n_output + 1)

    # --- Output validation (fail loudly per instructions.md §12.3) ---
    assert len(top_out) == n_output, (
        f"Stage 7: Expected exactly {n_output} rows in output, "
        f"got {len(top_out)}. This is a bug."
    )
    assert top_out["rank"].tolist() == list(range(1, n_output + 1)), (
        "Stage 7: Output ranks are not sequential 1-100. This is a bug."
    )
    assert (top_out["final_score"] >= 0.0).all(), (
        "Stage 7: Some final_score values are negative. "
        "Check final_base_score range and cross_norm computation."
    )
    assert (top_out["final_score"] <= 1.5).all(), (
        "Stage 7: Some final_score values exceed expected range (>1.5). "
        "final_base_score should be in [0,1] and cross_norm in [0,1], "
        "so max blend is ~1.0."
    )

    t_total = time.time() - t_start
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 7: Complete in {t_total:.1f}s. "
        f"Top-{n_output} selected. "
        f"Final score range: "
        f"[{top_out['final_score'].iloc[-1]:.4f}, "
        f"{top_out['final_score'].iloc[0]:.4f}]. "
        f"Model used: {model_name_used}."
    )

    return top_out


# ---------------------------------------------------------------------------
# Convenience wrapper (used by rank.py)
# ---------------------------------------------------------------------------

def run_stage7(
    top_500: pd.DataFrame,
    artifacts_dir: str = "artifacts",
    model_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    End-to-end Stage 7 wrapper for rank.py.

    Loads career_texts and jd_text from artifacts_dir, then runs
    rerank_top_100().

    Args:
        top_500: DataFrame from Stage 6 (retrieve_top_500).
        artifacts_dir: Path to the artifacts directory.
        model_cache_dir: Optional path where cross-encoder models are cached.

    Returns:
        pd.DataFrame with exactly 100 rows, ranked 1–100.
    """
    career_texts = load_career_texts(artifacts_dir=artifacts_dir)
    jd_text = load_jd_text(artifacts_dir=artifacts_dir)

    return rerank_top_100(
        top_500=top_500,
        career_texts=career_texts,
        jd_text=jd_text,
        model_cache_dir=model_cache_dir,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os

    print(
        f"[{time.strftime('%H:%M:%S')}] "
        "Stage 7: Cross-Encoder Reranking — standalone test start"
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")

    # -----------------------------------------------------------------------
    # Option A: Load a pre-saved top_500.parquet (fastest for iteration)
    # -----------------------------------------------------------------------
    top500_path = os.path.join(artifacts_dir, "top_500.parquet")

    if os.path.exists(top500_path):
        print(f"  Loading pre-saved top_500 from {top500_path}")
        top_500 = pd.read_parquet(top500_path)

    # -----------------------------------------------------------------------
    # Option B: Reconstruct top_500 from raw pipeline (full end-to-end)
    # -----------------------------------------------------------------------
    else:
        print(
            "  top_500.parquet not found — running Stages 0-6 to build it..."
        )
        from pipeline.stage0_jd_parser import parse_jd
        from pipeline.stage1_filters import apply_hard_filters, apply_honeypot_gate
        from pipeline.stage2_features import extract_features
        from pipeline.stage4_scorer import compute_weighted_scores
        from pipeline.stage5_behavioral import apply_behavioral_modifier
        from pipeline.stage6_retrieval import retrieve_top_500

        sample_path = os.path.join(project_root, "sample_candidates.json")
        career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
        skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
        jd_vec_path = os.path.join(artifacts_dir, "jd_vector.npy")
        ids_path = os.path.join(artifacts_dir, "candidate_ids.json")

        required = [sample_path, career_path, skills_path, jd_vec_path, ids_path]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            print(f"ERROR: Missing files: {missing}\nRun precompute.py first.")
            exit(1)

        with open(sample_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        career_emb = np.load(career_path)
        skills_emb = np.load(skills_path)
        jd_vec = np.load(jd_vec_path)
        with open(ids_path, "r") as f:
            cids = json.load(f)

        jd_parsed = parse_jd()
        filtered = apply_hard_filters(candidates, jd_parsed)
        gated = apply_honeypot_gate(filtered)
        features_df = extract_features(gated, jd_parsed)
        scored_df = compute_weighted_scores(
            features_df, career_emb, skills_emb, jd_vec, cids
        )
        gated_ids = {c["candidate_id"] for c in gated}
        final_df = apply_behavioral_modifier(
            scored_df, [c for c in gated if c["candidate_id"] in gated_ids]
        )
        top_500 = retrieve_top_500(final_df)

        # Save for future fast iteration
        top_500.to_parquet(top500_path, index=False)
        print(f"  Saved top_500 to {top500_path}")

    # -----------------------------------------------------------------------
    # Run Stage 7
    # -----------------------------------------------------------------------
    top_100 = run_stage7(top_500, artifacts_dir=artifacts_dir)

    print(f"\n  Top 10 Final Rankings (Stage 7 output):")
    display_cols = ["rank", "candidate_id", "final_score",
                    "final_base_score", "cross_norm"]
    print(top_100.head(10)[display_cols].to_string(index=False))

    print(
        f"\n[{time.strftime('%H:%M:%S')}] "
        "Stage 7: Cross-Encoder Reranking — standalone test complete"
    )
