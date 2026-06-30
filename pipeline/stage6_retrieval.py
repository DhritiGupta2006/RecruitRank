"""
Stage 6 — ANN Retrieval → Top 500

Runs: Inline in rank.py (after Stage 5 behavioral modifier).
Input: DataFrame with 'final_base_score' and 'honeypot_flag' columns.
Output: DataFrame of exactly 500 non-honeypot candidates (the recall set
        passed to Stage 7 cross-encoder for precision reranking).

Architecture Note (architecture.md Stage 6):
    "Sort all non-honeypot candidates by final_base_score descending,
     take top 500. This is not ANN in the traditional sense (no FAISS
     needed at this scale — numpy sort is fast enough on 28K rows).
     The name reflects the conceptual role: recall phase narrowing to
     500 for the precision phase."

Performance target (instructions.md §11.1): < 5 seconds wall clock.
"""

import time
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve_top_500(
    scored_df: pd.DataFrame,
    n: int = 500,
) -> pd.DataFrame:
    """
    Filter out honeypots, sort by final_base_score descending, return top n.

    This is the recall phase of the pipeline. A lightweight score-based
    sort on ~28K rows is all that's needed at this scale — no FAISS or
    ANN index required (see architecture.md Stage 6 note).

    Args:
        scored_df: DataFrame from Stage 5 with columns:
                   'candidate_id', 'final_base_score', 'honeypot_flag'
                   (and all other feature/score columns propagated forward).
        n: Number of candidates to retrieve (default 500, per spec).

    Returns:
        pd.DataFrame: Top n rows sorted by final_base_score descending.
                      Honeypots are excluded. Row count is exactly min(n,
                      len(valid_candidates)).

    Raises:
        AssertionError: If required columns are missing (fail loudly per
                        instructions.md §12.3).
        RuntimeError:   If fewer than 100 valid candidates remain after
                        honeypot exclusion (pipeline cannot produce top 100).
    """
    # --- Input validation ---
    required_cols = {"candidate_id", "final_base_score", "honeypot_flag"}
    missing_cols = required_cols - set(scored_df.columns)
    assert not missing_cols, (
        f"Stage 6: scored_df is missing required columns: {missing_cols}. "
        "Ensure Stage 5 has been run and its output is passed here."
    )

    total_candidates = len(scored_df)
    honeypot_count = scored_df["honeypot_flag"].sum()
    valid_count = total_candidates - honeypot_count

    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 6: ANN Retrieval — "
        f"{total_candidates:,} candidates in, "
        f"{honeypot_count:,} honeypots excluded, "
        f"{valid_count:,} valid candidates to sort."
    )

    # --- Safety check: must have enough valid candidates for top 100 ---
    if valid_count < 100:
        raise RuntimeError(
            f"Stage 6: Only {valid_count} valid (non-honeypot) candidates "
            f"remain after filtering. Cannot produce top 100 output. "
            "Check Stage 1 hard filter thresholds — they may be too aggressive."
        )

    # --- Core logic (architecture.md Stage 6 exact spec) ---
    #     ranked = df[df.honeypot_flag == False].sort_values(
    #                 "final_base_score", ascending=False)
    #     top_500 = ranked.head(500)
    t0 = time.time()

    valid_candidates = scored_df[scored_df["honeypot_flag"] == False]
    ranked = valid_candidates.sort_values("final_base_score", ascending=False)
    top_n = ranked.head(n).reset_index(drop=True)

    elapsed = time.time() - t0

    actual_n = len(top_n)
    bottom_score = top_n["final_base_score"].iloc[-1] if actual_n > 0 else 0.0
    top_score = top_n["final_base_score"].iloc[0] if actual_n > 0 else 0.0

    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 6: ANN Retrieval complete "
        f"in {elapsed:.2f}s. "
        f"Retrieved {actual_n} candidates. "
        f"Score range: [{bottom_score:.4f}, {top_score:.4f}]."
    )

    # Sanity check on output size
    if valid_count >= n:
        assert actual_n == n, (
            f"Stage 6: Expected exactly {n} candidates, got {actual_n}. "
            "This is a bug in the sort/head logic."
        )

    return top_n


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os

    import numpy as np

    from pipeline.stage0_jd_parser import parse_jd
    from pipeline.stage1_filters import apply_hard_filters, apply_honeypot_gate
    from pipeline.stage2_features import extract_features
    from pipeline.stage4_scorer import compute_weighted_scores
    from pipeline.stage5_behavioral import apply_behavioral_modifier

    print(
        f"[{time.strftime('%H:%M:%S')}] "
        "Stage 6: ANN Retrieval — standalone test start"
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")
    sample_path = os.path.join(project_root, "sample_candidates.json")

    career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
    skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
    jd_path = os.path.join(artifacts_dir, "jd_vector.npy")
    ids_path = os.path.join(artifacts_dir, "candidate_ids.json")

    required = [sample_path, career_path, skills_path, jd_path, ids_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"ERROR: Missing files: {missing}")
        print("Run precompute.py (Stages 0-3) first.")
        exit(1)

    # Load raw data
    with open(sample_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  Loaded {len(candidates)} candidates")

    # Load pre-computed artifacts
    career_emb = np.load(career_path)
    skills_emb = np.load(skills_path)
    jd_vec = np.load(jd_path)
    with open(ids_path, "r") as f:
        cids = json.load(f)

    # Run pipeline: Stage 0 → 1 → 2 → 4 → 5 → 6
    jd_parsed = parse_jd()

    filtered = apply_hard_filters(candidates, jd_parsed)
    gated = apply_honeypot_gate(filtered)
    print(
        f"  After Stage 1: {len(gated)} candidates "
        f"({sum(1 for c in gated if c.get('honeypot_flag'))} honeypots)"
    )

    features_df = extract_features(gated, jd_parsed)
    scored_df = compute_weighted_scores(
        features_df, career_emb, skills_emb, jd_vec, cids
    )

    gated_ids = {c["candidate_id"] for c in gated}
    gated_candidates = [c for c in gated if c["candidate_id"] in gated_ids]
    final_df = apply_behavioral_modifier(scored_df, gated_candidates)

    # Stage 6
    top_500 = retrieve_top_500(final_df)

    print(f"\n  Top 10 candidates (Stage 6 output):")
    print(
        top_500.head(10)[
            ["candidate_id", "final_base_score", "base_score",
             "behavioral_modifier"]
        ].to_string(index=False)
    )

    print(
        f"\n[{time.strftime('%H:%M:%S')}] "
        "Stage 6: ANN Retrieval — standalone test complete"
    )
