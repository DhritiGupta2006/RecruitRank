"""
Stage 4 — Weighted Score Combiner

Runs: Inline in rank.py — pure numpy/pandas, no ML.
Formula:
base_score = (
    0.35 * title_career_score
  + 0.30 * skills_score
  + 0.20 * experience_score
  + 0.10 * location_edu_score
  + 0.05 * embedding_score
)
"""

import numpy as np
import pandas as pd
from typing import List

def compute_weighted_scores(
    features_df: pd.DataFrame,
    career_embeddings: np.ndarray,
    skills_embeddings: np.ndarray,
    jd_vector: np.ndarray,
    embedding_cids: List[str]
) -> pd.DataFrame:
    """
    Computes the combined embedding score and the final base_score.
    
    Args:
        features_df: DataFrame containing candidate features (from Stage 2).
                     Must have columns: candidate_id, title_career_score,
                     skills_score, experience_score, location_edu_score.
        career_embeddings: (N, 384) L2-normalized array.
        skills_embeddings: (N, 384) L2-normalized array.
        jd_vector: (384,) L2-normalized JD array.
        embedding_cids: List of candidate_ids matching the embedding rows.
        
    Returns:
        pd.DataFrame: The features_df with added 'embedding_score' and 'base_score' columns.
    """
    # 1. Compute embedding similarities
    # Dot product of normalized vectors gives cosine similarity
    career_sim = career_embeddings @ jd_vector
    skills_sim = skills_embeddings @ jd_vector
    
    # Combined embedding score
    # Per instructions.md Section 5.1 and architecture.md: Fixed formula
    emb_scores = 0.65 * career_sim + 0.35 * skills_sim
    
    # 2. Create a DataFrame for the embeddings to ensure safe joining
    emb_df = pd.DataFrame({
        "candidate_id": embedding_cids,
        "embedding_score": emb_scores
    })
    
    # 3. Merge with features
    df = features_df.merge(emb_df, on="candidate_id", how="inner")
    
    # 4. Compute the weighted base score
    # Fixed weights per instructions.md Section 5.1
    df["base_score"] = (
        0.35 * df["title_career_score"] +
        0.30 * df["skills_score"] +
        0.20 * df["experience_score"] +
        0.10 * df["location_edu_score"] +
        0.05 * df["embedding_score"]
    )
    
    return df

# Simple standalone test
if __name__ == "__main__":
    import os
    import time
    
    print(f"[{time.strftime('%H:%M:%S')}] Stage 4: Weighted Score Combiner — start")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")
    
    features_path = os.path.join(artifacts_dir, "features.parquet")
    career_path = os.path.join(artifacts_dir, "career_embeddings.npy")
    skills_path = os.path.join(artifacts_dir, "skills_embeddings.npy")
    jd_path = os.path.join(artifacts_dir, "jd_vector.npy")
    
    # Check if artifacts exist
    if all(os.path.exists(p) for p in [features_path, career_path, skills_path, jd_path]):
        import json
        
        # Load artifacts
        features_df = pd.read_parquet(features_path)
        career_emb = np.load(career_path)
        skills_emb = np.load(skills_path)
        jd_vec = np.load(jd_path)
        
        with open(os.path.join(artifacts_dir, "candidate_ids.json"), "r") as f:
            cids = json.load(f)
            
        # Compute scores
        scored_df = compute_weighted_scores(features_df, career_emb, skills_emb, jd_vec, cids)
        
        print("\nScore Summary:")
        print(scored_df[["base_score", "embedding_score"]].describe())
        
        print("\nTop 5 Candidates by Base Score:")
        top5 = scored_df.sort_values("base_score", ascending=False).head(5)
        for _, row in top5.iterrows():
            print(f"  {row['candidate_id']}: {row['base_score']:.4f}")
            
    else:
        print("Missing artifacts. Run Stage 2 and Stage 3 first.")
        
    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 4: Weighted Score Combiner — complete")
