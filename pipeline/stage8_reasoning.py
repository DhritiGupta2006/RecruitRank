"""
Stage 8 — Grounded Reasoning Generation

Runs: Offline in precompute.py. Output saved to artifacts/reasoning.json.
Scope: Top 100 candidates only (after Stage 7 finalises the ranked list).

Model priority:
  1. Phi-3-mini-4k-instruct (GGUF, via llama-cpp-python) — preferred
  2. TinyLlama-1.1B-Chat   (GGUF, via llama-cpp-python) — fallback
  3. Deterministic template (no LLM) — last resort, always works

Key rules (instructions.md §8.*):
  §8.1  Reasoning is pre-computed offline. rank.py loads reasoning.json —
        it MUST NOT call any LLM at ranking time.
  §8.2  Anti-hallucination check is mandatory. Every skill + company
        mentioned in the reasoning must exist in the candidate's actual profile.
  §8.3  Use fallback template when LLM is unavailable.
  §8.4  Reasoning must never contain invented facts.
  §8.5  Max 300 characters per reasoning string.

Output:
  artifacts/reasoning.json — {candidate_id: reasoning_string}
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from pipeline.constants import COMPETITION_DATE
from pipeline.utils import parse_date


# ---------------------------------------------------------------------------
# Prompt templates (exact spec from architecture.md Stage 8)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a technical recruiter. Write exactly one sentence (max 250 characters) "
    "explaining why this candidate is ranked at their position. You MUST only mention "
    "facts that explicitly appear in the candidate profile provided. Do not invent "
    "skills, companies, or experience. Do not use the words \"seems\", \"appears\", "
    "\"likely\", or \"probably\"."
)

USER_PROMPT_TEMPLATE = """Rank: {rank}
Candidate title: {current_title}
Years of experience: {years_of_experience}
Career history (relevant): {career_summary}
Matched required skills: {matched_required}
Missing required skills: {missing_required}
Open to work: {open_to_work}
Notice period: {notice_period_days} days
Interview completion rate: {interview_completion_rate}

Write one grounded reasoning sentence for this rank."""

# Fallback template (instructions.md §8.3) — used when LLM unavailable or hallucinates.
FALLBACK_TEMPLATE = (
    "{title} with {yoe:.1f} yrs experience; "
    "{n_required} of {total_required} required skills matched; "
    "response rate {response_rate:.0%}."
)

# Words forbidden in reasoning (instructions.md §13 pre-submission checklist)
FORBIDDEN_WORDS = {"seems", "appears", "likely", "probably"}

# GGUF model search paths (checked in order)
_MODEL_SEARCH_DIRS = [
    "models",              # project root / models/
    os.path.expanduser("~/.cache/llama_models"),
    os.path.expanduser("~/.cache/huggingface/hub"),
]

# GGUF filename patterns (checked in order)
_GGUF_PATTERNS = [
    "Phi-3-mini",          # Phi-3-mini-4k-instruct-*.gguf
    "phi-3-mini",
    "TinyLlama",           # TinyLlama-1.1B-Chat-*.gguf
    "tinyllama",
]


# ---------------------------------------------------------------------------
# LLM model loader
# ---------------------------------------------------------------------------

def _find_gguf_model(project_root: str) -> Optional[str]:
    """
    Search known directories for a .gguf file matching priority patterns.
    Returns the first matching path, or None if no model is found.
    """
    search_dirs = [
        os.path.join(project_root, "models"),
        *_MODEL_SEARCH_DIRS[1:],
    ]

    for pattern in _GGUF_PATTERNS:
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            for fname in sorted(os.listdir(search_dir)):
                if fname.endswith(".gguf") and pattern.lower() in fname.lower():
                    return os.path.join(search_dir, fname)
    return None


def load_llm(project_root: str = "."):
    """
    Attempt to load a local GGUF LLM via llama-cpp-python.

    Search order:
      1. Phi-3-mini-4k-instruct GGUF (preferred per architecture.md Stage 8)
      2. TinyLlama-1.1B-Chat GGUF (fallback)

    Returns:
        Llama instance if a model is found and loads successfully,
        otherwise None (triggers fallback template mode).
    """
    model_path = _find_gguf_model(project_root)
    if not model_path:
        print(
            f"[{time.strftime('%H:%M:%S')}] Stage 8: No GGUF model found in "
            f"'models/' directory. Using deterministic fallback template for all "
            f"candidates. To enable LLM reasoning, download a Phi-3-mini or "
            f"TinyLlama GGUF file into the 'models/' directory."
        )
        return None

    try:
        from llama_cpp import Llama

        print(
            f"[{time.strftime('%H:%M:%S')}] Stage 8: "
            f"Loading LLM from '{model_path}'..."
        )
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,          # context window — sufficient for our prompt
            n_threads=4,         # CPU-only; don't over-saturate
            verbose=False,       # suppress llama.cpp output
        )
        print(f"[{time.strftime('%H:%M:%S')}] Stage 8: LLM loaded successfully.")
        return llm

    except ImportError:
        print(
            f"[{time.strftime('%H:%M:%S')}] Stage 8: llama-cpp-python not installed. "
            "Install with: pip install llama-cpp-python. "
            "Using deterministic fallback template."
        )
        return None
    except Exception as exc:
        print(
            f"[{time.strftime('%H:%M:%S')}] Stage 8: WARNING — "
            f"Failed to load LLM from '{model_path}': {exc}. "
            "Using deterministic fallback template."
        )
        return None


# ---------------------------------------------------------------------------
# Profile text builder (for anti-hallucination grounding)
# ---------------------------------------------------------------------------

def build_profile_text(candidate: Dict) -> str:
    """
    Build a single searchable text blob from all factual fields in a
    candidate's profile. Used as the ground-truth corpus for the
    anti-hallucination entity verification check.

    Includes: title, company, headline, summary, career descriptions,
    skill names, education institutions.
    """
    parts: List[str] = []

    profile = candidate.get("profile", {})
    for field in ("current_title", "current_company", "headline", "summary"):
        val = profile.get(field, "")
        if val:
            parts.append(str(val))

    for entry in candidate.get("career_history", []):
        for f in ("title", "company", "description"):
            val = entry.get(f, "")
            if val:
                parts.append(str(val))

    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        if name:
            parts.append(name)

    for edu in candidate.get("education", []):
        for f in ("institution", "degree", "field_of_study"):
            val = edu.get(f, "")
            if val:
                parts.append(str(val))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Anti-hallucination verification (instructions.md §8.2)
# ---------------------------------------------------------------------------

def extract_skill_mentions(
    reasoning: str, known_skills: List[str]
) -> List[str]:
    """
    Find which skills from the known profile skill list are mentioned
    in the reasoning string (case-insensitive substring match).
    Returns the list of matched skill names.
    """
    reasoning_lower = reasoning.lower()
    return [s for s in known_skills if s.lower() in reasoning_lower]


def extract_company_mentions(
    reasoning: str, known_companies: List[str]
) -> List[str]:
    """
    Find which company names from the candidate's career history are
    mentioned in the reasoning string (case-insensitive substring match).
    Returns the list of matched company names.
    """
    reasoning_lower = reasoning.lower()
    return [c for c in known_companies if c.lower() in reasoning_lower]


def verify_reasoning(
    reasoning: str,
    candidate: Dict,
    jd_parsed: Dict,
) -> Tuple[bool, List[str]]:
    """
    Verify that the reasoning string contains no hallucinated facts.

    Per instructions.md §8.2:
      - Every skill mentioned must exist in skills[*].name.
      - Every company mentioned must exist in career_history[*].company.
      - Year/duration claims must be consistent with actual YOE.

    Strategy:
      1. Build a ground-truth text blob from the full profile.
      2. Extract skill-like tokens from reasoning that appear in the JD
         skills list but NOT in the candidate's profile → hallucination.
      3. Extract company-name tokens that appear in common company
         vocabulary but NOT in the candidate's profile → hallucination.
      4. Check year claims against actual YOE ± 1.

    Args:
        reasoning: The generated reasoning string.
        candidate: The full candidate dict.
        jd_parsed: JD dict containing required_skills + nice_to_have_skills.

    Returns:
        (is_valid, hallucinated_entities): True if no hallucinations found.
    """
    profile_text = build_profile_text(candidate).lower()
    reasoning_lower = reasoning.lower()
    hallucinations: List[str] = []

    # --- Check 1: Skills mentioned in reasoning must exist in profile ---
    # Build the full skill set to check against: JD skills ∪ candidate skills
    candidate_skills = [
        s.get("name", "") for s in candidate.get("skills", []) if s.get("name")
    ]
    all_jd_skills = (
        jd_parsed.get("required_skills", [])
        + jd_parsed.get("nice_to_have_skills", [])
    )

    # For each JD skill that appears in the reasoning, verify it's in profile
    for skill in all_jd_skills:
        if skill.lower() in reasoning_lower:
            if skill.lower() not in profile_text:
                hallucinations.append(f"[skill] {skill}")

    # --- Check 2: Companies must exist in career history ---
    known_companies = [
        entry.get("company", "")
        for entry in candidate.get("career_history", [])
        if entry.get("company")
    ]
    # Look for any proper noun (≥5 chars, starts uppercase) in reasoning
    # that looks like a company name but isn't in the profile
    company_candidates = re.findall(r'\b[A-Z][A-Za-z]{4,}(?:\s[A-Z][A-Za-z]+)*\b', reasoning)
    for word in company_candidates:
        if word.lower() not in profile_text and word not in FORBIDDEN_WORDS:
            # Only flag if it's not a generic recruiter word
            generic_words = {
                "Senior", "Engineer", "Candidate", "Manager", "Director",
                "Lead", "Principal", "Staff", "Years", "Experience",
                "Python", "Machine", "Learning", "Systems", "Platform",
                "Developer", "Architect", "Scientist", "Researcher",
                "Models", "Skills", "Products", "Services", "Teams",
            }
            if word not in generic_words:
                hallucinations.append(f"[entity] {word}")

    # --- Check 3: Year claims must be consistent with actual YOE ---
    years_in_reasoning = re.findall(r'(\d+)\s*(?:years?|yrs?)', reasoning_lower)
    claimed_yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    for yr_str in years_in_reasoning:
        yr = int(yr_str)
        if yr > claimed_yoe + 2 or yr > 40:
            hallucinations.append(f"[years] {yr}")

    return len(hallucinations) == 0, hallucinations


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_user_prompt(
    candidate: Dict,
    rank: int,
    jd_parsed: Dict,
) -> str:
    """
    Build the USER_PROMPT_TEMPLATE for a single candidate.
    All values are sourced strictly from the candidate profile.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})

    current_title = profile.get("current_title", "Unknown Title")
    yoe = profile.get("years_of_experience", 0)

    # Career summary: last 2 roles
    career_history = candidate.get("career_history", [])
    career_parts = []
    for entry in career_history[:2]:
        t = entry.get("title", "")
        c = entry.get("company", "")
        d = entry.get("duration_months", 0)
        if t and c:
            career_parts.append(f"{t} at {c} ({d}mo)")
    career_summary = "; ".join(career_parts) if career_parts else "No history"

    # Skill matching
    candidate_skill_names_lower = {
        s.get("name", "").lower() for s in skills if s.get("name")
    }
    required_skills = jd_parsed.get("required_skills", [])
    matched = [s for s in required_skills if s.lower() in candidate_skill_names_lower]
    missing = [s for s in required_skills if s.lower() not in candidate_skill_names_lower]

    # Availability signals
    open_to_work = redrob.get("open_to_work_flag", False)
    notice_period = redrob.get("notice_period_days", 60)
    icr = redrob.get("interview_completion_rate", 0.5)

    return USER_PROMPT_TEMPLATE.format(
        rank=rank,
        current_title=current_title,
        years_of_experience=yoe,
        career_summary=career_summary,
        matched_required=", ".join(matched[:8]) if matched else "None",
        missing_required=", ".join(missing[:5]) if missing else "None",
        open_to_work="Yes" if open_to_work else "No",
        notice_period_days=notice_period,
        interview_completion_rate=f"{icr:.0%}",
    )


# ---------------------------------------------------------------------------
# Fallback template (instructions.md §8.3)
# ---------------------------------------------------------------------------

def generate_fallback_reasoning(
    candidate: Dict,
    rank: int,
    jd_parsed: Dict,
) -> str:
    """
    Generate a deterministic, fact-grounded reasoning string using the
    fallback template. Always correct — never hallucinates.

    Per instructions.md §8.3:
        FALLBACK_TEMPLATE = (
            "{title} with {yoe:.1f} yrs experience; "
            "{n_required} of {total_required} required skills matched; "
            "response rate {response_rate:.0%}."
        )
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Candidate")
    yoe = float(profile.get("years_of_experience", 0))

    candidate_skill_names_lower = {
        s.get("name", "").lower() for s in skills if s.get("name")
    }
    required_skills = jd_parsed.get("required_skills", [])
    n_required = sum(
        1 for s in required_skills if s.lower() in candidate_skill_names_lower
    )
    total_required = len(required_skills)
    response_rate = redrob.get("recruiter_response_rate", 0.5)

    reasoning = FALLBACK_TEMPLATE.format(
        title=title,
        yoe=yoe,
        n_required=n_required,
        total_required=total_required,
        response_rate=response_rate,
    )

    return _clean_reasoning(reasoning)


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def generate_llm_reasoning(
    candidate: Dict,
    rank: int,
    jd_parsed: Dict,
    llm,
) -> Optional[str]:
    """
    Generate a reasoning string using the local GGUF LLM.

    Returns the generated string (cleaned + length-capped), or None if
    the LLM call fails.
    """
    try:
        user_prompt = _build_user_prompt(candidate, rank, jd_parsed)
        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n<|end|>\n"
            f"<|user|>\n{user_prompt}\n<|end|>\n"
            f"<|assistant|>\n"
        )

        response = llm(
            full_prompt,
            max_tokens=120,    # one sentence ≤ 300 chars ≈ ~80 tokens
            temperature=0.1,   # low variance — grounded reasoning
            stop=["<|end|>", "<|user|>", "\n\n"],
            echo=False,
        )
        raw_text: str = response["choices"][0]["text"].strip()
        return _clean_reasoning(raw_text) if raw_text else None

    except Exception as exc:
        # LLM call failed — log and return None (triggers fallback)
        print(
            f"    WARNING: LLM call failed for candidate "
            f"{candidate.get('candidate_id', '?')}: {exc}"
        )
        return None


def _clean_reasoning(text: str) -> str:
    """
    Apply length cap and forbidden word removal per instructions.md §8.5.
    - Max 300 characters (truncate with '...' if over)
    - Remove leading/trailing whitespace
    - Strip surrounding quotes if present
    """
    # Strip quotes
    text = text.strip().strip('"').strip("'").strip()

    # Enforce 300-char limit per instructions.md §8.5
    if len(text) > 300:
        text = text[:297] + "..."

    return text


def _contains_forbidden_words(text: str) -> bool:
    """Return True if any forbidden word appears in the reasoning text."""
    text_lower = text.lower()
    return any(word in text_lower for word in FORBIDDEN_WORDS)


# ---------------------------------------------------------------------------
# Per-candidate reasoning generator (with hallucination retry)
# ---------------------------------------------------------------------------

def generate_one_reasoning(
    candidate: Dict,
    rank: int,
    jd_parsed: Dict,
    llm,
) -> str:
    """
    Generate a grounded reasoning string for a single candidate.

    Flow (per instructions.md §8.2):
      1. If LLM available: generate via LLM.
      2. Verify: check for hallucinations + forbidden words.
      3. If verification fails: regenerate once.
      4. If second attempt also fails: use fallback template.
      5. If LLM unavailable from the start: use fallback template.

    Args:
        candidate: Full candidate dict.
        rank: The candidate's final rank (1–100).
        jd_parsed: Parsed JD dict from Stage 0.
        llm: Llama instance or None (None → immediate fallback).

    Returns:
        A clean, grounded reasoning string, max 300 characters.
    """
    cid = candidate.get("candidate_id", "?")

    if llm is None:
        return generate_fallback_reasoning(candidate, rank, jd_parsed)

    # --- Attempt 1 ---
    reasoning = generate_llm_reasoning(candidate, rank, jd_parsed, llm)

    if reasoning:
        is_valid, hallucinations = verify_reasoning(reasoning, candidate, jd_parsed)
        has_forbidden = _contains_forbidden_words(reasoning)

        if is_valid and not has_forbidden:
            return reasoning  # ✅ first attempt passed

        # Log what went wrong
        if hallucinations:
            print(
                f"    [{cid}] rank={rank}: Hallucination detected "
                f"({hallucinations[:3]}). Regenerating..."
            )
        if has_forbidden:
            print(
                f"    [{cid}] rank={rank}: Forbidden words found. Regenerating..."
            )

        # --- Attempt 2 ---
        reasoning = generate_llm_reasoning(candidate, rank, jd_parsed, llm)
        if reasoning:
            is_valid2, _ = verify_reasoning(reasoning, candidate, jd_parsed)
            has_forbidden2 = _contains_forbidden_words(reasoning)
            if is_valid2 and not has_forbidden2:
                return reasoning  # ✅ second attempt passed

        # Both attempts failed → use fallback
        print(
            f"    [{cid}] rank={rank}: Both LLM attempts failed verification. "
            f"Using fallback template."
        )

    return generate_fallback_reasoning(candidate, rank, jd_parsed)


# ---------------------------------------------------------------------------
# Batch reasoning generator
# ---------------------------------------------------------------------------

def generate_all_reasoning(
    top_100_df: pd.DataFrame,
    candidates_dict: Dict[str, Dict],
    jd_parsed: Dict,
    llm,
) -> Dict[str, str]:
    """
    Generate reasoning strings for all 100 ranked candidates.

    Args:
        top_100_df: DataFrame from Stage 7 with 'candidate_id' and 'rank'
                    columns. Must have exactly 100 rows.
        candidates_dict: {candidate_id: candidate_dict} for profile lookup.
        jd_parsed: Parsed JD dict from Stage 0.
        llm: Llama instance or None (None → fallback template for all).

    Returns:
        Dict {candidate_id: reasoning_string} — 100 entries.

    Raises:
        AssertionError: If top_100_df doesn't have exactly 100 rows.
    """
    assert len(top_100_df) == 100, (
        f"Stage 8: Expected 100 candidates, got {len(top_100_df)}. "
        "Ensure Stage 7 output is passed (exactly top 100)."
    )

    mode = "LLM" if llm is not None else "fallback template"
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 8: Generating reasoning for "
        f"100 candidates (mode: {mode})..."
    )
    t_start = time.time()

    reasoning_dict: Dict[str, str] = {}
    llm_count = fallback_count = 0

    for _, row in top_100_df.iterrows():
        cid: str = row["candidate_id"]
        rank: int = int(row["rank"])

        candidate = candidates_dict.get(cid)
        if candidate is None:
            # Should never happen if pipeline is consistent — fail loudly
            raise KeyError(
                f"Stage 8: candidate_id '{cid}' (rank {rank}) not found in "
                f"candidates_dict. Ensure the same candidate set is used "
                f"throughout the pipeline."
            )

        reasoning = generate_one_reasoning(candidate, rank, jd_parsed, llm)

        # Final safety: assert no forbidden words made it through
        assert not _contains_forbidden_words(reasoning), (
            f"Stage 8: Forbidden word in final reasoning for {cid}: "
            f"'{reasoning}'. This is a bug in generate_one_reasoning."
        )
        # Assert length limit
        assert len(reasoning) <= 300, (
            f"Stage 8: Reasoning for {cid} exceeds 300 chars "
            f"(len={len(reasoning)}). This is a bug in _clean_reasoning."
        )

        reasoning_dict[cid] = reasoning

        # Track fallback usage
        is_fallback = (
            llm is None
            or reasoning == generate_fallback_reasoning(candidate, rank, jd_parsed)
        )
        if is_fallback:
            fallback_count += 1
        else:
            llm_count += 1

        if rank % 10 == 0 or rank == 1:
            print(f"  [{time.strftime('%H:%M:%S')}] rank {rank:3d}/{100}: {cid}")

    elapsed = time.time() - t_start
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 8: Reasoning generation complete "
        f"in {elapsed:.1f}s. "
        f"LLM: {llm_count}, Fallback: {fallback_count}."
    )

    assert len(reasoning_dict) == 100, (
        f"Stage 8: Expected 100 reasoning entries, got {len(reasoning_dict)}."
    )

    return reasoning_dict


# ---------------------------------------------------------------------------
# Artifact save / load
# ---------------------------------------------------------------------------

def save_reasoning(
    reasoning_dict: Dict[str, str],
    artifacts_dir: str = "artifacts",
) -> str:
    """
    Save the reasoning dict to artifacts/reasoning.json.
    rank.py loads this file to populate the CSV 'reasoning' column.

    Returns the path to the saved file.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    output_path = os.path.join(artifacts_dir, "reasoning.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reasoning_dict, f, ensure_ascii=False, indent=2)

    print(
        f"[Stage 8] Saved reasoning.json -> {output_path} "
        f"({len(reasoning_dict)} entries)"
    )
    return output_path


def load_reasoning(artifacts_dir: str = "artifacts") -> Dict[str, str]:
    """
    Load pre-computed reasoning strings from artifacts/reasoning.json.
    Called by rank.py at ranking time — no LLM involved.

    Raises:
        FileNotFoundError: If reasoning.json is missing.
    """
    path = os.path.join(artifacts_dir, "reasoning.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage 8: Missing artifact '{path}'.\n"
            "Fix: run precompute.py to generate reasoning.json offline."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Standalone execution (called from precompute.py)
# ---------------------------------------------------------------------------

def run_stage8(
    top_100_df: pd.DataFrame,
    candidates_dict: Dict[str, Dict],
    jd_parsed: Dict,
    artifacts_dir: str = "artifacts",
    project_root: str = ".",
) -> Dict[str, str]:
    """
    End-to-end Stage 8 wrapper for precompute.py.

    Loads (or skips) the LLM, generates reasoning for all 100 candidates,
    saves reasoning.json, and returns the dict.

    Args:
        top_100_df: Stage 7 output DataFrame (100 rows, 'rank' column present).
        candidates_dict: {candidate_id: full_candidate_dict}.
        jd_parsed: Parsed JD from Stage 0.
        artifacts_dir: Where to save reasoning.json.
        project_root: Project root for GGUF model search.

    Returns:
        {candidate_id: reasoning_string}
    """
    llm = load_llm(project_root=project_root)
    reasoning = generate_all_reasoning(top_100_df, candidates_dict, jd_parsed, llm)
    save_reasoning(reasoning, artifacts_dir=artifacts_dir)
    return reasoning


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print(
        f"[{time.strftime('%H:%M:%S')}] "
        "Stage 8: Reasoning Generation — standalone test start"
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")
    sample_path = os.path.join(project_root, "sample_candidates.json")

    required = [sample_path]
    missing_files = [p for p in required if not os.path.exists(p)]
    if missing_files:
        print(f"ERROR: Missing files: {missing_files}")
        exit(1)

    # Load raw candidates
    with open(sample_path, "r", encoding="utf-8") as f:
        all_candidates = json.load(f)
    candidates_dict = {c["candidate_id"]: c for c in all_candidates}
    print(f"  Loaded {len(candidates_dict)} candidates")

    # Load JD
    from pipeline.stage0_jd_parser import parse_jd
    jd_parsed = parse_jd()

    # Build a mock top_100_df for testing without full pipeline
    # (use first 100 candidate IDs as a stand-in for Stage 7 output)
    top500_path = os.path.join(artifacts_dir, "top_500.parquet")
    if os.path.exists(top500_path):
        top_500 = pd.read_parquet(top500_path)
        mock_top_100 = top_500.head(100).copy()
        mock_top_100["rank"] = range(1, 101)
    else:
        # Fall back to first 100 candidates in sample
        sample_ids = list(candidates_dict.keys())[:100]
        mock_top_100 = pd.DataFrame({
            "candidate_id": sample_ids,
            "final_score": [0.9 - i * 0.005 for i in range(len(sample_ids))],
            "final_base_score": [0.85 - i * 0.004 for i in range(len(sample_ids))],
            "rank": range(1, len(sample_ids) + 1),
        })
        if len(mock_top_100) < 100:
            print(
                f"WARNING: Only {len(mock_top_100)} candidates in sample. "
                "Stage 8 test will use available candidates."
            )
            # Pad to 100 if needed (test only — not production logic)
            while len(mock_top_100) < 100:
                mock_top_100 = pd.concat(
                    [mock_top_100, mock_top_100], ignore_index=True
                ).head(100)
            mock_top_100["rank"] = range(1, 101)

    reasoning = run_stage8(
        top_100_df=mock_top_100,
        candidates_dict=candidates_dict,
        jd_parsed=jd_parsed,
        artifacts_dir=artifacts_dir,
        project_root=project_root,
    )

    print(f"\n  Sample reasoning outputs (first 5):")
    for cid, r in list(reasoning.items())[:5]:
        rank = mock_top_100.loc[
            mock_top_100["candidate_id"] == cid, "rank"
        ].values[0]
        print(f"  [{rank:3d}] {cid}: {r}")
        print(f"        len={len(r)} chars")

    print(
        f"\n[{time.strftime('%H:%M:%S')}] "
        "Stage 8: Reasoning Generation — standalone test complete"
    )
