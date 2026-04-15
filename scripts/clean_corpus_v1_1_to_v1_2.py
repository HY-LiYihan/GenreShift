"""Clean corpus from v1.1 to v1.2 using Kimi k2.5 LLM.

Cleans both news article text and academic paper text fields by removing
dateline prefixes, noise paragraphs, newlines, and other artifacts.
Reads API key from .env file. Supports resume on interruption.
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Explicitly load .env from the project root (one level up from scripts/)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

_api_key = os.environ.get("MOONSHOT_API_KEY", "")
if not _api_key:
    raise RuntimeError(f".env not found or MOONSHOT_API_KEY missing (looked at {_env_path})")

client = OpenAI(
    api_key=_api_key,
    base_url="https://api.moonshot.cn/v1",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = PROJECT_ROOT / "logs" / "clean_corpus_v1_1_to_v1_2.lock"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

NEWS_SYSTEM_PROMPT = """\
You are a text cleaning assistant for a linguistics corpus. Clean the given news article paragraphs and return only the cleaned result as a JSON array of strings.

Rules:
1. Remove dateline prefix at the start of the first paragraph only (e.g. "AUSTIN, Texas —", "WASHINGTON —", "CITY —"). Keep everything after the dash.
2. Remove Science News reference markers: (SN: ...), (SN Online: ...), (SN: date)
3. Replace \\n within a paragraph with a single space
4. Delete entire paragraphs that are noise — do not keep them at all:
   - Multimedia placeholders: "View the video", "Story continues after video", "Story continues after graphic"
   - Byline attributions: "— Author Name", "—Science News, ..."
   - Image position markers: "(above)", "(left)", "(right)"
   - Microsoft Word XML artifacts containing "false" and "MicrosoftInternetExplorer"
   - Any paragraph with fewer than 8 words that is not a complete meaningful sentence
5. Do NOT alter the wording, meaning, or factual content of any kept paragraph.

Input: JSON array of paragraph strings
Output: JSON array of cleaned paragraph strings (shorter than input if noise paragraphs were removed)

Example input:
["WASHINGTON —Scientists have discovered a new species.", "The finding(SN: 3/12/20) was published last week.", "View the video", "It changes our understanding of evolution."]

Example output:
["Scientists have discovered a new species.", "The finding was published last week.", "It changes our understanding of evolution."]
"""

ACADEMIC_SYSTEM_PROMPT = """\
You are a text cleaning assistant for an academic paper corpus. Clean the given academic paper paragraphs and return only the cleaned result as a JSON array of strings.

Rules:
1. Replace \\n within a paragraph with a single space
2. Delete entire paragraphs that are noise — do not keep them at all:
   - Standalone URLs or web addresses (paragraphs that are just a URL)
   - Table remnants containing multiple pipe characters "|"
   - Publication metadata: volume/issue numbers, magazine dates, journal headers
   - Supplementary material descriptions: "This PDF file includes: Materials and Methods..."
   - Any paragraph with fewer than 8 words that is not a complete meaningful sentence
3. Do NOT alter the wording, meaning, or factual content of any kept paragraph.
4. Keep paragraphs with scientific content even if they contain formulas, units, or citations.

Input: JSON array of paragraph strings
Output: JSON array of cleaned paragraph strings

Example input:
["Lead (Pb) represents one of the major metallic contaminants.", "MAGAZINE ISSUE: VOL. 93 NO. #3", "www.sciencemag.org/cgi/content/full/DC1", "Terms of Use | Privacy Center | Report a Vulnerability", "Results are shown in Fig. 3."]

Example output:
["Lead (Pb) represents one of the major metallic contaminants.", "Results are shown in Fig. 3."]
"""


# ---------------------------------------------------------------------------
# LLM cleaning helpers
# ---------------------------------------------------------------------------

BATCH_SIZE = 1  # paragraphs per API call for easier debugging
MODEL = "kimi-k2-thinking"
MAX_WORKERS = max(1, int(os.environ.get("CLEAN_CORPUS_MAX_WORKERS", "32")))
LOG_PATH = PROJECT_ROOT / "logs" / "clean_corpus.log"
LOG_PATH.parent.mkdir(exist_ok=True)
LOG_LOCK = threading.Lock()


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with LOG_LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _acquire_lock() -> None:
    """Prevent multiple copies of the script from running at once."""
    if LOCK_PATH.exists():
        lock_text = LOCK_PATH.read_text(encoding="utf-8").strip()
        match = re.search(r"pid=(\d+)", lock_text)
        if match:
            lock_pid = int(match.group(1))
            try:
                os.kill(lock_pid, 0)
            except ProcessLookupError:
                LOCK_PATH.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    "Another cleaning process appears to be running. "
                    f"Remove {LOCK_PATH} only if that process has already stopped. "
                    f"Current lock info: {lock_text}"
                )
        else:
            raise RuntimeError(
                "A lock file already exists but could not be parsed. "
                f"Please check {LOCK_PATH}: {lock_text}"
            )

    LOCK_PATH.write_text(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")


def _release_lock() -> None:
    """Release the singleton lock file."""
    LOCK_PATH.unlink(missing_ok=True)


def _call_llm_batch(system_prompt: str, paragraphs: list[str], label: str = "", batch_idx: int = 0) -> list[str]:
    """Call the LLM on a single batch of paragraphs (up to BATCH_SIZE).

    Args:
        system_prompt: Cleaning rules prompt.
        paragraphs: Batch of paragraph strings.
        label: Short label for logging.
        batch_idx: Batch number for logging.

    Returns:
        Cleaned paragraph list.

    Raises:
        ValueError: If the response cannot be parsed as a JSON array.
    """
    user_content = json.dumps(paragraphs, ensure_ascii=False)
    _log(f"  -> {label} batch {batch_idx}: sending {len(paragraphs)} para")
    last_exc = None
    for attempt in range(1, 4):
        t0 = time.time()
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                timeout=240,
            )
            elapsed = time.time() - t0
            raw = completion.choices[0].message.content.strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON array in response: {raw[:200]}")
            result = json.loads(match.group())
            _log(f"  <- {label} batch {batch_idx}: got {len(result)} para in {elapsed:.1f}s")
            return result
        except Exception as exc:
            last_exc = exc
            wait = attempt * 5
            _log(f"  [WARN] {label} batch {batch_idx} attempt {attempt}/3 failed ({exc})")
            print(f"    [WARN] {label} batch {batch_idx} attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                _log(f"  [WARN] {label} batch {batch_idx}: retrying in {wait}s")
                time.sleep(wait)
    raise last_exc


def _call_llm(system_prompt: str, paragraphs: list[str], label: str = "") -> list[str]:
    """Call Kimi LLM to clean paragraphs, splitting into batches if needed.

    Args:
        system_prompt: The system prompt describing cleaning rules.
        paragraphs: List of paragraph strings to clean.
        label: Short description for logging (e.g. "news" or "pdf[0]").

    Returns:
        Cleaned list of paragraph strings.
    """
    t0 = time.time()
    batches = [paragraphs[i:i + BATCH_SIZE] for i in range(0, len(paragraphs), BATCH_SIZE)]
    worker_count = min(MAX_WORKERS, len(batches))

    if worker_count == 1:
        result = []
        for idx, batch in enumerate(batches):
            result.extend(_call_llm_batch(system_prompt, batch, label=label, batch_idx=idx + 1))
    else:
        def _clean_one(index_and_batch: tuple[int, list[str]]) -> tuple[int, list[str]]:
            idx, batch = index_and_batch
            cleaned = _call_llm_batch(system_prompt, batch, label=label, batch_idx=idx + 1)
            return idx, cleaned

        ordered_results: list[list[str]] = [[] for _ in batches]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for idx, cleaned in executor.map(_clean_one, enumerate(batches)):
                ordered_results[idx] = cleaned

        result = []
        for cleaned in ordered_results:
            result.extend(cleaned)

    elapsed = time.time() - t0
    _log(
        f"  == {label}: processed {len(paragraphs)} paras in {elapsed:.1f}s "
        f"with {worker_count} worker(s)"
    )
    return result


def clean_news_text(paragraphs: list[str]) -> list[str]:
    """Clean news article text paragraphs.

    Args:
        paragraphs: Raw paragraph list from the news text field.

    Returns:
        Cleaned paragraph list, or original on failure.
    """
    if not paragraphs:
        return paragraphs
    try:
        return _call_llm(NEWS_SYSTEM_PROMPT, paragraphs, label="news")
    except Exception as exc:
        print(f"    [WARN] news text cleaning failed: {exc}")
        return paragraphs


def clean_academic_text(paragraphs: list[str], index: int = 0) -> list[str]:
    """Clean academic paper text paragraphs.

    Args:
        paragraphs: Raw paragraph list from source_pdf[*].text field.
        index: Index of this source_pdf entry, used for logging.

    Returns:
        Cleaned paragraph list, or original on failure.
    """
    if not paragraphs:
        return paragraphs
    try:
        return _call_llm(ACADEMIC_SYSTEM_PROMPT, paragraphs, label=f"pdf[{index}]")
    except Exception as exc:
        print(f"    [WARN] pdf[{index}] text cleaning failed: {exc}")
        return paragraphs


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path) -> None:
    """Process a single corpus JSON file from v1.1 to v1.2.

    Saves progress after every record so the job can be safely interrupted
    and resumed. A sidecar file <output>.progress.json tracks how many
    records have been cleaned; on resume, already-cleaned records are
    loaded from the partial output file.

    Args:
        input_path: Path to the v1.1 JSON file.
        output_path: Path to write the cleaned v1.2 JSON file.
    """
    progress_path = output_path.with_suffix(".progress.json")

    print(f"\n{'=' * 60}")
    print(f"File: {input_path.name}")

    with open(input_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    # Resume: load already-cleaned records if progress file exists
    start_index = 0
    if progress_path.exists() and output_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        start_index = progress.get("completed", 0)
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Resuming from record {start_index}/{len(source_data)}")
    else:
        data = source_data

    remaining = len(data) - start_index
    print(f"Records to process: {remaining} (total: {len(data)})")
    print(f"{'=' * 60}")

    bar = tqdm(range(start_index, len(data)), initial=0, total=remaining,
               unit="rec", dynamic_ncols=True)

    for i in bar:
        record = data[i]
        title = record.get("title", "")[:60]
        bar.set_description(f"rec {i + 1}/{len(data)}: {title}")

        # Clean news text
        if "text" in record and isinstance(record["text"], list):
            record["text"] = clean_news_text(record["text"])

        # Clean each source_pdf entry's text
        for j, pdf_entry in enumerate(record.get("source_pdf", [])):
            if "text" in pdf_entry and isinstance(pdf_entry["text"], list):
                pdf_entry["text"] = clean_academic_text(pdf_entry["text"], index=j)

        # Save after every record
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"completed": i + 1, "total": len(data)}, f)

    bar.close()

    # Cleanup progress sidecar on successful completion
    progress_path.unlink(missing_ok=True)
    print(f"\n[OK] Completed: {output_path.name}")


def main() -> None:
    """Run the corpus cleaning pipeline."""
    _acquire_lock()
    try:
        v1_1_dir = PROJECT_ROOT / "corpus" / "v1.1" / "data"
        v1_2_dir = PROJECT_ROOT / "corpus" / "v1.2" / "data"

        json_files = sorted(f for f in v1_1_dir.glob("*.json") if f.name != "example.json")
        print(f"Found {len(json_files)} files to process")

        for idx, json_file in enumerate(json_files, 1):
            output_file = v1_2_dir / json_file.name
            progress_file = output_file.with_suffix(".progress.json")

            if output_file.exists() and not progress_file.exists():
                print(f"[{idx}/{len(json_files)}] Skipping {json_file.name} (already complete)")
                continue

            if output_file.exists() and progress_file.exists():
                with open(progress_file, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                if progress.get("completed", 0) >= progress.get("total", 0) > 0:
                    progress_file.unlink(missing_ok=True)
                    print(f"[{idx}/{len(json_files)}] Skipping {json_file.name} (progress already complete)")
                    continue

            print(f"[{idx}/{len(json_files)}] Starting {json_file.name}")
            process_file(json_file, output_file)

        print(f"\n{'=' * 60}")
        print("All files complete!")
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
