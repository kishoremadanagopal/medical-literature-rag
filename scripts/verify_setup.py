"""
Verify the Step 1 setup: dependencies, config, directories, and API access.

Run this after `pip install -r requirements.txt` and after creating `.env`:

    python scripts/verify_setup.py

Exit code 0 = ready to proceed to Step 2.
Exit code 1 = something needs fixing (details printed above).
"""

import sys
from pathlib import Path

# Make 'config' importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHECK = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = CHECK if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon} {label}{suffix}")
    return ok


def main() -> int:
    print("\n🧬 Medical RAG — Setup Verification\n")
    all_ok = True

    # ----- 1. Core dependencies -----
    print("Dependencies:")
    for pkg, import_name in [
        ("anthropic", "anthropic"),
        ("langchain", "langchain"),
        ("langchain-anthropic", "langchain_anthropic"),
        ("chromadb", "chromadb"),
        ("faiss-cpu", "faiss"),
        ("rank-bm25", "rank_bm25"),
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("biopython", "Bio"),
        ("pydantic-settings", "pydantic_settings"),
        ("streamlit", "streamlit"),
    ]:
        try:
            __import__(import_name)
            check(pkg, True)
        except ImportError as e:
            all_ok &= check(pkg, False, str(e))

    # ----- 2. Config loading -----
    print("\nConfiguration:")
    try:
        from config.config import settings
        check("config/config.py loads", True)

        if settings.anthropic_api_key.startswith("sk-ant-"):
            check("ANTHROPIC_API_KEY set", True)
        else:
            print(f"  {WARN} ANTHROPIC_API_KEY not set  "
                  "(only needed to generate answers; ingestion, indexing and evaluation work without it)")

        ok = "@" in settings.ncbi_email and "example.com" not in settings.ncbi_email
        all_ok &= check(
            "NCBI_EMAIL set to a real address",
            ok,
            "" if ok else "still showing placeholder",
        )
    except Exception as e:
        all_ok &= check("config loads", False, str(e))
        return 1

    # ----- 3. Directories -----
    print("\nDirectories:")
    settings.ensure_directories()
    for path in [
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.chroma_persist_dir,
        settings.faiss_index_dir,
        settings.log_file.parent,
    ]:
        all_ok &= check(str(path.relative_to(Path.cwd())), path.exists())

    # ----- 4. Logging -----
    print("\nLogging:")
    try:
        from config.logging_config import get_logger
        logger = get_logger("verify_setup")
        logger.info("Logger initialized successfully")
        check("logger emits to console + file", True)
    except Exception as e:
        all_ok &= check("logging setup", False, str(e))

    # ----- 5. Claude API ping (optional, costs ~0 tokens) -----
    print("\nClaude API:")
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.claude_model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
        ok = "ok" in resp.content[0].text.lower()
        check(f"reachable ({settings.claude_model})", ok)
    except Exception as e:
        print(f"  {WARN} Claude API not tested: {e}")
        print("     (Step 1 can still pass — just confirm before Step 5.)")

    # ----- Summary -----
    print()
    if all_ok:
        print(f"{CHECK} Step 1 complete. Ready for Step 2 (PubMed ingestion).\n")
        return 0
    else:
        print(f"{FAIL} Fix the issues above before proceeding.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
