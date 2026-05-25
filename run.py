"""
run.py — Single entry point to set up and launch the Medical Clinical Assistant.

"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      🏥  Multi-Agent Medical Clinical Assistant              ║
║         Groq · Gemini · PubMed · OpenFDA · ChromaDB         ║
╚══════════════════════════════════════════════════════════════╝
"""


def check_env() -> bool:
    """Verify .env exists and required keys are set."""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌  .env not found. Creating from .env.example ...")
        example = Path(".env.example")
        if example.exists():
            import shutil
            shutil.copy(example, env_path)
            print("✅  .env created. Please open it and add your API keys:\n")
            print("    GROQ_API_KEY   →  https://console.groq.com  (free)")
            print("    GEMINI_API_KEY →  https://aistudio.google.com  (free)\n")
            return False
        else:
            print("❌  .env.example also missing. Please recreate the project.")
            return False

    from dotenv import load_dotenv
    load_dotenv()

    groq_key   = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    issues = []
    if not groq_key or groq_key == "your_groq_api_key_here":
        issues.append("GROQ_API_KEY not set  →  get free key at https://console.groq.com")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        issues.append("GEMINI_API_KEY not set  →  get free key at https://aistudio.google.com")
    if not tavily_key or tavily_key == "your_tavily_api_key_here":
        issues.append("TAVILY_API_KEY not set  →  get free key at https://app.tavily.com  (web search fallback disabled)")

    if issues:
        print("⚠️  Missing API keys in .env:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n   Open .env and add your keys, then run `python run.py` again.\n")
        print("   NOTE: Without GROQ_API_KEY the pipeline will not work.")
        print("         Without GEMINI_API_KEY image analysis will be disabled.\n")
        return False

    print("✅  API keys found.")
    return True


def ensure_dirs():
    """Create required data directories."""
    dirs = ["data/chroma_db", "data/sample_docs", "data"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅  Data directories ready.")


def run_ingestion():
    """Run the document ingestion pipeline."""
    print("\n📚  Ingesting documents into ChromaDB ...")
    result = subprocess.run(
        [sys.executable, "ingest.py", "--sample"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("❌  Ingestion failed. Check logs above.")
        sys.exit(1)
    print("✅  Ingestion complete.\n")


def check_chroma_populated() -> bool:
    """Return True if ChromaDB already has documents."""
    try:
        from rag.vectorstore import count_documents
        count = count_documents()
        print(f"✅  Knowledge base has {count} chunks.")
        return count > 0
    except Exception:
        return False


def launch_streamlit():
    """Launch the Streamlit UI."""
    print("\n🚀  Launching Streamlit UI ...")
    print("   Open your browser at: http://localhost:8501\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py",
         "--server.port", "8501",
         "--server.headless", "false",
         "--browser.gatherUsageStats", "false"],
    )


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="Medical Clinical Assistant launcher")
    parser.add_argument("--ingest", action="store_true", help="Force re-ingest documents")
    parser.add_argument("--check", action="store_true", help="Check config only, don't launch")
    args = parser.parse_args()

    # Step 1: directories
    ensure_dirs()

    # Step 2: env check
    env_ok = check_env()
    if not env_ok:
        if args.check:
            sys.exit(1)
        print("⚠️  Continuing with missing keys — some features may be disabled.\n")

    if args.check:
        check_chroma_populated()
        print("\n✅  Config check complete.")
        return

    # Step 3: ingest if needed
    if args.ingest or not check_chroma_populated():
        run_ingestion()
    else:
        print("   (Skipping ingestion — knowledge base already populated)")
        print("   Run `python run.py --ingest` to re-ingest documents.\n")

    # Step 4: launch UI
    launch_streamlit()


if __name__ == "__main__":
    main()