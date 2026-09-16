"""
run_server.py — Launch the Heatwave Risk Prediction API with Uvicorn.

Usage
-----
    python run_server.py                      # default: localhost:8000
    python run_server.py --host 0.0.0.0      # expose on all interfaces
    python run_server.py --port 9000          # custom port
    python run_server.py --no-reload         # disable auto-reload (production)

Or directly with uvicorn:
    uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path (same pattern as main.py)
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the Heatwave Risk Prediction API server."
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--no-reload", action="store_true",
        help="Disable auto-reload (use in production)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker processes (default: 1; reload must be off for >1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Verify model exists before starting
    model_path = _PROJECT_ROOT / "models" / "model.pkl"
    if not model_path.exists():
        print(
            f"\n[WARNING] model.pkl not found at {model_path}\n"
            "  Run `python main.py` first to train and save the model.\n"
            "  The server will start but /predict and /current-weather will return 503.\n"
        )

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Run:\n"
            "  pip install uvicorn[standard]\n"
        )
        sys.exit(1)

    reload = not args.no_reload
    workers = args.workers if not reload else 1  # uvicorn forbids reload + workers > 1

    print(
        f"\n Starting Heatwave API\n"
        f"  URL     : http://{args.host}:{args.port}\n"
        f"  Docs    : http://{args.host}:{args.port}/docs\n"
        f"  Reload  : {'enabled' if reload else 'disabled'}\n"
        f"  Workers : {workers}\n"
    )

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=reload,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
