from __future__ import annotations

import argparse
from pathlib import Path

from .api import run_server
from .adapters import Gemma3Adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation Storm local server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug-export-prompt")
    parser.add_argument("--debug-export-dir", default="tmp/extraction_check")
    parser.add_argument("--include-special-tokens", action="store_true")
    args = parser.parse_args()

    if args.debug_export_prompt:
        adapter = Gemma3Adapter()
        result = adapter.export_extraction_check(
            prompt=args.debug_export_prompt,
            include_special_tokens=args.include_special_tokens,
            output_dir=Path(args.debug_export_dir),
        )
        print(f"Extraction check written to {result['output_dir']}")
        for file_path in result["files"]:
            print(file_path)
        return

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
