#!/usr/bin/env python3
"""
Skill Signature Extraction Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts lightweight signatures from all SKILL.md files.
Run this once to build the skill_signatures.json index.

Usage:
    python extract_signatures.py --all
    python extract_signatures.py --domain alfworld
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skillnet_ai.compiler_v2.midend.signature_extractor import SignatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract skill signatures")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract from all domains"
    )
    parser.add_argument(
        "--domain",
        choices=["alfworld", "scienceworld", "webshop"],
        help="Extract from specific domain"
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        help="Path to skills directory (auto-detect if not provided)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/skill_signatures.json"),
        help="Output path for signatures JSON"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini for cost efficiency)"
    )

    args = parser.parse_args()

    # Find skills directory
    if not args.skills_dir:
        # Try to auto-detect
        candidates = [
            Path(__file__).parent.parent / "experiments" / "src" / "skills",
            Path.cwd() / "experiments" / "src" / "skills",
        ]
        for candidate in candidates:
            if candidate.exists():
                args.skills_dir = candidate
                break

    if not args.skills_dir or not args.skills_dir.exists():
        print("ERROR: Could not find skills directory. Please specify with --skills-dir")
        sys.exit(1)

    print(f"Using skills directory: {args.skills_dir}")
    print(f"Using model: {args.model}")
    print(f"Output: {args.output}\n")

    # Initialize extractor
    extractor = SignatureExtractor(model=args.model)

    # Determine which domains to process
    if args.all:
        domains = {
            "alfworld": args.skills_dir / "alfworld",
            "scienceworld": args.skills_dir / "scienceworld",
            "webshop": args.skills_dir / "webshop",
        }
        # Filter out non-existent directories
        domains = {k: v for k, v in domains.items() if v.exists()}
    elif args.domain:
        domain_path = args.skills_dir / args.domain
        if not domain_path.exists():
            print(f"ERROR: Domain directory not found: {domain_path}")
            sys.exit(1)
        domains = {args.domain: domain_path}
    else:
        print("ERROR: Must specify either --all or --domain")
        sys.exit(1)

    print(f"Processing {len(domains)} domain(s): {', '.join(domains.keys())}\n")

    # Extract signatures
    try:
        extractor.build_index(domains, args.output)
        print(f"\n✅ SUCCESS! Signatures saved to {args.output}")
        print(f"\nTo use the signatures:")
        print(f"  from skillnet_ai.compiler_v2 import CompilerV2")
        print(f"  compiler = CompilerV2(domain='alfworld', signatures_path='{args.output}')")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
