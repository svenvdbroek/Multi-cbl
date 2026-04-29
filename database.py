"""
Convert all *-street.csv files to a single Parquet file.
Supports resuming after a crash — already-processed files are skipped.

Usage:
  python csv_to_parquet.py                      
  python csv_to_parquet.py "C:/path/to/data"    
  python csv_to_parquet.py "C:/path/to/data" "C:/path/to/output.parquet"
"""

import sys
import re
import json
from pathlib import Path
import polars as pl

# Configuration

DEFAULT_ROOT   = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\data"
DEFAULT_OUTPUT = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\all_streets.parquet"

CHUNK_SIZE = 200   # write a parquet chunk every N files

# Helpers

PATTERN = re.compile(r"(\d{4})-(\d{2})-(.+)-street\.csv$")

def parse_meta(path: Path) -> dict | None:
    m = PATTERN.match(path.name)
    if not m:
        return None
    return {"year": int(m.group(1)), "month": int(m.group(2)), "department": m.group(3)}

def load_progress(checkpoint_dir: Path) -> set[str]:
    """Return set of already-processed file paths."""
    progress_file = checkpoint_dir / "progress.json"
    if progress_file.exists():
        return set(json.loads(progress_file.read_text()))
    return set()

def save_progress(checkpoint_dir: Path, done: set[str]) -> None:
    (checkpoint_dir / "progress.json").write_text(json.dumps(list(done)))

def write_chunk(frames: list, chunk_index: int, checkpoint_dir: Path) -> None:
    chunk_path = checkpoint_dir / f"chunk_{chunk_index:05d}.parquet"
    pl.concat(frames, how="diagonal_relaxed").write_parquet(chunk_path, compression="zstd")
    print(f"  ✓ Saved chunk {chunk_index} → {chunk_path.name}")

# Main

def convert(root: Path, output: Path) -> None:
    checkpoint_dir = output.parent / "_csv_chunks"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(root.rglob("*-street.csv"))
    if not all_files:
        print(f"No *-street.csv files found under: {root}")
        return

    done = load_progress(checkpoint_dir)
    remaining = [f for f in all_files if str(f) not in done]

    print(f"Total files : {len(all_files)}")
    print(f"Already done: {len(done)}")
    print(f"To process  : {len(remaining)}")

    if remaining:
        frames: list[pl.DataFrame] = []
        errors: list[str] = []
        chunk_index = len(list(checkpoint_dir.glob("chunk_*.parquet")))

        for i, f in enumerate(remaining, 1):
            meta = parse_meta(f)
            if meta is None:
                errors.append(f"  Skipped (unexpected name): {f.name}")
                continue
            try:
                df = pl.read_csv(f, infer_schema_length=10_000)
                df = df.with_columns([
                    pl.lit(meta["year"]).alias("year"),
                    pl.lit(meta["month"]).alias("month"),
                    pl.lit(meta["department"]).alias("department"),
                ])
                frames.append(df)
                done.add(str(f))

                # Save chunk + progress every CHUNK_SIZE files
                if len(frames) >= CHUNK_SIZE:
                    write_chunk(frames, chunk_index, checkpoint_dir)
                    save_progress(checkpoint_dir, done)
                    frames = []
                    chunk_index += 1

                if i % 100 == 0 or i == len(remaining):
                    print(f"  [{i}/{len(remaining)}] processed  (total done: {len(done)}/{len(all_files)})")

            except Exception as e:
                errors.append(f"  ERROR reading {f}: {e}")

        # Write any leftover frames
        if frames:
            write_chunk(frames, chunk_index, checkpoint_dir)
            save_progress(checkpoint_dir, done)

        if errors:
            print("\nWarnings / errors:")
            for e in errors:
                print(e)

    # Merge all chunks into one final Parquet
    chunk_files = sorted(checkpoint_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        print("No chunks found to merge.")
        return

    print(f"\nMerging {len(chunk_files)} chunks into final Parquet …")
    combined = pl.concat([pl.read_parquet(c) for c in chunk_files], how="diagonal_relaxed")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output, compression="zstd")

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone!  {len(combined):,} rows  →  {output}  ({size_mb:.1f} MB)")
    print(f"Columns: {combined.columns}")
    print(f"\nYou can now safely delete the temp folder: {checkpoint_dir}")

# Entry point

if __name__ == "__main__":
    root   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_ROOT)
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_OUTPUT)

    convert(root, output)

    print("\n" + "─" * 60)
    print("Example queries:")
    print(EXAMPLE)
