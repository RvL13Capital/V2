"""
Create backup of AIv4 directory, handling Windows reserved filenames.
"""

import os
import shutil
from pathlib import Path
import datetime

def create_backup():
    """Create a backup of the AIv4 directory."""
    source_dir = Path("AIv4")
    backup_dir = Path(f"AIv4_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' not found!")
        return False

    print(f"Creating backup from: {source_dir}")
    print(f"Creating backup to: {backup_dir}")
    print("This may take a few minutes...")

    copied_files = 0
    skipped_files = 0
    errors = []

    # Create backup directory
    backup_dir.mkdir(exist_ok=True)

    # Walk through source directory
    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path
        rel_path = Path(root).relative_to(source_dir)
        dest_dir = backup_dir / rel_path

        # Create directory structure
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        for file in files:
            # Skip reserved Windows filenames
            if file.upper() in ['NUL', 'CON', 'PRN', 'AUX', 'COM1', 'COM2', 'COM3', 'COM4',
                                'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
                                'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']:
                print(f"  Skipping reserved filename: {rel_path / file}")
                skipped_files += 1
                continue

            src_file = Path(root) / file
            dest_file = dest_dir / file

            try:
                shutil.copy2(src_file, dest_file)
                copied_files += 1

                # Progress indicator
                if copied_files % 100 == 0:
                    print(f"  Copied {copied_files} files...")

            except Exception as e:
                errors.append(f"Error copying {src_file}: {e}")
                skipped_files += 1

    # Summary
    print("\n" + "="*70)
    print("BACKUP COMPLETE")
    print("="*70)
    print(f"Source: {source_dir.absolute()}")
    print(f"Backup: {backup_dir.absolute()}")
    print(f"Files copied: {copied_files}")
    print(f"Files skipped: {skipped_files}")

    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

    # Check key files
    print("\nVerifying key files in backup:")
    key_files = [
        "label_patterns_rolling_window.py",
        "output/patterns_labeled_rolling_20251028_212315_deduped.parquet",
        "COMPLETE_RULE_SET.md"
    ]

    for key_file in key_files:
        backup_file = backup_dir / key_file
        if backup_file.exists():
            size = backup_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {key_file} ({size:.1f} MB)")
        else:
            print(f"  ✗ {key_file} NOT FOUND")

    print(f"\nBackup created successfully at: {backup_dir.absolute()}")
    return True

if __name__ == "__main__":
    create_backup()