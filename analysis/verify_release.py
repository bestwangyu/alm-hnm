from __future__ import annotations

import csv
import hashlib
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


errors = []
manifest = ROOT / "SHA256SUMS"
for line in manifest.read_text(encoding="utf-8").splitlines():
    expected, relative = line.split("  ", 1)
    path = ROOT / relative
    if not path.is_file():
        errors.append(f"missing: {relative}")
    elif sha256(path) != expected:
        errors.append(f"checksum mismatch: {relative}")

archives = list((ROOT / "results/archives").glob("*.tar.gz"))
if len(archives) not in (0, 6):
    errors.append(f"expected 0 code-only or 6 full-evidence archives, found {len(archives)}")
for path in archives:
    sidecar = Path(str(path) + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256(path) != expected:
        errors.append(f"archive checksum mismatch: {path.name}")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
    for member in members:
        parts = PurePosixPath(member.name).parts
        if member.name.startswith("/") or ".." in parts or member.issym() or member.islnk():
            errors.append(f"unsafe archive member: {path.name}:{member.name}")
        if member.name.endswith(".pth"):
            errors.append(f"unexpected model weight: {path.name}:{member.name}")

expected_rows = {
    "results/source_data/figure2_ablation_summary.csv": 5,
    "results/source_data/figure3_robustness_summary.csv": 12,
    "results/source_data/figure4_generalization_paired.csv": 40,
    "results/source_data/figure5_loss_response.csv": 16,
    "results/source_data/figure6_convergence_repeats.csv": 15,
    "results/summaries/baseline_compare/baseline_compare_reconstructed.csv": 16,
}
for relative, expected in expected_rows.items():
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if count != expected:
        errors.append(f"row count mismatch: {relative}: {count} != {expected}")

if errors:
    print("RELEASE VERIFICATION FAILED")
    print("\n".join(f"- {error}" for error in errors))
    sys.exit(1)
mode = "full-evidence" if archives else "code-only"
print(f"RELEASE VERIFICATION PASSED ({mode}): {len(archives)} archives and {len(expected_rows)} tables checked")
