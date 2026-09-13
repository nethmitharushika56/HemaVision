from pathlib import Path

ROOT = Path(
    "data/raw/dataset2-master/dataset2-master/images"
)

CLASSES = [
    "EOSINOPHIL",
    "LYMPHOCYTE",
    "MONOCYTE",
    "NEUTROPHIL"
]

SPLITS = [
    "TRAIN",
    "TEST",
    "TEST_SIMPLE"
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

print("=" * 60)
print("HEMAVISION DATASET SPLITS")
print("=" * 60)

for split in SPLITS:

    split_path = ROOT / split

    print(f"\n{split}")
    print("-" * 40)

    total = 0

    if not split_path.exists():
        print("Folder does not exist.")
        continue

    for class_name in CLASSES:

        class_path = split_path / class_name

        if not class_path.exists():
            print(f"{class_name:<15}: folder missing")
            continue

        images = [
            f for f in class_path.iterdir()
            if f.is_file()
            and f.suffix.lower() in EXTENSIONS
        ]

        count = len(images)
        total += count

        print(f"{class_name:<15}: {count}")

    print(f"{'TOTAL':<15}: {total}")