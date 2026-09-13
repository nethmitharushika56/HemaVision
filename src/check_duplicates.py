from pathlib import Path
import hashlib
from collections import defaultdict

ROOT = Path(
    "data/raw/dataset2-master/dataset2-master/images"
)

SPLITS = ["TRAIN", "TEST", "TEST_SIMPLE"]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def file_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


hash_locations = defaultdict(list)

print("Calculating image hashes...\n")

for split in SPLITS:

    split_path = ROOT / split

    if not split_path.exists():
        continue

    for image in split_path.rglob("*"):

        if (
            image.is_file()
            and image.suffix.lower() in EXTENSIONS
        ):

            h = file_hash(image)

            hash_locations[h].append(
                (split, str(image))
            )


cross_split_duplicates = []

for h, locations in hash_locations.items():

    splits = {
        location[0]
        for location in locations
    }

    if len(splits) > 1:
        cross_split_duplicates.append(
            (h, locations)
        )


print("=" * 60)
print("CROSS-SPLIT DUPLICATES")
print("=" * 60)

print(
    f"\nDuplicate hashes appearing across splits: "
    f"{len(cross_split_duplicates)}"
)

for h, locations in cross_split_duplicates[:20]:

    print("\nDuplicate:")

    for split, path in locations:
        print(f"  [{split}] {path}")