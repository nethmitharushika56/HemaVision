from pathlib import Path
from collections import Counter, defaultdict
import re

ROOT = Path(
    "data/raw/dataset2-master/dataset2-master/images"
)

CLASSES = [
    "EOSINOPHIL",
    "LYMPHOCYTE",
    "MONOCYTE",
    "NEUTROPHIL"
]


def get_group(filename):
    """
    Example:
        _10_1234.jpeg -> 10
        _2_9876.jpeg  -> 2
    """
    match = re.match(
        r"_(\d+)_",
        filename
    )

    if match:
        return match.group(1)

    return None


for split in ["TRAIN", "TEST"]:

    print("\n" + "=" * 60)
    print(split)
    print("=" * 60)

    for class_name in CLASSES:

        folder = ROOT / split / class_name

        groups = []

        for image in folder.glob("*.jpeg"):

            group = get_group(
                image.name
            )

            if group is not None:
                groups.append(group)

        counts = Counter(groups)

        print(
            f"\n{class_name}"
        )

        print(
            "Unique groups:",
            len(counts)
        )

        print(
            "Largest groups:",
            counts.most_common(10)
        )