from pathlib import Path
import json


splits = ["train", "dev", "test", "other", "invalidated"]
if __name__ == "__main__":
    n_files = {}
    lang_dirs = [d for d in Path("audio").iterdir() if d.is_dir()]
    for lang_dir in lang_dirs:
        lang = lang_dir.name
        n_files[lang] = {}
        for split in splits:
            split_dir = lang_dir / split
            if split_dir.exists():
                n_files_per_split = len(list(split_dir.glob("*.tar")))
            else:
                n_files_per_split = 0
            n_files[lang][split] = n_files_per_split

    with open("n_shards.json", "w") as f:
        json.dump(dict(sorted(n_files.items(), key=lambda x: x[0])), f, ensure_ascii=False, indent=4)
