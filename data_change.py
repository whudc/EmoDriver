import json
from typing import Any


def replace_text(obj: Any,
                 old: str = "Nevigation instructions",
                 new: str = "Navigation instructions") -> Any:
    """
    Recursively replace text in a JSON-like object.
    """
    if isinstance(obj, dict):
        return {k: replace_text(v, old, new) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_text(item, old, new) for item in obj]
    elif isinstance(obj, str):
        return obj.replace(old, new)
    else:
        return obj


def fix_navigation_instruction(json_path: str, save_path: str = None):
    """
    Read JSON file, replace 'Nevigation instructions' with
    'Navigation instructions', and save back.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = replace_text(data)

    output_path = save_path if save_path else json_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished processing: {output_path}")


if __name__ == "__main__":
    # 示例用法
    fix_navigation_instruction("data/stage1_val_20k_processed.json", "data/stage1_val_20k_processed_fixed.json")
