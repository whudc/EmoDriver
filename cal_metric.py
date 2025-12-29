import os
import pandas as pd
from pathlib import Path
import json

# =========================
# 配置
# =========================
# folder_path = "simulation_save_root_path/simulation/asyncdriver_self_2"
folder_path = "simulation_save_root_path/simulation/asyncdriver_true_async_0.5s"


metrics = [
    'score',
    'drivable_area_compliance',
    'driving_direction_compliance',
    'ego_is_comfortable',
    'ego_is_making_progress',
    'no_ego_at_fault_collisions',
    'speed_limit_compliance',
    'time_to_collision_within_bound'
]

# =========================
# 收集 parquet 文件（按真实目录结构）
# =========================
parquet_files = []

root = Path(folder_path)

for scenario_dir in root.iterdir():
    if not scenario_dir.is_dir():
        continue

    base_dir = (
        scenario_dir
        / "qwen4drive_async_experiment_0.5s"
        / "closed_loop_reactive_agents"
    )

    if not base_dir.exists():
        continue

    for time_dir in base_dir.iterdir():
        if not time_dir.is_dir():
            continue

        metric_dir = time_dir / "aggregator_metric"
        if not metric_dir.exists():
            continue

        for parquet_file in metric_dir.glob("*.parquet"):
            parquet_files.append(parquet_file)

print(f"Found {len(parquet_files)} parquet files")

# =========================
# 聚合统计
# =========================
all_results = {
    "individual_files": [],
    "average_metrics": {}
}

for file_path in parquet_files:
    try:
        df = pd.read_parquet(file_path)

        # 保持你原有逻辑
        df = df.iloc[:-2]

        file_metrics = df[metrics].mean(numeric_only=True).to_dict()
        file_metrics = {k: float(v) for k, v in file_metrics.items()}

        file_result = {
            "scenario": file_path.parents[4].name,   # behind_long_vehicle
            "timestamp": file_path.parents[1].name,  # 2025.10.24.10.35.47
            "file_name": file_path.name,
            "file_path": str(file_path),
            "metrics": file_metrics
        }

        all_results["individual_files"].append(file_result)

    except Exception as e:
        print(f"⚠️ Failed to read {file_path}: {e}")

# =========================
# 计算总体平均
# =========================
if all_results["individual_files"]:
    metrics_df = pd.DataFrame(
        [f["metrics"] for f in all_results["individual_files"]]
    )

    avg_metrics = metrics_df.mean().to_dict()
    all_results["average_metrics"] = {k: float(v) for k, v in avg_metrics.items()}

    output_json_path = os.path.join(folder_path, "aggregated_metrics_summary.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Aggregated metrics saved to: {output_json_path}")

    print("\n=== Average Metrics Across All Scenarios ===")
    for k, v in all_results["average_metrics"].items():
        print(f"{k}: {v:.4f}")

    print(f"\n=== Summary ===")
    print(f"Total files processed: {len(all_results['individual_files'])}")

else:
    print("❌ No valid parquet files found.")
