import os
import pandas as pd
from pathlib import Path
import json

# 输入文件夹路径
folder_path = "/data/DC/AsyncDriver/simulation_save_root_path/async"

# 指标列
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

# 收集所有 parquet 文件路径
parquet_files = list(Path(folder_path).rglob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

# 存放所有文件的结果
all_results = {
    "individual_files": [],
    "average_metrics": {}
}

for file_path in parquet_files:
    try:
        df = pd.read_parquet(file_path)
        df = df.iloc[:-2]

        # 计算每个文件的指标平均值
        file_metrics = df[metrics].mean(numeric_only=True).to_dict()
        
        # 将数值转换为Python原生类型（避免numpy类型）
        file_metrics = {k: float(v) for k, v in file_metrics.items()}
        
        file_result = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "metrics": file_metrics
        }
        
        all_results["individual_files"].append(file_result)

    except Exception as e:
        print(f"⚠️ Failed to read {file_path}: {e}")

# 计算总体平均指标
if all_results["individual_files"]:
    # 提取所有文件的指标数据
    metrics_data = []
    for file_result in all_results["individual_files"]:
        metrics_data.append(file_result["metrics"])
    
    # 计算每个指标的平均值
    metrics_df = pd.DataFrame(metrics_data)
    avg_metrics = metrics_df.mean().to_dict()
    
    # 转换为Python原生类型
    all_results["average_metrics"] = {k: float(v) for k, v in avg_metrics.items()}
    
    # 保存为JSON文件
    output_json_path = os.path.join(folder_path, "aggregated_metrics_summary.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Aggregated metrics saved to: {output_json_path}")
    
    # 打印总体平均
    print("\n=== Average Metrics Across All Scenarios ===")
    for k, v in all_results["average_metrics"].items():
        print(f"{k}: {v:.4f}")
        
    # 打印文件数量统计
    print(f"\n=== Summary ===")
    print(f"Total files processed: {len(all_results['individual_files'])}")
    
else:
    print("❌ No valid parquet files found.")