#!/usr/bin/env python3
"""
生成PAD情感状态训练数据

使用方法:
python scripts/generate_pad_training_data.py \
    --input_file data/mixed_decision_driveqa_train_epoch3.json \
    --output_file data/pad_emotion_decision_train.json \
    --num_pad_variations 4
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Tuple
import random
import re

# 预定义的PAD情感状态
PREDEFINED_PAD_STATES = [
    {"pleasure": 0.3, "arousal": 0.0, "dominance": 0.5, "label": "calm"},
    {"pleasure": 0.5, "arousal": 0.8, "dominance": 0.9, "label": "aggressive"},
    {"pleasure": -0.2, "arousal": 0.3, "dominance": -0.3, "label": "cautious"},
    {"pleasure": -0.5, "arousal": 0.7, "dominance": -0.5, "label": "anxious"},
    {"pleasure": 0.7, "arousal": 0.4, "dominance": 0.6, "label": "confident"},
    {"pleasure": -0.3, "arousal": -0.4, "dominance": 0.2, "label": "tired"},
]

def classify_pad_emotion(pad_state: Dict) -> str:
    """根据PAD值分类情感状态"""
    p, a, d = pad_state['pleasure'], pad_state['arousal'], pad_state['dominance']

    if a > 0.5 and d > 0.5:
        return "aggressive"
    elif a < -0.3 or d < -0.3:
        return "cautious"
    elif a > 0.5 and p < 0:
        return "anxious"
    elif p > 0.5 and d > 0.3:
        return "confident"
    elif a < -0.2 and p < 0:
        return "tired"
    else:
        return "calm"

def get_emotion_description(label: str) -> str:
    """获取情感状态描述"""
    descriptions = {
        "calm": "You are in a calm and composed emotional state. Drive smoothly and maintain steady control.",
        "aggressive": "You are in an assertive and confident emotional state. You can drive more dynamically but safely.",
        "cautious": "You are in a cautious and careful emotional state. Prioritize safety and maintain conservative driving.",
        "anxious": "You are in an anxious emotional state. Be extra careful and avoid risky maneuvers.",
        "confident": "You are in a confident and positive emotional state. Drive assertively while maintaining safety.",
        "tired": "You are in a tired and low-energy emotional state. Drive carefully and avoid complex maneuvers."
    }
    return descriptions.get(label, "")

def parse_trajectory_from_text(text: str) -> List[Tuple[float, float]]:
    """从文本中解析轨迹点"""
    # 查找轨迹格式: [(x1, y1), (x2, y2), ...]
    pattern = r'\[\([\d\.\-, ]+\)\]'
    matches = re.findall(pattern, text)

    if not matches:
        return []

    # 解析第一个匹配的轨迹
    traj_str = matches[0]
    # 提取所有数字对
    points = re.findall(r'\(([\d\.\-]+),\s*([\d\.\-]+)\)', traj_str)
    trajectory = [(float(x), float(y)) for x, y in points]

    return trajectory

def adjust_trajectory_by_pad(trajectory: List[Tuple[float, float]], pad_state: Dict) -> List[Tuple[float, float]]:
    """
    根据PAD状态调整轨迹

    调整规则:
    - 高Arousal: 加速更快，轨迹更激进
    - 低Arousal: 加速更慢，轨迹更保守
    - 高Dominance: 更倾向于变道和超车
    - 低Dominance: 更倾向于保持车道
    - Pleasure影响整体平滑度
    """
    if not trajectory:
        return trajectory

    traj = np.array(trajectory)

    # Arousal影响速度（纵向距离）
    speed_factor = 1.0 + pad_state['arousal'] * 0.25  # 范围: 0.75 - 1.25

    # Dominance影响横向偏移
    lateral_offset = pad_state['dominance'] * 0.15

    # Pleasure影响平滑度（负面情绪导致更多抖动）
    if pad_state['pleasure'] < 0:
        noise_level = abs(pad_state['pleasure']) * 0.05
        noise = np.random.normal(0, noise_level, traj.shape)
        traj += noise

    # 应用速度调整（纵向）
    adjusted_traj = [traj[0]]  # 保持起点不变
    for i in range(1, len(traj)):
        direction = traj[i] - traj[i-1]
        new_point = adjusted_traj[-1] + direction * speed_factor
        adjusted_traj.append(new_point)

    adjusted_traj = np.array(adjusted_traj)

    # 应用横向调整
    adjusted_traj[:, 1] += lateral_offset

    return [(round(x, 2), round(y, 2)) for x, y in adjusted_traj]

def analyze_pad_impact(pad_state: Dict) -> Dict[str, str]:
    """分析PAD各维度的影响"""
    p, a, d = pad_state['pleasure'], pad_state['arousal'], pad_state['dominance']

    # Pleasure影响
    if p > 0.3:
        pleasure_impact = "Positive emotional state leads to smooth and confident driving behavior."
    elif p < -0.3:
        pleasure_impact = "Negative emotional state may cause hesitation and less smooth control."
    else:
        pleasure_impact = "Neutral emotional state maintains standard driving behavior."

    # Arousal影响
    if a > 0.5:
        arousal_impact = "High arousal level results in faster acceleration and more dynamic maneuvers."
    elif a < -0.3:
        arousal_impact = "Low arousal level leads to slower, more conservative driving."
    else:
        arousal_impact = "Moderate arousal level maintains balanced driving speed."

    # Dominance影响
    if d > 0.5:
        dominance_impact = "High dominance increases willingness to change lanes and assert position."
    elif d < -0.3:
        dominance_impact = "Low dominance leads to more passive driving and lane keeping."
    else:
        dominance_impact = "Moderate dominance maintains standard lane behavior."

    return {
        "pleasure": pleasure_impact,
        "arousal": arousal_impact,
        "dominance": dominance_impact
    }

def generate_pad_decision(pad_state: Dict, original_decision: str) -> str:
    """生成基于PAD状态的决策说明"""
    label = classify_pad_emotion(pad_state)

    decision_templates = {
        "calm": "maintain steady speed and follow the lane center trajectory",
        "aggressive": "accelerate more assertively and be ready for lane changes if needed",
        "cautious": "decelerate slightly and maintain larger safety margins",
        "anxious": "reduce speed and avoid any risky maneuvers",
        "confident": "drive assertively while maintaining safety standards",
        "tired": "maintain conservative speed and avoid complex maneuvers"
    }

    base_decision = decision_templates.get(label, "maintain current driving behavior")

    return f"Based on the {label} emotional state (P={pad_state['pleasure']:.2f}, A={pad_state['arousal']:.2f}, D={pad_state['dominance']:.2f}), the car should {base_decision}."

def generate_pad_thought_process(pad_state: Dict, scenario_info: str) -> str:
    """生成PAD影响的思考过程"""
    label = classify_pad_emotion(pad_state)
    impacts = analyze_pad_impact(pad_state)

    thought = f"""Given the current {label} emotional state:

1. Emotional State Analysis:
   - Pleasure ({pad_state['pleasure']:.2f}): {impacts['pleasure']}
   - Arousal ({pad_state['arousal']:.2f}): {impacts['arousal']}
   - Dominance ({pad_state['dominance']:.2f}): {impacts['dominance']}

2. Driving Behavior Adjustment:
   The emotional state influences the trajectory by adjusting acceleration patterns, lane positioning, and overall driving style while still respecting traffic rules and safety constraints.

3. Safety Considerations:
   Despite the emotional influence, all adjustments maintain compliance with traffic rules and prioritize collision avoidance."""

    return thought

def build_pad_input(original_input: str, pad_state: Dict) -> str:
    """构建包含PAD情感状态的输入提示"""
    label = classify_pad_emotion(pad_state)
    emotion_desc = get_emotion_description(label)

    # 在原始输入的Role部分后添加情感状态信息
    pad_info = f"""
Emotional State (PAD Model):
- Pleasure: {pad_state['pleasure']:.2f} (range: -1 to 1)
- Arousal: {pad_state['arousal']:.2f} (range: -1 to 1)
- Dominance: {pad_state['dominance']:.2f} (range: -1 to 1)
- Emotion Label: {label}
- Description: {emotion_desc}

"""

    # 在"Role:"之后插入PAD信息
    if "Role:" in original_input:
        parts = original_input.split("Context:", 1)
        if len(parts) == 2:
            new_input = parts[0] + pad_info + "Context:" + parts[1]
        else:
            new_input = original_input.replace("Role:", "Role:" + pad_info, 1)
    else:
        new_input = pad_info + original_input

    # 修改输出要求，强调情感影响
    new_input += "\n\nIMPORTANT: Your trajectory and decision should reflect your current emotional state as described by the PAD values above."

    return new_input

def build_pad_target(original_target: str, pad_state: Dict, original_input: str) -> str:
    """构建基于PAD状态调整的目标输出"""
    # 解析原始轨迹
    original_traj = parse_trajectory_from_text(original_target)

    if not original_traj:
        # 如果无法解析轨迹，返回原始target加上PAD分析
        impacts = analyze_pad_impact(pad_state)
        pad_analysis = f"\n\nPAD Influence Analysis:\n"
        pad_analysis += f"- Pleasure ({pad_state['pleasure']:.2f}): {impacts['pleasure']}\n"
        pad_analysis += f"- Arousal ({pad_state['arousal']:.2f}): {impacts['arousal']}\n"
        pad_analysis += f"- Dominance ({pad_state['dominance']:.2f}): {impacts['dominance']}"
        return original_target + pad_analysis

    # 根据PAD调整轨迹
    adjusted_traj = adjust_trajectory_by_pad(original_traj, pad_state)

    # 生成新的决策和思考过程
    decision = generate_pad_decision(pad_state, original_target)
    thought_process = generate_pad_thought_process(pad_state, original_input)

    # 构建新的target
    traj_str = str(adjusted_traj).replace("'", "")

    new_target = f"""- Refined Ego Future Trajectories in next 8.0s: {traj_str}

Decision:
{decision}

Thought Process:
{thought_process}"""

    return new_target

def generate_pad_data(input_file: str, output_file: str, num_pad_variations: int = 4):
    """生成PAD训练数据"""
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    print(f"Original data size: {len(original_data)}")

    pad_data = []

    for idx, item in enumerate(original_data):
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(original_data)}...")

        # 为每个原始样本生成多个PAD变体
        num_variations = min(num_pad_variations, len(PREDEFINED_PAD_STATES))
        selected_pads = random.sample(PREDEFINED_PAD_STATES, num_variations)

        for pad_state in selected_pads:
            # 构建新的输入和目标
            new_input = build_pad_input(item['input'], pad_state)
            new_target = build_pad_target(item['target'], pad_state, item['input'])

            # 创建新样本
            new_item = {
                "input": new_input,
                "target": new_target,
                "map_info": item.get('map_info', None),
                "pad_state": {
                    "pleasure": pad_state['pleasure'],
                    "arousal": pad_state['arousal'],
                    "dominance": pad_state['dominance'],
                    "label": pad_state['label']
                }
            }

            pad_data.append(new_item)

    print(f"\nGenerated {len(pad_data)} PAD training samples")
    print(f"Saving to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pad_data, f, indent=2, ensure_ascii=False)

    print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Generate PAD emotion training data')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file (e.g., mixed_decision_driveqa_train_epoch3.json)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file for PAD training data')
    parser.add_argument('--num_pad_variations', type=int, default=4,
                        help='Number of PAD variations per sample (default: 4)')

    args = parser.parse_args()

    generate_pad_data(args.input_file, args.output_file, args.num_pad_variations)

if __name__ == '__main__':
    main()

