"""
Simulator script for true asynchronous AsyncDriver
Based on simulator_qwen4drive.py with async support
"""

# IMPORTANT: Set multiprocessing start method BEFORE any other imports
# This is required for CUDA compatibility in multiprocessing
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import sys
import argparse
import os
from pathlib import Path
import tempfile
import time
import datetime
from nuplan.planning.script.run_simulation import main as main_simulation
import hydra
import warnings
warnings.filterwarnings("ignore", "invalid value encountered in.*", RuntimeWarning)

case_type = [
    'starting_left_turn',
    'starting_right_turn',
    'starting_straight_traffic_light_intersection_traversal',
    'stopping_with_lead',
    'high_lateral_acceleration',
    'high_magnitude_speed',
    'low_magnitude_speed',
    'traversing_pickup_dropoff',
    'waiting_for_pedestrian_to_cross',
    'behind_long_vehicle',
    'stationary_in_traffic',
    'near_multiple_vehicles',
    'changing_lane',
    'following_lane_with_lead'
]

def parse_args():
    parser = argparse.ArgumentParser(description='AsyncDriver True Async Inference')
    parser.add_argument('--openloop', '-o', action='store_true')
    parser.add_argument('--planner', '-p', type=str, default=None, help='Path to planner checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory name')
    parser.add_argument('--planner_type', type=str, default='qwen4drive_async', help='Planner type')
    parser.add_argument('--ref', type=str, default=None)
    parser.add_argument('--type', type=int, default=None, help='Scenario type (0-13)')

    # Async-specific parameters
    parser.add_argument('--async_mode', action='store_true', help='Enable true async mode')
    parser.add_argument('--llm_interval', type=float, default=1.5,
                       help='LLM update interval in seconds (default: 1.5s)')
    parser.add_argument('--llm_device', type=str, default='cuda:0',
                       help='Device for LLM (default: cuda:0)')
    parser.add_argument('--gameformer_device', type=str, default='cuda:0',
                       help='Device for GameFormer (default: cuda:0)')

    # Original parameters
    parser.add_argument('--llm_inf_step', type=int, default=1,
                       help='(Not used in async mode)')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--short_ins', type=int, default=-1)
    parser.add_argument('--disable_refpath', action='store_true')
    parser.add_argument('--ins_wo_stop', action='store_true')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--base_model', type=str, default=None, help='Path to base Qwen model')
    parser.add_argument('--simulation_root_path', type=str, default=None,
                       help='Root path for simulation results')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Print configuration
    print("=" * 60)
    print("AsyncDriver True Async Inference Configuration")
    print("=" * 60)
    print(f"Async Mode: {args.async_mode}")
    if args.async_mode:
        if args.llm_interval <= 0:
            print(f"LLM Mode: Event-Driven (immediate use after inference)")
        else:
            print(f"LLM Update Interval: {args.llm_interval}s (~{1/args.llm_interval:.2f} Hz)")
        print(f"LLM Device: {args.llm_device}")
        print(f"GameFormer Device: {args.gameformer_device}")
    else:
        print("Running in sync mode (fallback)")
    print(f"Scenario Type: {case_type[args.type] if args.type is not None else 'All'}")
    print(f"Planner Type: {args.planner_type}")
    print("=" * 60)

    # Setup paths
    CONFIG_PATH = '../../nuplan/planning/script/config/simulation'
    CONFIG_NAME = 'default_simulation'
    sim_path = args.simulation_root_path

    if args.type is not None:
        if not os.path.exists(Path(sim_path) / 'simulation' / args.save_dir):
            os.makedirs(Path(sim_path) / 'simulation' / args.save_dir, exist_ok=True)
        SAVE_DIR = Path(sim_path) / 'simulation' / args.save_dir / case_type[args.type]
    else:
        SAVE_DIR = Path(sim_path) / 'simulation' / args.save_dir

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    PLANNER = args.planner_type
    scenario_filter_type = 'test_scenarios_hard20'

    CHALLENGE = 'open_loop_boxes' if args.openloop else 'closed_loop_reactive_agents'

    # Build dataset parameters
    DATASET_PARAMS = [
        'scenario_builder=nuplan_challenge',
        f'scenario_filter={scenario_filter_type}',
        f'+planner.{PLANNER}.ins_mode={args.ref}',
        f'planner.{PLANNER}.ins_wo_stop={args.ins_wo_stop}',  # Remove + (exists in yaml)
        f'planner.{PLANNER}.disable_refpath={args.disable_refpath}',
        f'+planner.{PLANNER}.finetune_model_path={args.planner}',
        f'+planner.{PLANNER}.model_name_or_path={args.base_model}',
        f'planner.{PLANNER}.enable_pdm_scorer_in_multirefpath={args.refine}',  # Remove + (exists in yaml)
        f'planner.{PLANNER}.lora_r={args.lora_r}',  # Remove + (exists in yaml)
        f'planner.{PLANNER}.short_ins={args.short_ins}',  # Remove + (exists in yaml)
        f'scenario_filter.num_scenarios_per_type=20',
        "hydra.searchpath=[pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]",
    ]

    # Add async-specific parameters
    if args.async_mode:
        DATASET_PARAMS.extend([
            f'planner.{PLANNER}.async_mode={args.async_mode}',  # Remove + (exists in yaml)
            f'planner.{PLANNER}.llm_update_interval={args.llm_interval}',  # Remove + (exists in yaml)
            f'planner.{PLANNER}.llm_device={args.llm_device}',  # Remove + (exists in yaml)
            f'planner.{PLANNER}.gameformer_device={args.gameformer_device}',  # Remove + (exists in yaml)
        ])
    else:
        DATASET_PARAMS.append(f'planner.{PLANNER}.async_mode=False')  # Remove + (exists in yaml)

    # Add scenario type filter
    if args.type is not None:
        DATASET_PARAMS.append(f'scenario_filter.scenario_types=[{case_type[args.type]}]')

    # Name of the experiment
    if args.async_mode:
        if args.llm_interval <= 0:
            EXPERIMENT = 'qwen4drive_async_event_driven'
        else:
            EXPERIMENT = f'qwen4drive_async_experiment_{args.llm_interval}s'
    else:
        EXPERIMENT = 'qwen4drive_experiment'

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'experiment_name={EXPERIMENT}',
        f'group={SAVE_DIR}',
        f'planner={PLANNER}',
        f'+simulation={CHALLENGE}',
        'worker=sequential',
        *DATASET_PARAMS,
    ])

    print("\nStarting simulation...")
    print(f"Results will be saved to: {SAVE_DIR}")
    print("=" * 60)

    # Record start time
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    print(f"Simulation start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run simulation
    main_simulation(cfg)

    # Record end time and calculate duration
    end_time = time.time()
    end_datetime = datetime.datetime.now()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60
    duration_hours = duration_minutes / 60

    print("\n" + "=" * 60)
    print("Simulation completed!")
    print("=" * 60)
    if args.type is not None:
        print(f"Scenario Type: {case_type[args.type]}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes / {duration_hours:.2f} hours)")
    print("=" * 60)

    # Write timing info to a separate log file
    timing_log_path = SAVE_DIR / 'timing_summary.txt'
    with open(timing_log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Simulation Timing Summary\n")
        f.write("=" * 60 + "\n")
        if args.type is not None:
            f.write(f"Scenario Type: {case_type[args.type]}\n")
        f.write(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total duration: {duration_seconds:.2f} seconds\n")
        f.write(f"Total duration: {duration_minutes:.2f} minutes\n")
        f.write(f"Total duration: {duration_hours:.2f} hours\n")
        f.write("=" * 60 + "\n")
        if args.async_mode:
            if args.llm_interval <= 0:
                f.write(f"LLM Mode: Event-Driven (immediate use after inference)\n")
            else:
                f.write(f"LLM Update Interval: {args.llm_interval}s (~{1/args.llm_interval:.2f} Hz)\n")
            f.write(f"LLM Device: {args.llm_device}\n")
            f.write(f"GameFormer Device: {args.gameformer_device}\n")
        f.write("=" * 60 + "\n")

    print(f"\nTiming summary saved to: {timing_log_path}")
