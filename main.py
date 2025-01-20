
import argparse
import json
import os
from experiments.exp1 import *
from experiments.exp2 import *
from experiments.exp3 import *

def main():
    # Argument parser for selecting experiments
    parser = argparse.ArgumentParser(description="Run experiments for IPFLSTM Project")
    parser.add_argument(
        "--experiment",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Select the experiment to run: 1 (Model Comparison), 2 (LSTM Ablation), 3 (Time Window Study)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./experiment_config.json",
        help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found!")
        return

    with open(args.config, "r") as f:
        config = json.load(f)

    # Run the selected experiment
    if args.experiment == 1:
        print("Running Experiment 1: Model Comparison")
        run_experiment1(config)
    elif args.experiment == 2:
        print("Running Experiment 2: LSTM Ablation Study")
        run_experiment2(config)
    elif args.experiment == 3:
        print("Running Experiment 3: Time Window Study")
        run_experiment3(config)
    else:
        print("Invalid experiment choice!")

if __name__ == "__main__":
    main()
