#!/usr/bin/env python3

import argparse
from train import run_lora_experiment, run_lora_experiment_multiple_seeds
import numpy as np
import os

myrepr = lambda x: repr(round(x, 8)).replace('.',',') if isinstance(x, float) else repr(x)

def str2bool(v):
    return v.lower() in ("true", "1", "yes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_frequency", type=int, default=1)
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--init_train_zero", type=str2bool, default=False)
    parser.add_argument("--gaussian_resample", type=str2bool, default=False)
    parser.add_argument("--deterministic_init", type=str2bool, default=False)
    parser.add_argument("--train_A", type=str2bool, default=True)
    parser.add_argument("--train_B", type=str2bool, default=True)
    parser.add_argument("--init_method_A", type=str, default="zero")
    parser.add_argument("--init_method_B", type=str, default="gaussian")
    parser.add_argument("--use_stochastic", type=str2bool, default=True)

    args = parser.parse_args()

    # Create a job name
    job_name_parts = [
        f"mf-{args.merge_frequency}",
        f"p-{myrepr(args.prob)}",
        f"itZ-{int(args.init_train_zero)}",
        f"gR-{int(args.gaussian_resample)}",
        f"det-{int(args.deterministic_init)}",
        f"stoch-{int(args.use_stochastic)}",
        f"trainA-{int(args.train_A)}",
        f"trainB-{int(args.train_B)}",
        f"initA-{args.init_method_A}",
        f"initB-{args.init_method_B}"
    ]
    job_name = "_".join(job_name_parts)

    NUM_LAUNCHES = 20
    seeds_to_try = np.arange(1, NUM_LAUNCHES + 1)
    
    acc, median_acc, mean_acc, std_acc = run_lora_experiment_multiple_seeds(
        seeds=seeds_to_try,
        rank=1,                               
        train_A=args.train_A,
        train_B=args.train_B,
        init_method_A=args.init_method_A,
        init_method_B=args.init_method_B,
        merge_frequency=args.merge_frequency,
        use_stochastic=args.use_stochastic,                  
        prob=args.prob,
        deterministic_init=args.deterministic_init,
        init_train_zero=args.init_train_zero,
        gaussian_resample=args.gaussian_resample
    )

    output_dir = "slurm_outs"
    os.makedirs(output_dir, exist_ok=True)

    np.save(f"{output_dir}/acc_{job_name}.npy", acc)
    np.save(f"{output_dir}/std-acc_{job_name}.npy", std_acc)
    np.save(f"{output_dir}/median-acc_{job_name}.npy", median_acc)
    np.save(f"{output_dir}/mean-acc_{job_name}.npy", mean_acc)
    
