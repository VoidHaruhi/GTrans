import subprocess
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="GCN")
parser.add_argument("--dataset", type=str, default="cora")
parser.add_argument("--gpu_id", type=str, default="0")

args = parser.parse_args()

for lr_feat in [5e-3, 1e-3, 1e-4, 1e-5, 1e-6]:
    for lr_adj in [0.5, 0.1, 0.01]:
        for epochs in [5, 10]:
            for ratio in [0.01,0.05,0.0005]:
                result_files = []
                for seed in np.arange(0,10):
                    result_file = f"results/{args.dataset}_{args.model}_{lr_feat}_{lr_adj}_{epochs}_{ratio}_{seed}.out"
                    result_files.append(result_file)
                    if not os.path.exists(result_file):
                        subprocess.run("python train_both_all.py --seed {} --gpu_id {} --model {} --tune=1 --dataset {} "
                                "--lr_feat {} --lr_adj {} --epochs {} --ratio {} --debug=1 --test_val 1"
                                .format(seed, args.gpu_id, args.model, args.dataset, lr_feat, lr_adj, epochs, ratio), shell=True)
                accs = []
                for result_file in result_files:
                    with open(result_file, "+r") as f:
                        accs.append(float(f.readline()))
                for result_file in result_files:
                    os.remove(result_file)
                with open(f"results/{args.dataset}_{args.model}_tuning.csv", "a+") as f:
                    print(f"{lr_feat},{lr_adj},{epochs},{ratio},{np.mean(accs)},{np.std(accs)}", file=f)

