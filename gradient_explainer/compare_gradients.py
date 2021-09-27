import sys
sys.path.append("..")
import os
from utils import all_losses, all_folds, all_datasets, convert_to_valid_path
import pickle
import numpy as np
from multiprocessing import Pool
import argparse
import pandas as pd
from scipy.spatial.distance import jensenshannon

def calculate_stats(grads):
    grad1, grad2 = grads
    grad1 = grad1.flatten()
    grad2 = grad2.flatten()

    corr = np.corrcoef(grad1, grad2)[0, 1]

    grad1 /= np.sum(grad1)
    grad2 /= np.sum(grad2)
    jsd = np.square(jensenshannon(grad1, grad2))

    return {
        "correlation": corr,
        "jensenshannondiv": jsd
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss1", type=str, help="First loss to use for the comparison")
    parser.add_argument("--loss2", type=str, help="Second loss to use for the comparison")
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--fold", type=int, help="Fold of dataset")
    args = parser.parse_args()

    print(f"Comparing {args.loss1} and {args.loss2} on {args.dataset} (fold {args.fold})")
    
    loss1_path = f"results/{convert_to_valid_path(args.loss1)}/{args.dataset}/{args.fold}.p"
    loss2_path = f"results/{convert_to_valid_path(args.loss2)}/{args.dataset}/{args.fold}.p"
    results_dir = f"stats/{args.dataset}/{args.fold}/"
    results_path = results_dir + f"{convert_to_valid_path(args.loss1)}_{convert_to_valid_path(args.loss2)}.csv"
    
    if os.path.exists(results_path):
        print(f"This combination was already checked.")
        sys.exit(0)

    with open(loss1_path, "rb") as f:
        grads1 = pickle.load(f)
        print(f"Loaded grads for {args.loss1}")

    if not os.path.exists(loss1_path) or not os.path.exists(loss2_path):
        print("Not all files are available.")
        sys.exit("Not all files are available.")
    
    with open(loss2_path, "rb") as f:
        grads2 = pickle.load(f)
        print(f"Loaded grads for {args.loss2}")

    print("Starting statistic generation")
    with Pool(processes=4) as pool:
        stats = pool.map(calculate_stats, zip(grads1, grads2))
    
    del grads1
    del grads2

    print("Creating DataFrame")
    stats = pd.DataFrame(stats)

    os.makedirs(results_dir, exist_ok=True)
    print("Saving DataFrame")
    stats.to_csv(results_path, index=False)

    print("Finished:")
    print(stats.describe())
