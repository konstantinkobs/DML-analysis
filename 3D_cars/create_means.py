import sys
sys.path.append("..")

import numpy as np

from reality_check.utils import all_folds, all_attributes, loss_metadata, convert_to_valid_path
import pickle
import pyperclip

from collections import defaultdict
from scipy.stats import mannwhitneyu

types = ["Classification", "Ranking"]
accuracies = defaultdict(lambda: defaultdict(list))

for t in types:
    losses = [l for l in loss_metadata if loss_metadata[l]["type"] == t]
    
    for loss in losses:
        for fold in all_folds:
            with open(f"normalized r-precisions/{convert_to_valid_path(loss)}_{fold}.p", "rb") as f:
                acc = pickle.load(f)

            for a in all_attributes:
                accuracies[t][a].append(acc[a]['normalized_r_precision'])

mean_acc = {
    t: [np.mean(accuracies[t][a]) for a in all_attributes]
    for t in types
}

for t in mean_acc.keys():
    row = ' & '.join([f'\\cellcolor{{gray!{int(a/60*80)}}} {a:.2f}' for a in mean_acc[t]])
    print(f"{t} Mean & {row}")

for a in all_attributes:
    test = mannwhitneyu(accuracies['Classification'][a],accuracies['Ranking'][a])
    print(f"{a}: {test.pvalue:.5f}")