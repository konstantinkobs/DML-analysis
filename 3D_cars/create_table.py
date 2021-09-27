import sys
sys.path.append("..")

from table import Table

from reality_check.utils import all_folds, all_attributes, loss_metadata, convert_to_valid_path
import pickle
import pyperclip

from collections import defaultdict

all_losses = [l for l in loss_metadata if loss_metadata[l]["type"] == "Ranking"] + [l for l in loss_metadata if loss_metadata[l]["type"] == "Classification"] + ["None"]

table = Table(width=len(all_attributes)+2, height=len(all_losses)+2)

table[2, 0] = "\multirow{9}{*}{\\rotatebox{90}{\\textbf{Ranking}}}"
table[11, 0] = "\multirow{5}{*}{\\rotatebox{90}{\\textbf{Classif.}}}"

table.hlines = [2,11,16]

for row, loss in enumerate(all_losses):
    accuracies = defaultdict(float)
    
    for fold in all_folds:
        with open(f"normalized r-precisions/{convert_to_valid_path(loss)}_{fold}.p", "rb") as f:
            acc = pickle.load(f)

        for a in all_attributes:
            accuracies[a] += acc[a]['normalized_r_precision']
    
    table[row+2, 1] = f"\\textbf{{{loss}}}"
    
    for col, a in enumerate(all_attributes):
        accuracies[a] /= len(all_folds)
        value = f"\\underline{{{accuracies[a]:.2f}}}" if accuracies[a] >= 2.576 else f"{accuracies[a]:.2f}"
        table[row+2,col+2] = f"\\cellcolor{{gray!{int(accuracies[a]/60*80)}}} {value}"

pyperclip.copy(str(table))
