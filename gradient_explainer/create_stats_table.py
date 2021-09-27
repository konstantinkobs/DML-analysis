import sys
from typing import Callable
sys.path.append("..")
from os.path import isfile
import pandas as pd
import numpy as np
from utils import all_datasets, all_losses, convert_to_valid_path, loss_metadata


def fisher_z_aggr(arr: np.array, aggr_fun: Callable = np.mean) -> float:
    # According to https://link.springer.com/content/pdf/10.1007%2F978-3-642-12770-0.pdf (p. 160)
    zs = 0.5 * np.log((1 + arr) / (1 - arr))
    z_aggr = aggr_fun(zs)
    aggr = (np.exp(2 * z_aggr) - 1) / (np.exp(2 * z_aggr) + 1)
    return aggr


# dataset = "CUB200"
# dataset = "Cars196"
dataset = "SOP"

cols = len(all_losses)
rows = [["" for _ in range(cols)] for _ in range(cols)]


for i, loss1 in zip(range(len(all_losses)), all_losses):
    for j, loss2 in zip(range(len(all_losses)), all_losses):

        if loss1 == loss2:
            rows[i][j] = "100±0"
            continue

        file_path = f"stats/{dataset}/0/{convert_to_valid_path(loss1)}_{convert_to_valid_path(loss2)}.csv"
        if not isfile(file_path):
            file_path = f"stats/{dataset}/0/{convert_to_valid_path(loss2)}_{convert_to_valid_path(loss1)}.csv"
        if not isfile(file_path):
            continue
        stats = pd.read_csv(file_path)

        rows[i][j] = f"{int(fisher_z_aggr(stats['correlation'], np.mean)*100):d}±{int(fisher_z_aggr(stats['correlation'], np.std)*100):d}"

column_names = []
row_names = []
for i, loss in zip(range(0, len(all_losses)), all_losses):
    loss_str = loss_metadata[loss]["abbreviation"] if "abbreviation" in loss_metadata[loss] else loss
    col_loss_type = loss_metadata[loss]["type"]
    if col_loss_type == 'None':
        col_loss_type = ''
    column_names.append((f'\\textbf{{{col_loss_type}}}', f"\\textbf{{{loss_str}}}"))
    row_loss_type = loss_metadata[loss]["type"]
    if row_loss_type == 'None':
        row_loss_type = ''
    if row_loss_type == 'Classification':
        row_loss_type = 'Classif.'
    row_names.append((f'\\rotatebox{{90}}{{\\textbf{{{row_loss_type}}}}}', f"\\textbf{{{loss_str}}}"))
col_ind = pd.MultiIndex.from_tuples(column_names)
row_ind = pd.MultiIndex.from_tuples(row_names)
data = pd.DataFrame(rows, columns=col_ind, index=row_ind)

print(data.to_latex(index=True, multirow=True, multicolumn=True, multicolumn_format='c|', column_format='@{}c@{\hspace{1.5mm}}r|ccccccccc|ccccc|c@{}', escape=False).replace('cline{1-17}', 'hline').replace('toprule', '').replace('midrule', 'hline').replace('bottomrule', ''))
