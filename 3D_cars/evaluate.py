import sys
sys.path.append("..")
import os
from reality_check.utils import all_losses, all_folds, convert_to_valid_path
import pandas as pd
import pickle
import numpy as np
from itertools import product
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import pytorch_metric_learning.utils.common_functions as c_f

import stackprinter
stackprinter.set_excepthook()

class NormalizedRPrecisionCalculator(AccuracyCalculator):
    def maybe_get_avg_of_avgs(self, accuracy_per_sample, sample_labels, avg_of_avgs):
        if avg_of_avgs:
            unique_labels = np.unique(sample_labels, axis=0)
            mask = c_f.np_all_from_dim_to_end(sample_labels == unique_labels[:, None], 2)
            mask = np.transpose(mask)
            acc_sum_per_class = np.sum(accuracy_per_sample[:, None] * mask, axis=0)
            mask_sum_per_class = np.sum(mask, axis=0)
            average_per_class = acc_sum_per_class / mask_sum_per_class
            return np.mean(average_per_class)
        return np.mean(accuracy_per_sample)

    def get_relevance_mask(
        self,
        shape,
        gt_labels,
        embeddings_come_from_same_source,
        label_counts,
        label_comparison_fn,
    ):
        relevance_mask = np.zeros(shape=shape, dtype=np.int)

        for label, count in zip(*label_counts):
            matching_rows = np.where(c_f.np_all_from_dim_to_end(gt_labels == label, 1))[0]
            max_column = count
            if label_comparison_fn is np.equal and embeddings_come_from_same_source:
                max_column -= 1
            relevance_mask[matching_rows, :max_column] = 1
        return relevance_mask

    def try_getting_not_lone_labels(self, knn_labels, query_labels, not_lone_query_mask):
        if not any(not_lone_query_mask):
            return None, None
        return (
            knn_labels[not_lone_query_mask],
            query_labels[not_lone_query_mask],
        )

    def calculate_normalized_r_precision(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        label_counts,
        **kwargs
    ):
        knn_labels, query_labels = self.try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return self.normalized_r_precision(
            knn_labels,
            query_labels[:, None],
            embeddings_come_from_same_source,
            label_counts,
            self.avg_of_avgs,
            self.label_comparison_fn,
        )

    def normalized_r_precision(
        self,
        knn_labels,
        gt_labels,
        embeddings_come_from_same_source,
        label_counts,
        avg_of_avgs,
        label_comparison_fn,
    ):
        relevance_mask = self.get_relevance_mask(
            knn_labels.shape[:2],
            gt_labels,
            embeddings_come_from_same_source,
            label_counts,
            label_comparison_fn,
        )
        same_label = label_comparison_fn(gt_labels, knn_labels)
        X = np.sum(same_label * relevance_mask.astype(bool), axis=1)
        del same_label
        p = np.squeeze(label_counts[1][gt_labels]) / len(gt_labels)
        n = np.sum(relevance_mask, axis=1)
        del relevance_mask
        # z-normalization
        mean = n * p
        del n
        std = np.sqrt(mean * (1 - p))
        accuracy_per_sample = (X - mean) / std
        del X
        del mean
        del std
        return self.maybe_get_avg_of_avgs(accuracy_per_sample, gt_labels, avg_of_avgs)

    def requires_knn(self):
        return super().requires_knn() + ["normalized_r_precision"] 



df = pd.read_csv("samples.csv", index_col="frame")

for loss in all_losses:
    for fold in all_folds:
        path = f"normalized r-precisions/{convert_to_valid_path(loss)}_{fold}.p"
        if os.path.exists(path):
            print(f"Skipping {loss}, fold {fold}")
            continue

        with open(f"representations/{convert_to_valid_path(loss)}_{fold}.p", "rb") as f:
            embeddings = pickle.load(f)

        acc = NormalizedRPrecisionCalculator(include=("normalized_r_precision",))

        print(f"\nCalculating R-Precision for {loss}")
        accuracies = {}
        for attribute in df.columns:
            labels = np.array(df[attribute].astype('category').cat.codes)
            a = acc.get_accuracy(embeddings.numpy(), embeddings.numpy(), labels, labels, embeddings_come_from_same_source=True)
            accuracies[attribute] = a
            print(f"{attribute}: {a}")

        print(f"\nSaving Normalized R-Precision for {loss}")
        with open(path, "wb") as f:
            pickle.dump(accuracies, f)
