# Gradient Explainer

In our paper, we propose a simple yet effective technique to identify the areas in an input image that are used by a DML model to embed it.

`explainer.py` contains a class that can be used with any PyTorch image embedding model. With it, you can compute saliency maps for all images in a given dataset for a given model. If you want to reproduce the paper, you need to run this for all available datasets, models, and folds.

`compare_gradients.py` then compares all generated saliency maps for a given loss pair using the correlation coefficient and Jensen-Shannon divergence.

The created statistics are then used in `create_stats_table.py` in order to generate Table 2 in the paper as well as Table 1 and 2 in the Appendix.
