# Do Different Deep Metric Learning Losses Lead to Similar Learned Features?

This repository contains the code and data for the ICCV 2021 paper "Do Different Deep Metric Learning Losses Lead to Similar Learned Features?" by Konstantin Kobs, Michael Steininger, Andrzej Dulny, and Andreas Hotho:

> Recent studies have shown that many deep metric learning loss functions perform very similarly under the same experimental conditions. One potential reason for this unexpected result is that all losses let the network focus on similar image regions or properties. In this paper, we in- vestigate this by conducting a two-step analysis to extract and compare the learned visual features of the same model architecture trained with different loss functions: First, we compare the learned features on the pixel level by correlating saliency maps of the same input images. Second, we compare the clustering of embeddings for several image properties, e.g. object color or illumination. To provide independent control over these properties, photo-realistic 3D car renders similar to images in the Cars196 dataset are generated. In our analysis, we compare 14 pretrained models from a recent study and find that, even though all models perform similarly, different loss functions can guide the model to learn different features. We especially find differences between classification and ranking based losses. Our analysis also shows that some seemingly irrelevant properties can have significant influence on the resulting embedding. We encourage researchers from the deep metric learning community to use our methods to get insights into the features learned by their proposed methods.

## Getting started

In order to use our code, install all requirements given in the `requirements.txt` file.

You also need the pre-trained models by [Musgrave et al.](https://github.com/KevinMusgrave/powerful-benchmarker) You can find them [here](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/edit#gid=0) (we only use the models trained with batch size 32). Download all folders ending with "_reproduction0" and put them into `reality_check/models`. The paths given in `utils.py` should then match the correct paths to the model checkpoints.

In our paper, we perform a two-step analysis: We create *pixel-level saliency maps* and compare them between models and we generate a large synthetic car dataset and analyze the influence of different *high-level image properties* on the embeddings. Go from here to any of these two analysis steps:

- [Pixel-level saliency maps](gradient_explainer)
- [High-level image properties](3D_cars)


## Citation

If you are using code or data from this repo or find our work helpful for your research, please cite our paper:

```
@InProceedings{Kobs_2021_ICCV,
    author    = {Kobs, Konstantin and Steininger, Michael and Dulny, Andrzej and Hotho, Andreas},
    title     = {Do Different Deep Metric Learning Losses Lead to Similar Learned Features?},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10644-10654}
}
```
