# 3D Cars Dataset

Here, you find the code necessary to generate the synthetic car dataset as well as computing the Normalized R-Precision (NR-Prec) as proposed in the paper.

## Dataset

If you want to just use the dataset we already created, you can [download it here](https://oc.informatik.uni-wuerzburg.de/s/XY5mHYX5cpytPgE) (password: `iccv2021`).
The file `3D cars dataset.zip` contains all generated images as well as a CSV file with all parameters we chose uniformly at random for each frame.

## Generating the dataset by yourself

You can [download the Blender project file here](https://oc.informatik.uni-wuerzburg.de/s/BzH3HE9mrcYayDL) (password: `iccv2021`).
The archive contains `cars.blend`, the Blender project file containing everything to render the dataset.
Inside, we provide a Python script making use of the Blender Python API to generate new frames.
The generation is seeded and thus reproducible.
We used Blender v2.91 for rendering the project.

While it is possible to render the Blender project using the GUI, `render_and_optimize.sh` provides the commands to use the command line for that.
Since Blender's outputs are usually quite large due to unnecessary metadata, we used ImageOptim to optimize the file sizes without any quality loss.

## Compare different DML models on the dataset

To compare different DML models on the dataset, you can use `generate_representations.py` to feed all images through each model and save the resulting embeddings to the disk.

Then, `evaluate.py` reads these files and computes the Normalized R-Precisions for all models. The resulting NR-Precs are saved to disk.

`create_table.py` and `create_means.py` are used to generate Table 6 in the paper.

## Used models

All car models are taken from [CGTrader](https://www.cgtrader.com/free-3d-models/car?file_types%5B%5D=21)

Specific car models:
- [Ferrari Enzo](https://www.cgtrader.com/free-3d-models/car/sport/enzo-ferrari)
- [Mercedes Benz 300sel](https://www.cgtrader.com/free-3d-models/vehicle/other/mercedes-benz-300sel-63-1972-3d-model)
- [Renault Megane RS](https://www.cgtrader.com/free-3d-models/car/standard/renault-megane-rs-wide-body-by-kaiser-design)
- [Mercedes Benz AMG Coupe](https://www.cgtrader.com/free-3d-models/car/sport/mercedes-benz-s-class-2019-s63-amg-coupe)
- [Range Rover Evoque](https://www.cgtrader.com/free-3d-models/car/standard/range-rover-evoque-4fc9d2ae-c404-4c7c-9672-1ca5fb52a6d7)
- [Tesla Model S](https://www.cgtrader.com/free-3d-models/car/luxury/tesla-model-s-all-colors-high-quality)
