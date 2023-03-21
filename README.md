# Learning Sparse Latent Graph Representations for Anomaly Detection in Multivariate Time Series (KDD '22)

This repository contains the official PyTorch implementation* of Fused Sparse Autoencoder and Graph Net (FuSAGNet), introduced in ["Learning Sparse Latent Graph Representations for Anomaly Detection in Multivariate Time Series" (KDD '22)](https://dl.acm.org/doi/abs/10.1145/3534678.3539117).

\* Partly based on the implementation of [GDN](https://github.com/d-ailin/GDN), introduced in ["Graph Neural Network-Based Anomaly Detection in Multivariate Time Series" (AAAI '21)](https://ojs.aaai.org/index.php/AAAI/article/view/16523).

## Repository Organization
    
    ├── data
    |   └── swat
    |       ├── list.txt
    |       ├── test.csv
    |       └── train.csv
    ├── datasets
    |   └── TimeDataset.py
    ├── models
    |   ├── FuSAGNet.py
    |   └── graph_layer.py
    ├── util
    |   ├── data.py
    |   ├── net_struct.py
    |   ├── preprocess.py
    |   └── time.py
    ├── .gitattributes
    ├── .gitignore
    ├── README.md
    ├── __init__.py
    ├── evaluate.py
    ├── main.py
    ├── test.py
    └── train.py

## Requirements

* Python >= 3.6
* CUDA == 10.2
* PyTorch == 1.5.0
* PyTorch Geometric == 1.5.0

## Datasets

This repository includes [SWaT](https://link.springer.com/chapter/10.1007/978-3-319-71368-7_8) as the default dataset (see the `data` directory). The [WADI](https://dl.acm.org/doi/abs/10.1145/3055366.3055375) dataest can be requested [here](https://itrust.sutd.edu.sg/itrust-labs_datasets/) and the [HAI](https://www.usenix.org/system/files/cset20-paper-shin.pdf) dataset can be downloaded [here](https://github.com/icsdataset/hai).

## Run

You can run the code using the following command.

```
python main.py
```

## Citation

If you are to cite our paper, please consider using the BibTeX citation below.

```
@inproceedings{han2022learning,
  title={Learning Sparse Latent Graph Representations for Anomaly Detection in Multivariate Time Series},
  author={Han, Siho and Woo, Simon S},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2977--2986},
  year={2022}
}
```