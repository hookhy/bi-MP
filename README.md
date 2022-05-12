# Bilateral Message Passing

This repository is the official implementation of [Bilateral Message Passing](https://arxiv.org/abs/2202.04768). 

+ Analogous to *the bilateral image filter*, we propose a bi-MP scheme to address over-smoothing in classic MP GNNs.

+ Our proposed scheme **can be generalized to all ordinary MP GNNs** (e.g. SOTA MP-GNNs such as GCN, GraphSAGE, and GAT).

![Figure1_upload](https://user-images.githubusercontent.com/84267304/152954507-846c98ec-3858-4143-b448-e10b072e7a9f.jpg)

Various categories contains scripts from [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns).

>📋   [Follows the instructions](https://github.com/hookhy/bi-MP/blob/HyeokjinK/add_bimp.md) to add a custom bilateral-MP layer.

## Requirements

To install requirements:

```setup
# Install python environment
conda env create -f environment_gpu.yml 

# Activate environment
conda activate benchmark_gnn
```

## Training & evaluation

To train & evaluate the model(s) in the paper with specified dataset, and model, run this command:

```train
python main_{DatasetDependentName}.py --config Configpath/Configfname.json --gpu_id 0 --model ModelName
```

## Results

Our model achieves the following performance on :

### ZINC Dataset

| Model name         | TEST MAE (mean,std) |    #Params (#Layers)   |
| ------------------ | ------------------- | ---------------------- |
| bi-GatedGCN (ours) |     0.166 (0.009)   |      511974 (16)       |
| GatedGCN           |     0.214 (0.013)   |      505011 (16)       |
| bi-GraphSage (ours)|     0.245 (0.009)   |      516651 (16)       |
| bi-GCN (ours)      |     0.276 (0.007)   |      536482 (16)       |
| bi-GAT (ours)      |     0.277 (0.012)   |      535536 (16)       |
| GCN                |     0.367 (0.011)   |      505079 (16)       |
| GAT                |     0.384 (0.007)   |      531345 (16)       |
| GraphSage          |     0.398 (0.002)   |      505341 (16)       |

### TSP Dataset

| Model name         | TEST F1  (mean,std) |    #Params (#Layers)   |
| ------------------ | ------------------- | ---------------------- |
| bi-GatedGCN (ours) |     0.812 (0.004)   |      125832 (4)        |
| GatedGCN           |     0.808 (0.003)   |      97858 (4)         |
| bi-GraphSage (ours)|     0.693 (0.016)   |      131861 (4)        |
| bi-GAT (ours)      |     0.675 (0.002)   |      115609 (4)        |
| GAT                |     0.673 (0.002)   |      96182 (4)         |
| GraphSage          |     0.665 (0.003)   |      99263 (4)         |
| bi-GCN (ours)      |     0.642 (0.001)   |      118496 (4)        |
| GCN                |     0.630 (0.001)   |      95702 (4)         |

### MNIST Dataset

| Model name         | TEST ACC  (mean,std)|    #Params (#Layers)   |
| ------------------ | ------------------- | ---------------------- |
| bi-GatedGCN (ours) |    97.575 (0.085)   |      101365 (4)        |
| bi-GraphSage (ours)|    97.438 (0.155)   |      110400 (4)        |
| GatedGCN           |    97.340 (0.143)   |      125815 (4)        |
| GraphSage          |    97.312 (0.097)   |      114169 (4)        |
| GAT                |    95.535 (0.205)   |      114507 (4)        |
| bi-GAT (ours)      |    95.363 (0.199)   |      104337 (4)        |
| bi-GCN (ours)      |    90.805 (0.299)   |      104217 (4)        |
| GCN                |    90.705 (0.218)   |      110807 (4)        |

### CIFAR10 Dataset

| Model name         | TEST ACC  (mean,std)|    #Params (#Layers)   |
| ------------------ | ------------------- | ---------------------- |
| bi-GatedGCN (ours) |    67.850 (0.522)   |      110632 (4)        |
| GatedGCN           |    67.312 (0.311)   |      104307 (4)        |
| GraphSage          |    65.767 (0.308)   |      104517 (4)        |
| bi-GraphSage (ours)|    64.863 (0.445)   |      114312 (4)        |
| bi-GAT (ours)      |    64.275 (0.458)   |      114311 (4)        |
| GAT                |    64.223 (0.455)   |      110704 (4)        |
| GCN                |    55.710 (0.381)   |      101657 (4)        |
| bi-GCN (ours)      |    54.450 (0.137)   |      125564 (4)        |

### CLUSTER Dataset

| Model name         | TEST ACC  (mean,std)|    #Params (#Layers)   |
| ------------------ | ------------------- | ---------------------- |
| bi-GatedGCN (ours) |    76.896 (0.213)   |      516211 (16)       |
| GatedGCN           |    76.082 (0.196)   |      504253 (16)       |
| bi-GCN (ours)      |    71.199 (0.882)   |      505149 (16)       |
| bi-GAT (ours)      |    71.113 (0.869)   |      445438 (16)       |
| GAT                |    70.587 (0.447)   |      527824 (16)       |
| GCN                |    68.498 (0.976)   |      501687 (16)       |
| bi-GraphSage (ours)|    64.088 (0.182)   |      490569 (16)       |
| GraphSage          |    63.844 (0.110)   |      503350 (16)       |

## Reference

  @misc{kwon2022boosting, <br>
       title={Boosting Graph Neural Networks by Injecting Pooling in Message Passing}, <br>
       author={Hyeokjin Kwon and Jong-Min Lee}, <br>
       year={2022}, <br>
       eprint={2202.04768}, <br>
       archivePrefix={arXiv}, <br>
       primaryClass={cs.LG} <br>
       }


