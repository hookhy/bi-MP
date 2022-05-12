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

| Model name         | TEST MAE (mean,std) | Model Params (#Layers) |
| ------------------ | ------------------- | ---------------------- |
| bi-GatedGCN        |     0.166 (0.009)   |      511974 (16)       |


## Reference

  @misc{kwon2022boosting, <br>
       title={Boosting Graph Neural Networks by Injecting Pooling in Message Passing}, <br>
       author={Hyeokjin Kwon and Jong-Min Lee}, <br>
       year={2022}, <br>
       eprint={2202.04768}, <br>
       archivePrefix={arXiv}, <br>
       primaryClass={cs.LG} <br>
       }


