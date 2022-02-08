# Bilateral Message Passing

We provide the implementaion & detail of the bilateral message passing [bi-MP]() in PyTorch, DGL frameworks. 

Analogous to the bilateral image filter, we propose a bi-MP scheme to address over-smoothing in classic MP GNNs. Instead of directly propagating information through local edges, the proposed model defines a pairwise modular gradient between nodes and uses it to apply a gating mechanism to the MP layer’s aggregating function. More specifically, the bilateral-MP takes a soft assignment matrix of as input and extracts the modular gradient by applying metric learning layers to selectively transfer the messages. The key intuition is that the propagation of useful information within the same node class survives while the extraneous noise between different classes is reduced. Thus, the bilateral-MP layer results in better graph representation and improved performance by preventing over-smoothing.

[Figure1.pdf](https://github.com/hookhy/bi-MP/files/8022287/Figure1.pdf)

Various categories contains scripts from [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns).

# Reference




