# Classifying Nodes in a Citation Network with GNNs

Semi-supervised node classification on citation networks (Cora, Citeseer, PubMed).
We compare two baselines that ignore the graph (**Logistic Regression**, **MLP**)
with three Graph Neural Networks (**GCN**, **GraphSAGE**, **GAT**), and study how
the number of layers (oversmoothing), dropout, and embedding size affect accuracy.