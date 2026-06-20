# Classifying Nodes in a Citation Network with GNNs

Semi-supervised node classification on citation networks (Cora, Citeseer, PubMed).
We compare two baselines that ignore the graph (**Logistic Regression**, **MLP**)
with three Graph Neural Networks (**GCN**, **GraphSAGE**, **GAT**), and study how
the number of layers (oversmoothing), dropout, and embedding size affect accuracy.

```
citation-gnn/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_loader.py       
│   ├── graph_analysis.py           
│   ├── mlp_baseline.py            
│   ├── logreg_baseline.py        
│   ├── gcn.py                   
│   ├── graph_sage.py           
│   ├── gat.py                          
│   ├── train.py                       
│   ├── run_comparison.py               
│   ├── run_layer_study.py              
│   ├── run_dropout_embedding_study.py  
│   └── visualize.py 
├── results/
│   ├── tsne.png                   
├── notebooks/
│   └── exploration.ipynb
└── report/
    └── report.md          
```

## Setup: use virtual environment

```bash
python -m venv venv
source venv/bin/activate        
# Windows: venv\Scripts\activate

pip install -r requirements.txt
```
## How it works?

Project downloads the datasets automatically
the first time into a `data/` folder.

```bash
# compare baselines and GNNS
python src/run_comparison.py --dataset Cora

# oversmoothing study
python src/run_layer_study.py --dataset Cora

# dropout and embedding-size study
python src/run_dropout_embedding_study.py --dataset Cora

# t-SNE plot
python src/visualize.py --dataset Cora
```

To use other datasets use `--dataset Cora` or `--dataset Citeseer` or `--dataset PubMed`
