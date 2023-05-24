# Predict Anderson impurity model with pretrain Transformer

## Getting Started

### Prerequisites

Install SCALIN package in the main directory, no further kernal needed to be installed.

```bash
bash install.sh 
```

### Download example database and pretrained Transformer

We uplodaed the database to two chaneels, figsahre `https://figshare.com/s/814f1ede101553876d43` and Dropbox. While, here we show case that to download the pretrained Transformer and database

```bash
wget https://www.dropbox.com/s/9tho8mqkyir8ak2/pretrained_transformer.tar && tar -xvf pretrained_transformer.tar

wget https://www.dropbox.com/s/82raq4eksbadirb/example_db.tar && tar -xvf example_db.tar.tar 

```

## Prediction with SCALIN_run.sh 

```bash
bash SCALING_run.sh
```

#TODO correct bugs in this pages. 