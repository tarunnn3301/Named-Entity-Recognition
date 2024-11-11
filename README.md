<h1 align="center">NER For Social Media Code Mixed Text</h1>

This repository contains code for utilizing a Named Entity Recognition (NER) model for code-mixed text.

A Named Entity Recognition (NER) model is a machine learning model used to identify and classify entities (such as names, locations, and organizations) within a given text. This is particularly useful for processing social media text where languages are often mixed, as in Hindi-English. NER is essential for various applications like information extraction, sentiment analysis, and social media monitoring.

In this repository, we are utilizing several NER approaches tailored for code-mixed data, including BiLSTM-CRF, CRF, Decision Tree, LSTM, BERT, and IndicBERT models. The dataset used in this repository comprises Twitter data, which contains code-mixed text for training and evaluation.

You can find the paper link here.[here](https://aclanthology.org/W18-2405.pdf).

<h2 align="left" id="setup">Setup ⚙️</h2>

To set up the environment, follow these steps:

1. Install Anaconda or Miniconda if you haven't already.

```
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh
bash ~/anaconda.sh
rm -rf ~/anaconda.sh
```

2. Create a new conda environment

```
conda create --name ner python=3.11.7
```

3. Activate the created environment:

```
conda activate ner
```

4. After setting up the environment, install the required libraries using the provided requirements.txt file:

```
pip install -r requirements.txt
```

5. Upgrade the datasets library to the latest version available, ensuring you have the most recent features and bug fixes:

```
pip install --upgrade datasets
```

6. Upgrade the transformers library to the latest version, ensuring access to the newest models, features, and improvements:

```
pip install transformers --upgrade
```

7. Use this command to execute a Python script by specifying the script's filename:

```
python <filename>.py
```
