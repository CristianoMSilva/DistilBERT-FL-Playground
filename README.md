# DistilBERT-FL-Playground

A hands-on playground for learning how to detect phishing emails with DistilBERT and Federated Learning.

## About

This repository is my practice space for the experiments of my undergraduate thesis in Computer Science at UNESP (Brazil), written in Portuguese. The title translates to *Optimizing Banking Phishing Detection Using DistilBERT and Federated Learning*.

The goal is to go through the whole pipeline step by step, on a public dataset, before running the final experiments:

1. Understand and clean the data
2. Turn emails into tokens that DistilBERT can read
3. Train a centralized model (all data in one place)
4. Train a federated model (data split across simulated clients)
5. Compare both results

## Notebooks

| Notebook | Step | Status |
|---|---|---|
| [`01_eda.ipynb`](https://colab.research.google.com/github/CristianoMSilva/DistilBERT-FL-Playground/blob/main/01_eda.ipynb) | Exploratory data analysis and cleaning | Done |
| [`02_tokenization.ipynb`](https://colab.research.google.com/github/CristianoMSilva/DistilBERT-FL-Playground/blob/main/02_tokenization.ipynb) | Tokenization with the DistilBERT tokenizer and train/validation/test split | Done |
| [`03_baseline_training.ipynb`](https://colab.research.google.com/github/CristianoMSilva/DistilBERT-FL-Playground/blob/main/03_baseline_training.ipynb) | Centralized fine-tuning and metrics (256 vs 512 tokens) | Done |
| [`04_federated.ipynb`](https://colab.research.google.com/github/CristianoMSilva/DistilBERT-FL-Playground/blob/main/04_federated.ipynb) | Federated Learning simulation (FedAvg, IID and non-IID clients) | Done |

Each link opens the notebook in Google Colab.

## Dataset

[zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset), loaded from the Hugging Face Hub. Each row is one email labeled as `Safe Email` or `Phishing Email`.

Summary after cleaning (`01_eda.ipynb`):

| | Emails |
|---|---|
| Original dataset | 18,650 |
| Removed (empty, duplicated, conflicting labels, corrupted row) | 1,116 |
| **Final dataset** | **17,534** |

- Class balance: 62.6% safe, 37.4% phishing
- Median length: 184 words (safe) and 145 words (phishing)
- Split (`02_tokenization.ipynb`): 80% train (14,027), 10% validation (1,753), 10% test (1,754), stratified by class
- Emails longer than the token limit are cut: 48% at 256 tokens and 24% at 512 tokens

## Results: centralized baseline

`03_baseline_training.ipynb` fine-tunes `distilbert-base-uncased` twice, changing only the maximum number of tokens read per email (`max_length`).

Setup: 3 epochs, learning rate 2e-5, batch size 16, class-weighted loss (phishing is the minority class), best epoch chosen by validation F1, Colab T4 GPU.

Test set (1,754 emails, never seen during training; phishing is the positive class):

| max_length | Accuracy | Precision | Recall | F1 | False positive rate | Training time |
|---|---|---|---|---|---|---|
| 256 | 98.63% | 98.47% | 97.87% | 98.17% | 0.91% | 6.0 min |
| 512 | **99.14%** | **99.08%** | **98.63%** | **98.85%** | **0.55%** | 11.2 min |

- 512 tokens was better on every metric: missed phishing dropped from 14 to 9 emails, and false alarms from 10 to 6.
- The cost is about 2x the training time.
- These numbers come from a single run (seed 42), and the difference is a few emails, so they show a trend, not proof.

## Results: federated vs centralized

`04_federated.ipynb` splits the same 14,027 training emails across 3 simulated clients (think of 3 banks) and trains with **FedAvg**: in each round, every client trains the global model for 1 epoch on its own emails, and the server averages the clients' weights, weighted by how many emails each one has. The emails never leave the clients; only the weights are shared.

Two ways of splitting the data:

- **IID**: each client gets a random third of the emails, all with 37.4% phishing.
- **Non-IID**: clients get very different mixes, like banks that face different attacks: 78%, 37% and 9% phishing, with different sizes.

Setup: `max_length` 256, 5 rounds, 1 local epoch per round, class-weighted loss computed by each client from its own labels, best round chosen by validation F1.

Test set (same 1,754 emails as the centralized run):

| Training | Accuracy | Precision | Recall | F1 | False positive rate | Missed phishing | False alarms |
|---|---|---|---|---|---|---|---|
| Centralized (256) | 98.63% | 98.47% | 97.87% | 98.17% | 0.91% | 14 | 10 |
| Federated, IID | 98.69% | 97.59% | **98.93%** | 98.26% | 1.46% | **7** | 16 |
| Federated, non-IID | **98.80%** | **98.62%** | 98.17% | **98.40%** | **0.82%** | 12 | **9** |

- **Federated training matched the centralized model**: all three F1 scores are within 0.25 points. Keeping the data on each client cost practically nothing here.
- **The non-IID effect shows during training, not in the final result**: it started lower (round 1 F1: 97.09% vs 97.96% for IID), oscillated more, and was still improving at the last round, while IID stabilized at round 3.
- **Why the non-IID impact was small**: the skew is mild (every client has both classes), per-client class weights compensate the imbalance, and 1 local epoch per round keeps clients close to the global model.
- **Caveats**: single run (seed 42), so differences of a few emails show a trend, not proof. Training time is not comparable, because the clients trained one after another on one GPU instead of in parallel.
- **Next ideas**: a harsher non-IID split (a client with no phishing), training without class weights, more rounds and several seeds.

*About Flower: the plan was to use [Flower](https://flower.ai/), but its simulation engine (Ray) crashed on the Colab GPU with Flower 1.39. FedAvg was implemented by hand instead (about 20 lines), which also makes each step of the algorithm visible in the notebook.*

## How to run

No installation needed. Open a notebook in Google Colab (links in the table above) and click **Runtime > Run all**. Notebooks 03 and 04 need a GPU (**Runtime > Change runtime type > T4 GPU**).

## Tech stack

Python, pandas, scikit-learn, Hugging Face `datasets` and `transformers`, PyTorch and Google Colab. FedAvg implemented by hand for the Federated Learning step.

## Author

Cristiano Marques Silva
