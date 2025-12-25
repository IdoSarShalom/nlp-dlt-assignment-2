# Emotion Analysis with Deep Learning 🎭

Dana Gibor(322274234), Ido Sar Shalom(212410146), Natalya Sigal(306688466)

Assignment 2 as part of Bar-Ilan's 83374 "NLP using DL techniques" 🌠.

Implemented in PyTorch/Transformers 🔥.

## Description 📝

In this project we implement deep learning models for classifying text into **6 emotion categories**: Sadness, Joy, Love, Anger, Fear, and Surprise. We fine-tune and compare three pre-trained transformer models - **[BERT](https://arxiv.org/abs/1810.04805)** (Devlin et al., 2018), **[RoBERTa](https://arxiv.org/abs/1907.11692)** (Liu et al., 2019), and **[ELECTRA](https://arxiv.org/abs/2003.10555)** (Clark et al., 2020) - achieving **>80% accuracy** on social media text.

## The Repository 🧭

We provide here a short explanation about the structure of this repository:

* `data/train.csv` and `data/validation.csv` contain the raw datasets from the Emotion dataset.
* `data/bert`, `data/roberta`, and `data/electra` contain the trained models and tokenizers after running the training notebooks.
* `00_eda.ipynb` contains Exploratory Data Analysis with comprehensive visualizations, class distribution analysis, and text statistics.
* `01_preprocessing.ipynb` contains the text preprocessing pipeline including tokenization, padding, and stopword removal. This notebook runs on any data split.
* `02_train_bert.ipynb` contains the **BERT** model fine-tuning with HuggingFace transformers (~110M parameters).
* `03_train_roberta.ipynb` contains the **RoBERTa** model fine-tuning with improved pre-training (~125M parameters).
* `04_train_electra.ipynb` contains the **ELECTRA** model fine-tuning with discriminative pre-training (~110M parameters).
* `05_inference.ipynb` contains the inference pipeline, model evaluation, and side-by-side performance comparison on test data.
* `report.pdf` contains the PDF for the project report.
* `requirements.txt` contains the Python package dependencies.

## Running The Project 🏃

### Inference 🔎

To predict emotions on a new test dataset (`test.csv`):

1. **Place Data**: Put the `test.csv` file in the `data/` directory.
2. **Preprocess**: Open `01_preprocessing.ipynb`.
   * Set `split = 'test'` in the configuration cell.
   * Run all cells. This will create `data/test_preprocessed.pkl`.
3. **Predict & Evaluate**: Run `05_inference.ipynb`. This notebook will:
   - Load the preprocessed test data.
   - Load the trained BERT, RoBERTa, and ELECTRA models.
   - Generate predictions.
   - Compare performance and display confusion matrices.

### Training 🏋️

In order to train the models and reproduce the results:

1. **Preprocessing**: Open `01_preprocessing.ipynb`.
   * Run first with `split = 'train'` to generate `data/train_preprocessed.pkl`.
   * Run again with `split = 'validation'` to generate `data/validation_preprocessed.pkl`.
2. **Train BERT**: Run `02_train_bert.ipynb`. This will download BERT model and fine-tune it on the emotion dataset.
3. **Train RoBERTa**: Run `03_train_roberta.ipynb`. This will download RoBERTa model and fine-tune it on the emotion dataset.
4. **Train ELECTRA**: Run `04_train_electra.ipynb`. This will download ELECTRA model and fine-tune it on the emotion dataset.

**Note:** Training transformer models requires a GPU for reasonable training times (~30-60 minutes on GPU vs 8-10 hours on CPU). Use Google Colab with GPU runtime if needed.

## Libraries to Install 📚

**Before trying to run anything please make sure to install all the packages below.**

| Library | Command to Run | Minimal Version |
| :--- | :--- | :--- |
| NumPy | pip install numpy | 2.2.5 |
| pandas | pip install pandas | 2.3.3 |
| matplotlib | pip install matplotlib | 3.10.6 |
| seaborn | pip install seaborn | 0.13.2 |
| NLTK | pip install nltk | 3.9.2 |
| scikit-learn | pip install scikit-learn | 1.7.2 |
| Transformers | pip install transformers | 4.36.0 |
| PyTorch | pip install torch | 2.1.0 |
| Datasets | pip install datasets | 2.16.0 |
| Accelerate | pip install accelerate | 0.25.0 |
