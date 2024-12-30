# DL for NLP - Transformers

This repository contains the implementation of **Exercise 03** for the "Deep Learning for NLP" course. The exercise focuses on fine-tuning two Transformer-based models, **BERT** for sequence classification and **T5** for text summarization. The project utilizes the Hugging Face ecosystem, including its datasets, tokenizers, and models.

## Overview

The project includes the following tasks:

1. **Fine-tuning BERT for Sequence Classification**:
   - Sentiment classification task using the DAIR-AI Emotion Dataset.
   - Tokenization and vocabulary building using Hugging Face's `AutoTokenizer`.
   - Model fine-tuning using Hugging Face's `Trainer`.
   - Incorporation of class imbalance handling via loss weighting.

2. **Fine-tuning T5 for Text Summarization**:
   - Summarization task using the SamSum dataset.
   - Text-to-text model fine-tuning using Hugging Face's `Seq2SeqTrainer`.
   - Evaluation using ROUGE scores for qualitative and quantitative analysis.

This repository follows the initial template provided for the course. The instructors provided:
- Skeleton code for preprocessing, training, and evaluation pipelines.
- General submission guidelines, including required imports and dataset structures.

## My Contributions

### Task 1: Sequence Classification
- Preprocessed the Emotion dataset.
- Implemented the BERT fine-tuning pipeline, including:
  - Training loop.
  - Hyperparameter tuning (e.g., learning rate, batch size, epochs).
  - Handling class imbalance using weighted loss functions.
- Evaluated and visualized classification results.

### Task 2: Text Summarization
- Preprocessed the SamSum dataset for summarization.
- Fine-tuned T5 for summarization using Hugging Face's `Seq2SeqTrainer`.
- Evaluated the model using ROUGE metrics and analyzed performance.

## Key Features

### Task 1: Sequence Classification with BERT
- **Dataset**:
  - Emotion classification dataset from Hugging Face.
  - Includes six emotion classes with a predefined train-validation-test split.
- **Model Architecture**:
  - Pre-trained `bert-base-uncased` model fine-tuned for multi-class classification.
  - Weighted cross-entropy loss to address class imbalance.
- **Training**:
  - Hyperparameter tuning with early stopping and TensorBoard logging.
  - F1 score as the primary evaluation metric.
- **Evaluation**:
  - Detailed predictions per class.
  - Visualization of class distributions and results.

### Task 2: Text Summarization with T5
- **Dataset**:
  - SamSum dataset, containing dialogues and human-written summaries.
- **Model Architecture**:
  - Pre-trained `t5-small` model fine-tuned for text summarization.
- **Training**:
  - Sequence-to-sequence fine-tuning with adjusted loss functions.
  - Evaluation using ROUGE metrics.
- **Evaluation**:
  - Analysis of generated summaries for coherence and adherence to context.

## Requirements
matplotlib==3.7.1
nltk==3.8.1
pandas==1.5.3
scikit-learn==1.2.2
scipy==1.10.1
seaborn==0.12.2
spacy==3.6.1
datasets==3.0.1
torch==2.0.1
torchtext==0.15.2
transformers==4.24.0
evaluate==0.4.0
rouge-score==0.1.2
py7zr
bertviz

