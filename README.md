# Roman Urdu NLP — Sentiment Analysis

A sentiment classification model for Roman Urdu (Urdu written in Latin script), fine-tuned on XLM-RoBERTa. Built to address a low-resource, code-mixed language largely underserved by mainstream NLP tooling.

## Overview

Roman Urdu is the informal script most commonly used across Pakistani social media, messaging, and reviews — yet it's inconsistent, code-mixed with English, and rarely covered by pretrained multilingual models out of the box. This project fine-tunes XLM-RoBERTa on a large Roman Urdu sentiment corpus to classify text as positive, negative, or neutral.

## Dataset

- Source: `Khubaib01/RomanUrdu-NLP-Sentiment-Corpus`
- ~134K raw samples, cleaned down to ~129K after preprocessing

## Pipeline

1. **Preprocessing** — cleaning, normalization, and filtering of the raw corpus (134K → 129K rows).
2. **Tokenization** — XLM-RoBERTa tokenizer applied to the cleaned dataset.
3. **Fine-tuning** — trained on Kaggle's free T4 GPU environment (`03_train_model.py`), with loss tracked across epochs.
4. **Evaluation** — performance measured via scikit-learn's classification report (precision, recall, F1 per class).

## Stack

- Python
- Hugging Face Transformers (XLM-RoBERTa)
- PyTorch
- scikit-learn
- Kaggle (T4 GPU training)

## Status

Model has completed preprocessing, tokenization, and fine-tuning; evaluation is in progress. Published on Hugging Face.

## Why This Project

Local-language, code-mixed NLP is a niche most large models don't handle well — that gap is the point. Building for Roman Urdu specifically is both a technical exercise and a bet on underserved language data as a differentiator.
