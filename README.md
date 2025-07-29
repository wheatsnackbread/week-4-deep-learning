# NLP Disaster Tweets Classification

## Project Overview

This project implements a **Bidirectional LSTM neural network** to classify tweets as either relating to real disasters or not. It was completed as part of the Week 4 assignment for a Machine Learning course, participating in the Kaggle "Natural Language Processing with Disaster Tweets" competition.

## Problem Statement

Given a dataset of tweets, the goal is to predict whether each tweet is about a real disaster (1) or not (0). This is a binary classification problem that involves natural language processing techniques to understand the semantic content of short text messages.

## Dataset

The dataset consists of:

- **Training set**: ~7,600 labeled tweets
- **Test set**: ~3,260 unlabeled tweets for prediction
- **Features**:
  - `text`: The tweet content
  - `keyword`: A disaster-related keyword (may be missing)
  - `location`: Tweet location (may be missing)
  - `target`: Binary label (1=disaster, 0=non-disaster)

## Methodology

### 1. Exploratory Data Analysis

- Class distribution analysis
- Text length analysis
- Word frequency analysis
- Word clouds for visual understanding
- Missing value assessment

### 2. Text Preprocessing

- URL removal
- Lowercasing and punctuation removal
- Stop word removal
- Tokenization using NLTK

### 3. Feature Engineering

- TF-IDF vectorization for traditional ML comparison
- Word embedding using Keras Tokenizer
- Sequence padding for uniform input length

### 4. Model Architecture

- **Embedding Layer**: Converts word indices to dense vectors
- **Bidirectional LSTM**: Captures context in both directions
- **Dropout Layers**: Prevents overfitting
- **Dense Layers**: Final classification with sigmoid activation

### 5. Training Optimization

- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Stratified train-validation split
  
## Files Structure

```
├── disaster_tweets_classification.ipynb  # Main analysis notebook
├── nlp-getting-started/
│   ├── train.csv                         # Training data
│   ├── test.csv                          # Test data
│   └── sample_submission.csv             # Submission format
├── disaster_tweets_submission.csv        # Final predictions
└── README.md                            # Project documentation
```

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn tensorflow
```

### NLTK Data

The notebook automatically downloads required NLTK data, but you can also run:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Execution

1. Ensure all data files are in the `nlp-getting-started/` directory
2. Open `disaster_tweets_classification.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially
4. The final submission file will be saved as `disaster_tweets_submission.csv`

## Key Features

### Technical Implementation

- **Text Preprocessing**: Comprehensive cleaning pipeline
- **LSTM Architecture**: Bidirectional processing for better context
- **Regularization**: Dropout and early stopping
- **Evaluation**: Multiple metrics and visualization

### Analysis Depth

- **Error Analysis**: Understanding model failures
- **Confidence Analysis**: Prediction reliability assessment
- **Hyperparameter Tuning**: Architecture comparison
- **Performance Visualization**: Training curves and confusion matrices

## Future Improvements

1. **Pre-trained Embeddings**: GloVe or Word2Vec integration
2. **Transformer Models**: BERT-based architectures
3. **Ensemble Methods**: Combining multiple models
4. **Class Balancing**: Addressing dataset imbalance
5. **Feature Enhancement**: Better use of keywords and location data

## Assignment Requirements Met

- ✅ **Problem Description**: Clear NLP context and dataset overview
- ✅ **EDA**: Comprehensive visualization and analysis
- ✅ **Model Architecture**: LSTM with detailed reasoning
- ✅ **Results**: Performance metrics and comparisons
- ✅ **Conclusion**: Thorough discussion of findings
- ✅ **Documentation**: Professional presentation

## Academic Context

This project demonstrates understanding of:

- **NLP Pipeline Development**: From raw text to predictions
- **Neural Network Design**: LSTM architecture choices
- **Model Evaluation**: Comprehensive performance assessment
- **Scientific Communication**: Clear presentation of methodology

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation.
2. Kaggle Competition: Natural Language Processing with Disaster Tweets
3. TensorFlow and Keras Documentation
4. NLTK Documentation for text processing

---

**Note**: This project was completed as coursework with AI assistance used for articulating explanations and comments, while all core analysis and implementation decisions were made independently.
