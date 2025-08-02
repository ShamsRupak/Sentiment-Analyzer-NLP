# 🎬 Sentiment Analyzer NLP - Movie Review Classification

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Last Commit](https://img.shields.io/github/last-commit/ShamsRupak/Sentiment-Analyzer-NLP)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
[![GitHub Stars](https://img.shields.io/github/stars/ShamsRupak/Sentiment-Analyzer-NLP?style=social)](https://github.com/ShamsRupak/Sentiment-Analyzer-NLP)

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="400">
</p>

## 🎯 Overview

Welcome to **Sentiment Analyzer NLP**! 🚀 This beginner-friendly project uses Natural Language Processing (NLP) and machine learning to analyze the sentiment of movie reviews. Simply input any movie review, and the model will predict whether it's **positive** 😊 or **negative** 😞 with confidence scores!

### ✨ Key Features

- 🎭 **Binary Sentiment Classification** - Positive or Negative
- 📊 **High Accuracy** - Achieves ~85% accuracy on test data
- 💬 **Interactive Mode** - Analyze your own reviews in real-time
- 🎯 **Confidence Scores** - See how confident the model is
- 🚀 **Quick Setup** - Run in minutes with minimal dependencies
- 📈 **Multiple Models** - Choose between Naive Bayes or Logistic Regression

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="96">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="48" height="48" alt="Python" />
<br><b>Python</b>
</td>
<td align="center" width="96">
<img src="https://avatars.githubusercontent.com/u/1194118?s=200&v=4" width="48" height="48" alt="NLTK" />
<br><b>NLTK</b>
</td>
<td align="center" width="96">
<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="48" height="48" alt="Scikit-learn" />
<br><b>Scikit-learn</b>
</td>
<td align="center" width="96">
<img src="https://numpy.org/images/logo.svg" width="48" height="48" alt="NumPy" />
<br><b>NumPy</b>
</td>
</tr>
</table>

### 📚 Key Libraries & Components

- **NLTK** - Natural Language Toolkit for text preprocessing
- **TfidfVectorizer** - Converts text to numerical features
- **Naive Bayes / Logistic Regression** - ML classifiers
- **movie_reviews corpus** - 2000 labeled movie reviews

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShamsRupak/Sentiment-Analyzer-NLP.git
   cd Sentiment-Analyzer-NLP
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analyzer**
   ```bash
   python sentiment_analyzer.py
   ```

That's it! 🎉 The script will:
- Download required NLTK data automatically
- Load and preprocess the movie reviews dataset
- Train a sentiment classifier
- Enter interactive mode for you to test

## 💻 How to Use

### Running the Script

```bash
python sentiment_analyzer.py
```

### Choose Your Model

When prompted, select your preferred classifier:

```
📚 Choose classifier type:
1. Naive Bayes (faster, good for text)
2. Logistic Regression (more accurate, slower)

Enter choice (1 or 2) [default: 1]: 
```

### Interactive Mode

After training, enter your own movie reviews:

```
💬 Your review: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.

==================================================
📊 ANALYSIS RESULTS
==================================================
🎭 Sentiment: Positive 😊
💪 Confidence: 94.2%
📊 [████████████████████████████░░] 94.2%
✨ Very confident prediction!
==================================================
```

## 📊 Sample Output

<details>
<summary><b>Click to see full output example</b></summary>

```
🎬 SENTIMENT ANALYZER - Movie Review Classification
============================================================
Author: Shams Rupak | GitHub: @ShamsRupak
============================================================

🔧 Initializing Sentiment Analyzer...
📥 Downloading required NLTK data...

📊 Loading movie reviews dataset...
✅ Loaded 2000 reviews (1000 positive, 1000 negative)
🔧 Preprocessing text...

✂️  Splitting data into train/test sets (80/20 split)...
📊 Training samples: 1600
📊 Testing samples: 400

🔤 Converting text to numerical features using TF-IDF...
✅ Feature dimensions: 5000

🤖 Training Naive Bayes classifier...
✅ Model training complete!

📊 Evaluating model performance...
🎯 Model Accuracy: 84.75%

📈 Classification Report:
--------------------------------------------------
              precision    recall  f1-score   support

Negative 😞       0.83      0.88      0.85       200
Positive 😊       0.87      0.82      0.84       200

    accuracy                           0.85       400
   macro avg       0.85      0.85      0.85       400
weighted avg       0.85      0.85      0.85       400

🔢 Confusion Matrix:
   Predicted: Neg  Pos
Actual Neg:   175   25
Actual Pos:    36  164
```

</details>

## 🧠 How It Works

<details>
<summary><b>Click to expand the technical explanation</b></summary>

### 1. **Data Loading** 📥
- Loads the NLTK movie_reviews corpus
- Contains 1000 positive and 1000 negative reviews

### 2. **Text Preprocessing** 🔧
- Convert to lowercase
- Remove special characters and numbers
- Tokenization using NLTK
- Remove stopwords (common words like "the", "is", etc.)
- Filter out words shorter than 3 characters

### 3. **Feature Extraction** 🔤
- **TF-IDF Vectorization**: Converts text to numerical features
- Considers word frequency and importance
- Creates a sparse matrix of 5000 top features

### 4. **Model Training** 🤖
- **Naive Bayes**: Probabilistic classifier, fast and effective for text
- **Logistic Regression**: Linear model, often more accurate but slower
- 80/20 train-test split with stratification

### 5. **Interactive Prediction** 💬
- Preprocesses user input using the same pipeline
- Transforms to TF-IDF features
- Predicts sentiment with confidence score

</details>

## 📈 Performance Metrics

| Metric | Naive Bayes | Logistic Regression |
|--------|-------------|-------------------|
| Accuracy | ~84-86% | ~86-88% |
| Training Time | <1 second | ~2-3 seconds |
| Prediction Speed | Very Fast | Fast |

## 🎮 Try These Reviews!

Test the analyzer with these example reviews:

### Positive Examples 😊
```
"This movie is a masterpiece! The cinematography is breathtaking and the performances are outstanding."

"I was blown away by this film. It's rare to see such perfect storytelling and character development."
```

### Negative Examples 😞
```
"Terrible movie. The plot made no sense and the acting was wooden. Complete waste of time."

"I fell asleep halfway through. Boring, predictable, and poorly executed."
```

### Ambiguous Examples 🤔
```
"The movie had some good moments but overall felt rushed and incomplete."

"Interesting concept but the execution could have been much better."
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests
- ⭐ Star the repository

## 📚 Learning Resources

Want to learn more about NLP and sentiment analysis?

- 📖 [NLTK Book](https://www.nltk.org/book/)
- 🎓 [Scikit-learn Text Processing](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- 📺 [Sentiment Analysis Tutorial](https://www.youtube.com/watch?v=ujId4ipkBio)
- 🔬 [TF-IDF Explained](https://monkeylearn.com/blog/what-is-tf-idf/)

## 🚀 Future Enhancements

- [ ] Add support for neutral sentiment
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add visualization of most important words
- [ ] Create a web interface
- [ ] Support for multiple languages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Shams Rupak**
- GitHub: [@ShamsRupak](https://github.com/ShamsRupak)
- Project Link: [https://github.com/ShamsRupak/Sentiment-Analyzer-NLP](https://github.com/ShamsRupak/Sentiment-Analyzer-NLP)

---

<p align="center">
  Made with ❤️ and 🐍 by Shams Rupak
  <br>
  ⭐ Star this repo if you found it helpful!
</p>
