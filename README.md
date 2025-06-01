# 📨 Naive Bayes Spam Classifier (From Scratch)

This repository demonstrates a complete implementation of a **Naive Bayes classifier** to detect spam messages in an inbox. The project includes both **Python** and **C++** implementations **without using any external machine learning libraries** like `scikit-learn`. It's designed to help beginners understand the theory and math behind Naive Bayes in a hands-on way.

---

## 📘 Concepts Covered

This project walks through the following key concepts in machine learning and text classification:

### 🔹 Bayes' Theorem
The core principle of Naive Bayes, based on:
\[ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \]

### 🔹 Prior
Represents the probability of a class occurring:
\[ P(\text{spam}) \text{ or } P(\text{not spam}) \]

### 🔹 Likelihood
The probability of seeing a message given the class:
\[ P(\vec{w} | \text{spam}) \]

### 🔹 Posterior
What we want to compute:
\[ P(\text{spam} | \vec{w}) \]

### 🔹 Evidence
The total probability of the message appearing:
\[ P(\vec{w}) = P(\vec{w}|spam)P(spam) + P(\vec{w}|not\ spam)P(not\ spam) \]

### 🔹 Conditional Independence (Naive Assumption)
Each word is treated independently of others given the class.

### 🔹 Laplace Smoothing
Used to handle unseen words:
\[ P(w | class) = \frac{count(w)+1}{total\ words + V} \]

---

## 🧪 Dataset
- 50 spam messages
- 50 non-spam (ham) messages
- Each message is a short sentence commonly found in email inboxes

---

## 🚀 Implementations

### ✔ Python
- Uses only built-in modules like `collections`, `math`, and `random`
- Trains on labeled messages
- Predicts the class of new messages

### ✔ C++
- Written using STL: `map`, `set`, `vector`, `string`
- Mirrors the Python logic for learning and prediction
- Includes tokenization, training, and prediction logic

---

## 🧠 How It Works

1. Tokenize each message
2. Count word frequencies per class
3. Compute class priors
4. Use Laplace smoothing for unseen words
5. Compute log-probabilities to avoid underflow
6. Return class with the highest score

---

## 📂 Structure
```
📁 naive-bayes-spam-classifier
├── naive_bayes.ipynb           # Python implementation
├── naive_bayes.cpp          # C++ implementation
└── README.md                # This file
```

---

## 📚 References
- [Wikipedia - Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)

---

## 🙌 Contributing
Pull requests and issues are welcome. Improve the dataset, add more languages, or build a web version!

---

## 📜 License
This project is licensed under the MIT License.

---

## 💡 Author
KareemShaik.com

