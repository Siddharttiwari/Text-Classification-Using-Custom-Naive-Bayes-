
# 📝 Text Classification Using Naive Bayes

### From-Scratch Implementation vs Scikit-learn | NLP | Streamlit

A machine learning project that classifies text documents into **20 news categories** using the **Multinomial Naive Bayes algorithm**.

The project implements Naive Bayes manually to understand its underlying mathematics and compares its predictions with Scikit-learn's built-in `MultinomialNB` classifier.

An interactive **Streamlit application** allows users to enter text, select a classifier, and compare predictions from both implementations.

---

## 🚀 Key Features

- **20-Class Text Classification** using the 20 Newsgroups dataset.
- **Manual Naive Bayes Implementation** built from scratch.
- **Scikit-learn MultinomialNB** for comparison.
- **Custom Text Preprocessing** using NLTK.
- **9,980 Word-Frequency Features** for document representation.
- **Interactive Streamlit Application** for real-time predictions.
- **Model Comparison** showing predictions from both classifiers.
- **Prediction Probabilities** for the Scikit-learn model.
- **Saved Models** using Joblib for inference without retraining.

---

## 📊 Dataset

**Dataset:** 20 Newsgroups

The dataset contains approximately **11,314 text documents** across 20 discussion categories.

Examples include:

| Category | Description |
|----------|-------------|
| `comp.graphics` | Computer graphics |
| `comp.os.ms-windows.misc` | Microsoft Windows |
| `rec.sport.hockey` | Hockey discussions |
| `rec.sport.baseball` | Baseball discussions |
| `sci.space` | Space and astronomy |
| `sci.med` | Medicine and healthcare |
| `sci.electronics` | Electronics |
| `talk.politics.misc` | Political discussions |
| `alt.atheism` | Atheism discussions |
| `misc.forsale` | Items for sale |

**Total Classes:** 20

**Source:** [20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)

---

## 🧠 Project Workflow

```text
                   Raw Text Documents
                           |
                           v
                    Text Tokenization
                           |
                           v
                  Vocabulary Generation
                           |
                           v
               Word-Frequency Extraction
                           |
                           v
                9,980 Feature Vectors
                           |
                           v
                  Train / Test Split
                           |
             +-------------+-------------+
             |                           |
             v                           v
       Manual Naive Bayes       Scikit-learn MultinomialNB
             |                           |
             v                           v
       Manual Prediction        Built-in Prediction
             |                           |
             +-------------+-------------+
                           |
                           v
                   Model Comparison
                           |
                           v
                    Streamlit App
```

---

## ⚙️ Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Machine Learning | Scikit-learn |
| NLP | NLTK |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Development | Jupyter Notebook, VS Code |

---

## 🔬 Model Implementations

### 1. Manual Naive Bayes

Implemented Naive Bayes classification using custom Python logic.

Key components:

- Vocabulary construction
- Word-frequency counting
- Conditional probability estimation
- Laplace smoothing
- Log-probability calculations
- Document classification

This implementation demonstrates how Naive Bayes works internally without relying on Scikit-learn's classifier implementation.

### 2. Scikit-learn Multinomial Naive Bayes

Used Scikit-learn's `MultinomialNB` to train and evaluate a second classifier on numerical text features.

```python
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(x_train_modified, y_train)

predictions = clf.predict(x_test_modified)
```

Both implementations are available for comparison in the Streamlit application.

---

## 📈 Model Performance

The project achieved approximately **78.4% classification accuracy** during notebook evaluation.

| Metric | Result |
|--------|--------|
| Classification task | Multiclass |
| Number of classes | 20 |
| Input features | 9,980 |
| Reported accuracy | 78.4% |
| Algorithm | Naive Bayes |

The notebook also compares the manually implemented classifier with Scikit-learn's MultinomialNB.

> The reported accuracy is from the notebook evaluation, not from individual examples entered into Streamlit.

---

## 🖥️ Streamlit Application

The application provides three classification modes:

1. **Manual Naive Bayes**
2. **Scikit-learn Naive Bayes**
3. **Compare Both**

### Example Input

```text
NASA scientists launched a new satellite
to explore planets and space.
```

### Example Output

```text
Manual Naive Bayes:
sci.space

Scikit-learn Naive Bayes:
sci.space
```

The application also displays the top predicted categories and their estimated probabilities for the Scikit-learn model.

### Screenshots

Add screenshots of your application to a `screenshots` folder.

```markdown
![Text Classification App](screenshots/app.png)

![Model Comparison](screenshots/comparison.png)
```

---

## 📁 Project Structure

```text
Text_Classification/
│
├── app.py
├── Text_Classification.ipynb
│
├── text_classifier.pkl
├── manual_nb.pkl
├── vocabulary.pkl
├── category_names.pkl
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── screenshots/
    ├── app.png
    └── comparison.png
```

The `screenshots` directory is optional.

---

## 🛠️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Text_Classification.git
cd Text_Classification
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Tokenizer Data

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
```

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
streamlit
scikit-learn==1.6.1
joblib
numpy
pandas
nltk
```

Scikit-learn 1.6.1 matches the version used to save the current trained model.

---

## 🔍 Example Predictions

| Input Topic | Predicted Category |
|-------------|-------------------|
| NASA satellite mission | `sci.space` |
| OpenGL and 3D rendering | `comp.graphics` |
| Windows display driver issues | `comp.os.ms-windows.misc` |

These are examples tested using the application.

---

## 🎯 What I Learned

- Implementing Naive Bayes classification from scratch.
- Understanding conditional probabilities and Laplace smoothing.
- Building custom word-frequency feature representations.
- Comparing custom algorithms with Scikit-learn implementations.
- Maintaining consistent preprocessing during training and inference.
- Saving trained models and vocabulary using Joblib.
- Deploying machine learning inference through Streamlit.

---

## 🔮 Future Improvements

- Add model evaluation dashboards.
- Compare precision, recall, and F1-score.
- Visualize confusion matrices.
- Explain influential words for predictions.
- Improve the user interface.
- Deploy the application to Streamlit Community Cloud.

---

## 👨‍💻 Author

**Siddharth Tiwari**

MCA Student | Python | Machine Learning | Deep Learning | Generative AI

- GitHub: [Siddharth Tiwari](https://github.com/Siddharttiwari)
- LinkedIn: Add your LinkedIn profile URL

---

⭐ If you find this project useful, consider starring the repository.
