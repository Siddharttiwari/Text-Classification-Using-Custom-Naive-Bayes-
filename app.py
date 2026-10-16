
import streamlit as st
import joblib
import numpy as np

from pathlib import Path
from collections import Counter
from nltk.tokenize import word_tokenize

st.set_page_config(
    page_title="Naive Bayes Comparison",
    page_icon="📝"
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    sklearn_model = joblib.load(
        BASE_DIR / "text_classifier.pkl"
    )
    manual_model = joblib.load(
        BASE_DIR / "manual_nb.pkl"
    )
    vocabulary = joblib.load(
        BASE_DIR / "vocabulary.pkl"
    )
    categories = joblib.load(
        BASE_DIR / "category_names.pkl"
    )

    assert len(vocabulary) == sklearn_model.n_features_in_

    return sklearn_model, manual_model, vocabulary, categories


def vectorize(text, vocabulary):
    tokens = word_tokenize(text)
    counts = Counter(tokens)

    return np.array(
        [[counts.get(word, 0) for word in vocabulary]],
        dtype=np.float64
    )


def manual_predict(vector, dictionary, vocabulary):
    # Reproduce the notebook's manual probability function.
    # It uses presence, not frequency, for prediction.
    present_indices = np.flatnonzero(vector)

    best_class = None
    best_score = float("-inf")

    for class_id in dictionary.keys():
        total = dictionary[class_id]["total"]
        denominator = total + len(vocabulary)

        score = 0.0

        for index in present_indices:
            word = vocabulary[index]
            count = dictionary[class_id][word]

            score += np.log(
                (count + 1) / denominator
            )

        if best_class is None or score > best_score:
            best_class = class_id
            best_score = score

    return int(best_class)


st.title("📝 Text Classification")
st.write("Compare manual and Scikit-learn Naive Bayes.")

try:
    sklearn_model, manual_model, vocabulary, categories = (
        load_artifacts()
    )
except Exception as error:
    st.error(f"Loading error: {error}")
    st.stop()

choice = st.radio(
    "Select model",
    [
        "Manual Naive Bayes",
        "Scikit-learn Naive Bayes",
        "Compare Both"
    ]
)

text = st.text_area(
    "Enter a news article",
    height=160
)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            vector = vectorize(text, vocabulary)

            if not np.any(vector):
                st.warning(
                    "No vocabulary words matched. "
                    "Predictions may be unreliable."
                )

            if choice in ["Manual Naive Bayes", "Compare Both"]:
                manual_id = manual_predict(
                    vector[0], manual_model, vocabulary
                )

                st.subheader("Manual Naive Bayes")
                st.success(categories[manual_id])

            if choice in ["Scikit-learn Naive Bayes", "Compare Both"]:
                sklearn_id = int(
                    sklearn_model.predict(vector)[0]
                )

                st.subheader("Scikit-learn Naive Bayes")
                st.success(categories[sklearn_id])

                probabilities = sklearn_model.predict_proba(
                    vector
                )[0]

                st.subheader("Top 5 Probabilities")

                top_indices = np.argsort(
                    probabilities
                )[::-1][:5]

                for index in top_indices:
                    class_id = int(sklearn_model.classes_[index])
                    st.write(
                        f"{categories[class_id]}: "
                        f"{probabilities[index]:.2%}"
                    )
                    st.progress(float(probabilities[index]))

            if choice == "Compare Both":
                if manual_id == sklearn_id:
                    st.info("Both models agree!")
                else:
                    st.warning(
                        "Models predicted different categories."
                    )

        except Exception as error:
            st.error(f"Prediction error: {error}")
