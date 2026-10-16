
import joblib
from sklearn.pipeline import Pipeline

model = joblib.load("text_classifier.pkl")

print("Model type:", type(model))
print("Model module:", type(model).__module__)

if isinstance(model, Pipeline):
    print("\nPipeline steps:")
    for name, step in model.steps:
        print(f"{name}: {type(step).__name__}")
else:
    print("\nSaved object is not a sklearn Pipeline.")

print("\nHas predict:", hasattr(model, "predict"))
print("Has predict_proba:", hasattr(model, "predict_proba"))
print("Has classes_:", hasattr(model, "classes_"))
print("\nNumber of features:", model.n_features_in_)
print("Classes:", model.classes_)