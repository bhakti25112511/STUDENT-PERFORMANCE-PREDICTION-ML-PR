from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Student Marks Prediction System")
st.caption("Machine Learning mini-project | Linear Regression")

DATA_PATH = Path(__file__).parent / "student-mat.csv"

@st.cache_data
def load_data():
    # Dataset is bundled with the project, so the app works without a dataset download.
    if not DATA_PATH.exists():
        raise FileNotFoundError("student-mat.csv is missing. Keep it in the same folder as app.py.")
    df = pd.read_csv(DATA_PATH, sep=";")
    needed = ["studytime", "G1", "G2", "G3"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[needed].copy()
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)

@st.cache_resource
def train_model(data):
    X = data[["studytime", "G1", "G2"]]
    y = data["G3"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R²": float(r2_score(y_test, predictions)),
    }
    comparison = pd.DataFrame({
        "Actual grade": y_test.to_numpy(),
        "Predicted grade": predictions
    })
    return model, metrics, comparison, X_test, y_test, predictions

try:
    data = load_data()
except Exception as exc:
    st.error("The dataset could not be downloaded automatically.")
    st.write("**Quick fix:** Download the Mathematics CSV from the official UCI dataset page, save it as `student-mat.csv`, and place it beside `app.py` in the project folder. Then refresh this page.")
    st.markdown("[Open the official UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance)")
    st.caption(f"Technical detail: {exc}")
    st.stop()

model, metrics, comparison, X_test, y_test, predictions = train_model(data)

st.success(f"Dataset loaded successfully: {len(data)} student records.")
st.warning("This bundled file is a synthetic demonstration dataset generated for offline testing, not the original UCI student records. Replace it with the official UCI student-mat.csv before reporting this as a real-data study.")
tab1, tab2, tab3, tab4 = st.tabs(["Predict Marks", "Model Performance", "Explore Dataset", "About Project"])

with tab1:
    st.subheader("Enter student information")
    left, right = st.columns(2)
    with left:
        studytime = st.selectbox(
            "Weekly study-time category",
            options=[1, 2, 3, 4],
            index=1,
            format_func=lambda x: {
                1: "1 — Less than 2 hours/week",
                2: "2 — 2 to 5 hours/week",
                3: "3 — 5 to 10 hours/week",
                4: "4 — More than 10 hours/week"
            }[x]
        )
        g1 = st.slider("First-period grade (G1)", min_value=0, max_value=20, value=12)
    with right:
        g2 = st.slider("Second-period grade (G2)", min_value=0, max_value=20, value=13)
        st.info("Grades are on the dataset's 0–20 scale.")
    if st.button("Predict Final Grade", type="primary", use_container_width=True):
        sample = pd.DataFrame({"studytime": [studytime], "G1": [g1], "G2": [g2]})
        raw_prediction = float(model.predict(sample)[0])
        prediction = float(np.clip(raw_prediction, 0, 20))
        st.markdown("### Predicted final Mathematics grade")
        st.metric("Estimated G3", f"{prediction:.2f} / 20")
        st.progress(min(max(prediction / 20, 0.0), 1.0))
        if prediction >= 10:
            st.success("The estimated grade is at or above 10/20 in this dataset's grading scale.")
        else:
            st.warning("The estimated grade is below 10/20. This is only a model estimate, not a judgement of the student.")
        st.caption("This is an educational estimate, not a guarantee. Prediction quality depends on the training data and model.")

with tab2:
    st.subheader("Evaluation on held-out test data")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.3f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.3f}")
    c3.metric("R²", f"{metrics['R²']:.3f}")
    st.write("MAE and RMSE are errors in grade points. R² compares the model with predicting the test-set mean; it can be negative.")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, predictions, alpha=0.75)
    ax.plot([0, 20], [0, 20], linestyle="--")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Actual final grade (G3)")
    ax.set_ylabel("Predicted final grade (G3)")
    ax.set_title("Actual vs Predicted Grades")
    st.pyplot(fig)
    st.subheader("Sample predictions")
    display_df = comparison.copy()
    display_df["Predicted grade"] = display_df["Predicted grade"].round(2)
    st.dataframe(display_df.head(15), use_container_width=True)

with tab3:
    st.subheader("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True)
    st.subheader("Final grade distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data["G3"], bins=np.arange(-0.5, 21.5, 1), edgecolor="black")
    ax.set_xlabel("Final grade (G3), 0–20")
    ax.set_ylabel("Number of students")
    ax.set_title("Distribution of Final Mathematics Grades")
    st.pyplot(fig)
    st.subheader("Summary statistics")
    st.dataframe(data.describe().round(2), use_container_width=True)

with tab4:
    st.subheader("Project overview")
    st.write(
        "This project trains a Linear Regression model to estimate a student's final Mathematics grade (G3) "
        "from weekly study-time category (studytime), first-period grade (G1), and second-period grade (G2)."
    )
    st.markdown("""
    **Machine learning workflow**
    1. Load the UCI Student Performance dataset.
    2. Select relevant features and remove missing records.
    3. Split the records into training (80%) and testing (20%) sets.
    4. Train the Linear Regression model.
    5. Evaluate using MAE, RMSE and R².
    6. Use the trained model to estimate a final grade for new inputs.

    **Limitations**
    - Grades are on a 0–20 scale, not 0–100.
    - `studytime` is a category, not an exact number of hours.
    - `G1` and `G2` are earlier grades, so this is not a prediction made before the student begins studying.
    - A prediction is an estimate and should not be used as the sole basis for important educational decisions.
    """)
    st.markdown("**Dataset note:** The included CSV is synthetic demo data so the project runs offline. For a real-data submission, download the official `student-mat.csv` from [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) and replace the bundled file.")
    st.markdown("**Reference:** Demo CSV: synthetically generated for offline application testing. Official dataset reference (only if replaced with official data): Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T")

st.divider()
st.caption("Student Marks Prediction System | Built with Python, Pandas, scikit-learn, Matplotlib and Streamlit")
