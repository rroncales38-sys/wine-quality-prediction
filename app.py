import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

model = load_model()

st.title("🍷 Wine Quality Prediction System")
st.caption("Preliminary Screening Tool — Not a final laboratory certification")

data = pd.read_csv("wine_quality.csv")
c1, c2, c3 = st.columns(3)
c1.metric("Total Samples", len(data))
c2.metric("Average Quality", round(data["quality"].mean(), 2))
c3.metric("Good Quality (≥6)", int((data["quality"] >= 6).sum()))

st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Upload Your Data")
template = data.drop(columns=["quality"]).head(3)
st.download_button(
    label="📥 Download Sample CSV Template",
    data=template.to_csv(index=False),
    file_name="sample_wine_input.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    new_data = pd.read_csv(uploaded)
    required = list(model.feature_names_in_)
    missing = [col for col in required if col not in new_data.columns]
    
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
    else:
        preds = model.predict(new_data[required])
        results = new_data.copy()
        results["Prediction"] = ["GOOD QUALITY" if p == 1 else "NEEDS IMPROVEMENT" for p in preds]
        
        st.success("✅ Prediction Complete!")
        st.subheader("Results")
        st.dataframe(results, use_container_width=True)
        
        good = sum(preds == 1)
        total = len(preds)
        st.write(f"**Summary:** {good} GOOD | {total-good} NEEDS IMPROVEMENT | {total} total")
        
        st.download_button(
            label="📤 Download Results",
            data=results.to_csv(index=False),
            file_name="wine_quality_results.csv",
            mime="text/csv"
        )
