import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import os
import json
from sklearn.exceptions import NotFittedError

st.set_page_config(
    page_title="Credit Card — Advanced Risk Prototype",
    layout="wide",
    initial_sidebar_state="expanded",
)

TITLE = "💳 Credit Card — Advanced Scoring & Explainability Prototype"
st.title(TITLE)
st.markdown("**Interactive prototype:** enter applicant data, tune the decision threshold and business rules, view SHAP explanations and base-learner outputs. Designed for interviews & demos.")


# ---------------------------
# Helpers: load artifacts safely
# ---------------------------
@st.cache_resource
def load_artifact(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_all_models():
    stacked = load_artifact("stacked_pipeline.pkl")
    preproc = load_artifact("preprocessor.pkl")
    xgb = load_artifact("xgb_model.pkl")
    feature_names = load_artifact("feature_names.pkl")
    return stacked, preproc, xgb, feature_names

stacked_pipeline, preprocessor, xgb_model, feature_names = load_all_models()

if stacked_pipeline is None:
    st.warning(
        "Model artifacts not found in working directory. "
        "Please put stacked_pipeline.pkl (full pipeline) in this folder. "
        "You can still interact with the UI layout but predictions will be disabled."
    )

# ---------------------------
# Sidebar: business controls & inputs
# ---------------------------
st.sidebar.header("Business Controls")

# Threshold for classifying default from probability
threshold = st.sidebar.slider("Default probability threshold", min_value=0.01, max_value=0.99, value=0.30, step=0.01, help="Lower threshold -> higher recall (catch more defaulters).")

# Decision buckets (customizable)
approve_cutoff = st.sidebar.number_input("Approve cutoff (<=", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
review_cutoff = st.sidebar.number_input("Review cutoff (<=", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
# approve <= approve_cutoff -> Approve
# approve_cutoff < p <= review_cutoff -> Review
# p > review_cutoff -> Decline

st.sidebar.markdown("---")
st.sidebar.subheader("Model & SHAP settings")
show_shap_global = st.sidebar.checkbox("Show global SHAP summary (uses sample)", value=False)
shap_sample_size = st.sidebar.number_input("SHAP sample size (global)", min_value=50, max_value=5000, value=500, step=50)

st.sidebar.markdown("---")
st.sidebar.caption("Place these files in the app folder:\n• stacked_pipeline.pkl\n• preprocessor.pkl\n• xgb_model.pkl\n• feature_names.pkl")

# ---------------------------
# Applicant input form (main columns)
# ---------------------------
st.header("Applicant Input")
c1, c2, c3 = st.columns([1,1,1])

with c1:
    CODE_GENDER = st.selectbox("Gender", options=["M", "F"], index=0)
    AMT_INCOME_TOTAL = st.number_input("Annual Income", min_value=0.0, value=36000.0, step=1000.0)
    NAME_EDUCATION_TYPE = st.selectbox("Education", options=["Secondary", "Higher", "Incomplete Higher", "Lower Secondary", "Academic Degree"])
    DAYS_BIRTH = st.number_input("Age (years)", min_value=18, max_value=90, value=30)
with c2:
    FLAG_OWN_CAR = st.selectbox("Owns car", options=["Y", "N"], index=1)
    CNT_CHILDREN = st.number_input("Number of children", min_value=0, max_value=10, value=0)
    NAME_FAMILY_STATUS = st.selectbox("Family Status", options=["Single", "Married", "Separated", "Widowed"])
    DAYS_EMPLOYED = st.number_input("Years employed", min_value=0, max_value=60, value=2)
with c3:
    NAME_HOUSING_TYPE = st.selectbox("Housing Type", options=["House", "Rented", "Municipal", "With Parents"])
    FLAG_MOBIL = st.selectbox("Mobile phone", options=["Y", "N"])
    FLAG_EMAIL = st.selectbox("Email registered", options=["Y", "N"])
    AMT_CREDIT = st.number_input("Applied credit amount", min_value=0.0, value=500.0, step=50.0)

# compute derived fields if your original pipeline expects DAYS_BIRTH / DAYS_EMPLOYED in days
# If your training used age in years or different feature names, adjust below to match training X
# Here we assume DAYS_BIRTH stored as age in years in training (rename to appropriate column in the input_df creation)

# Build input dataframe (adjust keys to match the exact training columns used in X)
input_df = pd.DataFrame([{
    "CODE_GENDER": CODE_GENDER,
    "FLAG_OWN_CAR": FLAG_OWN_CAR,
    "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
    "CNT_CHILDREN": CNT_CHILDREN,
    "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
    "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
    "NAME_HOUSING_TYPE": NAME_HOUSING_TYPE,
    "YEARS_EMPLOYED": DAYS_EMPLOYED,
    "AGE": DAYS_BIRTH,
    "FLAG_MOBIL": FLAG_MOBIL,
    "FLAG_EMAIL": FLAG_EMAIL,
    "AMT_CREDIT": AMT_CREDIT
}])

st.write("Input preview (this will be sent to the model):")
st.dataframe(input_df.T, use_container_width=True)

# ---------------------------
# Predictions & Explanations
# ---------------------------
st.markdown("---")
col_left, col_right = st.columns([2,3])

with col_left:
    st.subheader("Model scoring")
    if stacked_pipeline is None:
        st.info("Place stacked_pipeline.pkl in this folder to enable scoring. See sidebar for required files.")
    else:
        try:
            proba = stacked_pipeline.predict_proba(input_df)[:,1][0]
            # thresholded prediction
            pred_label = (proba >= threshold).astype(int)

            # Decision rule based on approve/review/decline cutoffs
            if proba <= approve_cutoff:
                decision = "✅ Approve"
                decision_style = "success"
            elif proba <= review_cutoff:
                decision = "🟡 Review (Manual)"
                decision_style = "warning"
            else:
                decision = "❌ Decline"
                decision_style = "error"

            st.metric(label="Default probability", value=f"{proba:.3f}")
            if decision_style == "success":
                st.success(decision)
            elif decision_style == "warning":
                st.warning(decision)
            else:
                st.error(decision)

            # show probability gauge using progress
            st.progress(min(max(proba, 0.0), 1.0))

            # show base learner probs (if available)
            try:
                fitted_stack = stacked_pipeline.named_steps["model"]
                base_probs = {}
                # attempt to get per-estimator proba using preprocessor transform
                # preprocessed row:
                prep_row = preprocessor.transform(input_df)
                if hasattr(prep_row, "toarray"):
                    prep_row = prep_row.toarray()
                for name, est in fitted_stack.named_estimators_.items():
                    try:
                        p = est.predict_proba(prep_row)[:,1][0]
                        base_probs[name] = float(p)
                    except Exception:
                        base_probs[name] = None
                st.subheader("Base learner probabilities")
                st.write(base_probs)
            except Exception:
                st.info("Base learner breakdown not available for this stacking object.")

        except NotFittedError:
            st.error("Model not fitted. Re-train and save stacked_pipeline.pkl")
        except Exception as e:
            st.error("Prediction failed: " + str(e))

with col_right:
    st.subheader("Local explanation (fast SHAP)")
    if xgb_model is None or preprocessor is None or feature_names is None:
        st.info("For fast local SHAP explanations, ensure xgb_model.pkl, preprocessor.pkl and feature_names.pkl are present.")
    else:
        try:
            # Preprocess input to get aligned feature vector
            prep_row = preprocessor.transform(input_df)
            if hasattr(prep_row, "toarray"):
                prep_row = prep_row.toarray()
            # create TreeExplainer on loaded xgb model
            expl = shap.TreeExplainer(xgb_model)
            shap_vals = expl.shap_values(prep_row)
            # shap_vals shape handling
            if isinstance(shap_vals, list):
                # multiclass -> choose index 1 or last class
                shap_vals = np.array(shap_vals[-1]).flatten()
            else:
                shap_vals = np.array(shap_vals).flatten()

            # Map back to feature names
            fnames = feature_names if feature_names is not None else [f"f{i}" for i in range(len(shap_vals))]
            contrib = pd.Series(shap_vals, index=fnames)
            topk = contrib.abs().sort_values(ascending=False).head(10).index.tolist()
            contrib_df = pd.DataFrame({
                "feature": topk,
                "shap_value": contrib[topk].values
            }).sort_values("shap_value")

            st.write("Top contributing features (impact on model output):")
            st.table(contrib_df.set_index("feature"))

            # horizontal bar chart
            fig, ax = plt.subplots(figsize=(6,3))
            contrib_df_sorted = contrib_df.sort_values("shap_value")
            ax.barh(contrib_df_sorted["feature"], contrib_df_sorted["shap_value"])
            ax.set_xlabel("SHAP value (impact on model output)")
            st.pyplot(fig)

        except Exception as e:
            st.error("SHAP explanation failed: " + str(e))

# ---------------------------
# Optional: Global SHAP summary computed on a sample of training data (if available)
# ---------------------------
st.markdown("---")
st.subheader("Global model insights (sample-based)")

if show_shap_global:
    if xgb_model is None or preprocessor is None:
        st.info("Need xgb_model.pkl and preprocessor.pkl to compute SHAP global summary.")
    else:
        # If you have an example training dataset saved (e.g., X_train_sample.pkl), use it.
        # Fallback: ask user to upload a CSV sample of preprocessed data mapped to original columns.
        st.info("Computing SHAP summary on a random sample (this may take a few seconds).")
        # Attempt to load a saved feature sample: X_train_sample_preprocessed.npy or similar
        sample_df = None
        # if user provided preprocessed sample file, use it; otherwise we can't compute global summary
        uploaded = st.file_uploader("Upload a CSV of **raw** training sample (n rows) to compute SHAP summary (optional)", type=["csv"])
        if uploaded is not None:
            try:
                raw_sample = pd.read_csv(uploaded)
                # transform using preprocessor
                X_sample_p = preprocessor.transform(raw_sample)
                if hasattr(X_sample_p, "toarray"):
                    X_sample_p = X_sample_p.toarray()
                # sample rows
                n = min(int(shap_sample_size), X_sample_p.shape[0])
                idx = np.random.choice(X_sample_p.shape[0], n, replace=False)
                Xs = X_sample_p[idx]
                expl = shap.TreeExplainer(xgb_model)
                shap_vals = expl.shap_values(Xs)
                # convert to array if list
                if isinstance(shap_vals, list):
                    shap_vals = np.array(shap_vals[-1])
                # summary plot
                st.pyplot(shap.summary_plot(shap_vals, Xs, feature_names=feature_names, show=False))
            except Exception as e:
                st.error("Global SHAP failed: " + str(e))

# ---------------------------
# Footer: quick tips and export
# ---------------------------
st.markdown("---")
st.write("Quick tips:")
st.write("- Tweak threshold on the left to balance recall vs precision.")
st.write("- Use the Review bucket to route borderline applicants to manual underwriting.")
st.write("- For interviews: show a few sample applicants and explain SHAP drivers.")

# Allow download of prediction + explanation as JSON
if stacked_pipeline is not None:
    if st.button("Export last prediction (JSON)"):
        try:
            # Build export payload
            payload = {
                "input": input_df.to_dict(orient="records")[0],
                "probability": float(proba),
                "decision": decision,
            }
            b = BytesIO()
            b.write(json.dumps(payload, indent=2).encode())
            b.seek(0)
            st.download_button("Download prediction JSON", data=b, file_name="prediction.json", mime="application/json")
        except Exception as e:
            st.error("Failed to export: " + str(e))
