# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score

# =========================================================
# 2. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="LightGBM Classifier Dashboard",
    page_icon="🌿",
    layout="wide"
)

# =========================================================
# 3. CUSTOM CSS
# =========================================================
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #4ade80;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
}
.card {
    background-color: #020617;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(74,222,128,0.3);
}
.metric-box {
    background-color: #020617;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. TITLE
# =========================================================
st.markdown('<div class="title">🌿 LightGBM Classifier Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Social Network Ads Dataset</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# 5. LOAD MODEL & SCALER
# =========================================================
with open("LGBM_Classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================================================
# 6. LOAD DATA (NO UPLOAD)
# =========================================================
dataset = pd.read_csv(r"C:\Users\Admin\Desktop\lgbm\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

X_scaled = scaler.transform(X)

# =========================================================
# 7. SIDEBAR INPUT
# =========================================================
st.sidebar.header("🔧 User Input")

age = st.sidebar.slider("Age", 18, 60, 30)
salary = st.sidebar.slider("Estimated Salary", 15000, 150000, 50000)

user_input = np.array([[age, salary]])
user_input_scaled = scaler.transform(user_input)

# =========================================================
# 8. PREDICTION
# =========================================================
prediction = model.predict(user_input_scaled)[0]
probability = model.predict_proba(user_input_scaled)[0][1]

st.markdown("### 🔮 Prediction Result")
st.markdown('<div class="card">', unsafe_allow_html=True)

if prediction == 1:
    st.success(f"✅ Customer **WILL BUY** (Probability: {probability:.2f})")
else:
    st.error(f"❌ Customer **WILL NOT BUY** (Probability: {probability:.2f})")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 9. MODEL METRICS
# =========================================================
y_pred = model.predict(X_scaled)

accuracy = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Accuracy", f"{accuracy:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Bias (Train Score)", f"{model.score(X_scaled, y):.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Estimators", model.n_estimators)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 10. CONFUSION MATRIX
# =========================================================
st.markdown("### 📊 Confusion Matrix")

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# =========================================================
# 11. ROC CURVE
# =========================================================
st.markdown("### 📈 ROC Curve")

y_prob = model.predict_proba(X_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y, y_prob)
auc_score = roc_auc_score(y, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
ax.plot([0, 1], [0, 1])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

st.pyplot(fig)

# =========================================================
# 12. FOOTER
# =========================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<center>🚀 Built with Streamlit | LightGBM Classifier</center>",
    unsafe_allow_html=True
)
