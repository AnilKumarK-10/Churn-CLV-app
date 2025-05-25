import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Customer Churn & CLV Dashboard", layout="wide")
st.title("📊 Customer Churn & CLV Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("final_outputs.csv")

data = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Customers")
action_filter = st.sidebar.selectbox("Filter by Recommended Action", options=["All", "Retain", "Ignore"])
min_churn_prob = st.sidebar.slider("Minimum Churn Probability", 0.0, 1.0, 0.0)

# Apply filters to the data
filtered_data = data.copy()
if action_filter != "All":
    filtered_data = filtered_data[filtered_data["Action"] == action_filter]
filtered_data = filtered_data[filtered_data["Churn_Prob"] >= min_churn_prob]

# Overview Metrics
st.subheader("🔢 Overview Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(filtered_data):,}")
col2.metric("Avg. Predicted CLV", f"${filtered_data['Predicted_CLV'].mean():.2f}")
col3.metric("Avg. Churn Probability", f"{filtered_data['Churn_Prob'].mean():.1%}")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Customer Table", "📈 Visual Insights", "🔍 Customer Drilldown", "📤 Export"])

# =================== TAB 1: Table ====================
with tab1:
    st.subheader("🧾 Filtered Customer Table")
    st.dataframe(filtered_data.sort_values("Churn_Prob", ascending=False).reset_index(drop=True))

# =================== TAB 2: Visuals ===================
with tab2:
    st.subheader("💡 Churn Probability Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.hist(filtered_data["Churn_Prob"], bins=30, color="salmon", edgecolor="black")
    ax1.set_title("Churn Probability of All Customers")
    ax1.set_xlabel("Churn_Prob")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    st.subheader("💡 CLV Segmentation")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    tertile_labels = ['Low', 'Medium', 'High']
    filtered_data['CLV_Segment'] = pd.qcut(filtered_data['Predicted_CLV'], q=3, labels=tertile_labels)
    clv_counts = filtered_data['CLV_Segment'].value_counts().reindex(tertile_labels)
    ax2.bar(clv_counts.index, clv_counts.values, color='skyblue')
    ax2.set_title("CLV Segments (Tertiles)")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

# =================== TAB 3: Drilldown ===================
with tab3:
    st.subheader("🔍 Customer Details")

    if filtered_data.empty:
        st.warning("No customers match the selected filters.")
    else:
        cust_id = st.selectbox("Select Customer ID", filtered_data["Customer_Id"].sort_values().unique())
        row = filtered_data[filtered_data["Customer_Id"] == cust_id].iloc[0]

        st.markdown(f"🎯 **Action:** `{row['Action']}`")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Churn Probability", f"{row['Churn_Prob']:.1%}")
        col2.metric("Predicted CLV", f"${row['Predicted_CLV']:.2f}")
        col3.metric("Expected Loss", f"${row['ExpectedLoss']:.2f}")
        percentile = np.round((filtered_data['Predicted_CLV'] < row['Predicted_CLV']).mean() * 100, 1)
        col4.metric("CLV Percentile Rank", f"{percentile}th")

        st.subheader("📊 Customer in Context")
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(filtered_data["Predicted_CLV"], bins=50, color="lightblue", edgecolor="black")
            ax.axvline(row["Predicted_CLV"], color="red", linestyle="--", linewidth=2, label="Selected Customer")
            ax.set_title("CLV Distribution")
            ax.set_xlabel("CLV")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(filtered_data["Churn_Prob"], bins=30, color="salmon", edgecolor="black")
            ax.axvline(row["Churn_Prob"], color="blue", linestyle="--", linewidth=2, label="Selected Customer")
            ax.set_title("Churn Probability Distribution")
            ax.set_xlabel("Churn Prob")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)

# =================== TAB 4: Export ===================
with tab4:
    st.subheader("📤 Export Filtered Customers")
    st.download_button("Download CSV", filtered_data.to_csv(index=False), file_name="filtered_customers.csv")

# Footer
st.markdown("---")
st.caption("Created Anil,Abhi,Rupesh,Venu")
