import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Studio - Upload", page_icon="📁", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>📁 Machine Learning Studio</h1>
    <h3 style='text-align: center; color: #666;'>Step 1: Data Upload & Exploration</h3>
    <hr style='border-top: 2px solid #4A90E2;'>
""", unsafe_allow_html=True)
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = None
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel format)", 
    type=["csv", "xlsx", "xls"],
    help="Support multiple file extensions for your ML pipeline"
)
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
                st.session_state['dataset'] = df
        st.session_state['file_name'] = uploaded_file.name
        st.success(f"🎉 Success! '{uploaded_file.name}' has been uploaded successfully.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
if st.session_state['dataset'] is not None:
    df = st.session_state['dataset']
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Rows Count (Samples)", value=df.shape[0])
    with col2:
        st.metric(label="Columns Count (Features)", value=df.shape[1])
    with col3:
        st.metric(label="Missing Values Total", value=df.isna().sum().sum())
        
    st.write("---")
    st.subheader("👀 Data Preview (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)
    st.write("---")
    st.subheader("⚙️ Dataset Structure & Summary")
    tab1, tab2, tab3 = st.tabs(["Column Types", "Statistical Summary", "Missing Data per Column"])
    with tab1:
        data_types = pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str)
        st.dataframe(data_types, use_container_width=True)
    with tab2:
        st.dataframe(df.describe(include='all').fillna('-'), use_container_width=True)
    with tab3:
        missing_df = pd.DataFrame({
            'Missing Values': df.isna().sum(),
            'Percentage (%)': (df.isna().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
else:
    st.info("ℹ️ Please upload a dataset to start the machine learning pipeline.")
