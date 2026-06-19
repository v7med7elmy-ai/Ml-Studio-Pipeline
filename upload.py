import streamlit as st
import pandas as pd

# إعدادات الصفحة (تخلي الشكل عريض ومريح للعين)
st.set_page_config(page_title="ML Studio - Upload", page_icon="📁", layout="wide")

# عنوان الصفحة بشكل عصري
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>📁 Machine Learning Studio</h1>
    <h3 style='text-align: center; color: #666;'>Step 1: Data Upload & Exploration</h3>
    <hr style='border-top: 2px solid #4A90E2;'>
""", unsafe_allow_html=True)

# 1. التأكد من تهيئة الـ Session State لحفظ الداتا عبر الصفحات
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = None

# 2. مكان رفع الملف (بيدعم CSV و Excel)
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel format)", 
    type=["csv", "xlsx", "xls"],
    help="Support multiple file extensions for your ML pipeline"
)

# إذا قام المستخدم برفع ملف جديد
if uploaded_file is not None:
    try:
        # قراءة الملف بناءً على الامتداد
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # حفظ الداتا واسم الملف في الـ Session State لضمان ثباتها
        st.session_state['dataset'] = df
        st.session_state['file_name'] = uploaded_file.name
        
        st.success(f"🎉 Success! '{uploaded_file.name}' has been uploaded successfully.")
        
    except Exception as e:
        st.error(f"Error loading file: {e}")

# 3. عرض تفاصيل الداتا (إذا كانت مرفوعة ومحفوظة في الـ State)
if st.session_state['dataset'] is not None:
    df = st.session_state['dataset']
    
    st.markdown("### 📊 Dataset Overview")
    
    # استخدام Columns لعرض الإحصائيات السريعة كـ Cards شكلها حلو
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Rows Count (Samples)", value=df.shape[0])
    with col2:
        st.metric(label="Columns Count (Features)", value=df.shape[1])
    with col3:
        st.metric(label="Missing Values Total", value=df.isna().sum().sum())
        
    st.write("---")
    
    # عرض أول جزء من الداتا
    st.subheader("👀 Data Preview (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("---")
    
    # تفاصيل الأعمدة ونوعها والإحصائيات الاستدلالية
    st.subheader("⚙️ Dataset Structure & Summary")
    
    tab1, tab2, tab3 = st.tabs(["Column Types", "Statistical Summary", "Missing Data per Column"])
    
    with tab1:
        # تجهيز جدول بأنواع البيانات ليكون شكله منظم
        data_types = pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str)
        st.dataframe(data_types, use_container_width=True)
        
    with tab2:
        # الإحصائيات الوصفية (Mean, STD, Min, Max...)
        st.dataframe(df.describe(include='all').fillna('-'), use_container_width=True)
        
    with tab3:
        # حساب القيم المفقودة لكل عمود
        missing_df = pd.DataFrame({
            'Missing Values': df.isna().sum(),
            'Percentage (%)': (df.isna().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)

else:
    # رسالة تظهر لو مفيش ملف اترفع لسه أو لو اليوزر رجع للصفحة وهي فاضية
    st.info("ℹ️ Please upload a dataset to start the machine learning pipeline.")