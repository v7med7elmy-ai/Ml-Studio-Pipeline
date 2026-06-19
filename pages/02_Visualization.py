import streamlit as st
import pandas as pd
import warnings

# استدعاء دوال الرسم بتاعة زميلتك من الملف الجديد جوه core
from core.visualization import plot_line, plot_scatter, plot_box

warnings.filterwarnings('ignore')

# إعدادات الصفحة في ستريم ليت
st.set_page_config(page_title="ML Studio - Visualization", page_icon="📊", layout="wide")

# عنوان الصفحة
st.markdown("""
    <h1 style='text-align: center; color: #2ECC71;'>📊 Data Visualization Studio</h1>
    <h3 style='text-align: center; color: #666;'>Step 2: Explore Data Distributions & Relationships</h3>
    <hr style='border-top: 2px solid #2ECC71;'>
""", unsafe_allow_html=True)

# تفقد الـ Session State وضمان ثبات الداتا
if 'processed_dataset' in st.session_state and st.session_state['processed_dataset'] is not None:
    df = st.session_state['processed_dataset']
elif 'dataset' in st.session_state and st.session_state['dataset'] is not None:
    df = st.session_state['dataset']
else:
    df = None

if df is None:
    st.warning("⚠️ No data found! Please upload a dataset in the Upload page first.")
else:
    # فصل الأعمدة بناءً على النوع
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_cols = df.columns.tolist()

    # تقسيم الصفحة: التحكم على اليسار والرسمة على اليمين
    col_control, col_plot = st.columns([1, 2])

    with col_control:
        st.subheader("⚙️ Plot Settings")
        
        plot_type = st.selectbox(
            "Select Plot Type", 
            ["Scatter Plot", "Line Plot", "Box Plot"],
            help="Choose the visualization technique"
        )
        
        st.write("---")

        # تخصيص الاختيارات بناءً على دوال زميلتك ومقترحاتها
        if plot_type == "Scatter Plot":
            st.info("💡 Best for finding relationships between two Numeric variables.")
            x_axis = st.selectbox("Select X-Axis (Numeric)", num_cols if num_cols else all_cols, key="sc_x")
            y_axis = st.selectbox("Select Y-Axis (Numeric)", num_cols if num_cols else all_cols, key="sc_y")
            hue_col = st.selectbox("Color by (Hue) - Optional", ["None"] + cat_cols + num_cols, index=0, key="sc_h")

        elif plot_type == "Line Plot":
            st.info("💡 Best for trends over time or ordered numeric sequences.")
            x_axis = st.selectbox("Select X-Axis (Optional, default is Index)", ["Index"] + all_cols, key="ln_x")
            y_axis = st.selectbox("Select Y-Axis (Numeric)", num_cols if num_cols else all_cols, key="ln_y")
            hue_col = st.selectbox("Color by (Hue) - Optional", ["None"] + cat_cols, index=0, key="ln_h")

        elif plot_type == "Box Plot":
            st.info("💡 Best for comparing distributions across Categorical groups.")
            x_axis = st.selectbox("Select Grouping Variable X (Optional)", ["None"] + cat_cols + num_cols, index=0, key="bx_x")
            y_axis = st.selectbox("Select Numeric Variable Y (Value)", num_cols if num_cols else all_cols, key="bx_y")
            hue_col = st.selectbox("Color by (Hue) - Optional", ["None"] + cat_cols, index=0, key="bx_h")

        # زر لتأكيد الرسم
        generate_plot = st.button("🚀 Generate Visualization", use_container_width=True)

    # جزء عرض الرسمة على اليمين
    with col_plot:
        st.subheader("🖼️ Visualization Output")
        
        if generate_plot:
            try:
                # تنفيذ واستدعاء دوال الـ core مباشرة وعرض الـ fig الناتج
                if plot_type == "Scatter Plot":
                    fig = plot_scatter(df, x=x_axis, y=y_axis, hue=hue_col)
                    
                elif plot_type == "Line Plot":
                    x_param = None if x_axis == "Index" else x_axis
                    fig = plot_line(df, y=y_axis, x=x_param, hue=hue_col)
                    
                elif plot_type == "Box Plot":
                    fig = plot_box(df, y=y_axis, x=x_axis, hue=hue_col)
                
                # العرض الفعلي للرسمة المرجعة من الفانكشن
                st.pyplot(fig)
                st.success(f"🎉 Successfully generated {plot_type} using Core Functions!")
                
            except Exception as e:
                st.error(f"❌ Couldn't generate plot: {e}. Ensure column types are correct.")
        else:
            st.info("Configure the options on the left and click 'Generate Visualization' to see the chart.")