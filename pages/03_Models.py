import streamlit as st
import pandas as pd
import numpy as np

# استيراد ملفات الباك إيند من فولدر core
from core import regression as reg
from core import classification_clustering as clf_cl
from core import evaluation as eval_mod

# إعدادات الصفحة
st.set_page_config(page_title="ML Studio - Training & Evaluation", page_icon="🤖", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #9B59B6;'>🤖 Model Training & Evaluation Studio</h1>
    <h3 style='text-align: center; color: #666;'>Step 4 & 5: Train Models and Analyze Performance Metrics</h3>
    <hr style='border-top: 2px solid #9B59B6;'>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# تفقد الـ Session State وضمان ثبات البيانات
# -------------------------------------------------------------
if 'processed_dataset' in st.session_state and st.session_state['processed_dataset'] is not None:
    df = st.session_state['processed_dataset']
elif 'dataset' in st.session_state and st.session_state['dataset'] is not None:
    df = st.session_state['dataset']
else:
    df = None

if df is None:
    st.warning("⚠️ No data found! Please upload a dataset or apply preprocessing first.")
else:
    # تقسيم الصفحة إلى جزأين: التحكم على اليسار والنتائج على اليمين
    col_setup, col_results = st.columns([1, 1.2])

    with col_setup:
        st.subheader("🛠️ Configuration Panel")
        
        # 1. تحديد نوع المهمة المطلوبة
        task_type = st.selectbox(
            "Choose ML Problem Type", 
            ["Classification", "Regression", "Clustering"]
        )
        
        st.write("---")
        
        params = {}
        target_column = None
        
        # ------------------- سيناريو الـ Classification -------------------
        if task_type == "Classification":
            target_column = st.selectbox("Select Target Variable (Y)", df.columns.tolist())
            model_choice = st.selectbox(
                "Select Classifier", 
                ["Decision Tree", "Random Forest", "KNN", "SVM", "Logistic Regression",
                 "Bayesian Classifier", "Neural Network"]  # ← الإضافتين الجديدتين هنا بس
            )
            
            st.markdown("#### ⚙️ Hyperparameters")
            if model_choice == "Decision Tree":
                max_depth = st.slider("Max Depth", 1, 30, 5)
                params = {"max_depth": max_depth}
            elif model_choice == "Random Forest":
                n_estimators = st.slider("Number of Estimators", 10, 300, 100, 10)
                params = {"n_estimators": n_estimators}
            elif model_choice == "KNN":
                n_neighbors = st.slider("N Neighbors", 1, 20, 5)
                params = {"n_neighbors": n_neighbors}
            elif model_choice == "SVM":
                c_val = st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
                params = {"C": c_val}
            elif model_choice == "Logistic Regression":
                params = {}
            # ---- Hyperparameters للإضافتين الجديدتين ----
            elif model_choice == "Bayesian Classifier":
                params = {}  # GaussianNB مش محتاج hyperparams
            elif model_choice == "Neural Network":
                nn_layers = st.slider("Hidden Layer Size (neurons)", 10, 300, 100, 10)
                nn_iter = st.slider("Max Iterations", 100, 1000, 300, 50)
                params = {"hidden_layer_sizes": (nn_layers,), "max_iter": nn_iter}

        # ------------------- سيناريو الـ Regression -------------------
        elif task_type == "Regression":
            target_column = st.selectbox("Select Target Variable (Y)", df.select_dtypes(include=['number']).columns.tolist())
            model_choice = st.selectbox(
                "Select Regressor", 
                ["LinearRegression", "Ridge", "Lasso", "DecisionTree", "RandomForest"]
            )
            
            st.markdown("#### ⚙️ Hyperparameters")
            if model_choice in ["Ridge", "Lasso"]:
                alpha = st.slider("Alpha", 0.01, 10.0, 1.0, 0.05)
                params = {"alpha": alpha}
            elif model_choice == "DecisionTree":
                max_depth = st.slider("Max Depth", 1, 30, 5)
                params = {"max_depth": max_depth}
            elif model_choice == "RandomForest":
                n_estimators = st.slider("Number of Estimators", 10, 300, 100, 10)
                params = {"max_depth": 10, "n_estimators": n_estimators}
            else:
                params = {}

        # ------------------- سيناريو الـ Clustering -------------------
        elif task_type == "Clustering":
            st.info("Clustering will auto-select all Numeric features from the current dataset.")
            n_clusters = st.slider("Select Number of Clusters (K)", 2, 15, 3)

        st.write("---")
        start_training = st.button("🚀 Train & Evaluate Model", use_container_width=True)

    # 2. لوحة عرض النتائج والـ Evaluation على اليمين
    with col_results:
        st.subheader("📊 Execution & Evaluation Outputs")
        
        if start_training:
            with st.spinner("Processing pipeline and running models..."):
                try:
                    # تنفيذ الـ Classification بناءً على دالة زميلك المعدلة
                    if task_type == "Classification":

                        # ---- الإضافتين الجديدتين: Bayesian و Neural Network ----
                        if model_choice == "Bayesian Classifier":
                            model, accuracy, report, X_test, y_test = clf_cl.run_bayesian(df, target_column)
                        elif model_choice == "Neural Network":
                            model, accuracy, report, X_test, y_test = clf_cl.run_neural_network(
                                df, target_column,
                                hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
                                max_iter=params.get("max_iter", 300)
                            )
                        else:
                            # استدعاء الدالة المحدثة اللي بترجع 5 متغيرات (تم التعديل هنا لتتوافق مع الباك إيند بالظبط)
                            model, accuracy, report, X_test, y_test = clf_cl.run_classification(df, target_column, model_choice, params)
                        
                        # تمرير الـ X_test و y_test المطرود منهم النصوص مباشرة لملف الـ evaluation
                        eval_results = eval_mod.evaluate_classification(model, X_test, y_test)
                        formatted_text = eval_mod.format_classification_results(eval_results)
                        
                        # عرض كروت النتائج الشيك
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Accuracy Score", f"{eval_results['accuracy']:.4f}")
                        c2.metric("Precision (Weighted)", f"{eval_results['precision']:.4f}")
                        c3.metric("F1-Score", f"{eval_results['f1_score']:.4f}")
                        
                        st.markdown("#### 📝 Detailed Summary Report")
                        st.text_area("Metrics & Confusion Matrix", formatted_text, height=230)
                        
                        st.markdown("#### 📝 Classification Report Matrix")
                        st.text(report)

                    # تنفيذ الـ Regression
                    elif task_type == "Regression":
                        X = df.drop(columns=[target_column]).select_dtypes(include=['number'])
                        y = df[target_column]
                        
                        try:
                            if model_choice == "LinearRegression":
                                model, y_test, y_pred, results = reg.regression_model(X, y, model_choice, params={})
                            else:
                                model, y_test, y_pred, results = reg.regression_model(X, y, model_choice, params=params)
                        except Exception:
                            from sklearn.model_selection import train_test_split
                            from sklearn.metrics import mean_absolute_error, r2_score
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            model = reg.get_model(model_choice, params if model_choice != "LinearRegression" else {})
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            results = {
                                "Test": r2_score(y_test, y_pred),
                                "MAE": mean_absolute_error(y_test, y_pred)
                            }
                        
                        r1, r2 = st.columns(2)
                        r1.metric("R2 Score (Test)", f"{results['Test']:.4f}")
                        r2.metric("Mean Absolute Error (MAE)", f"{results['MAE']:.4f}")
                        
                        st.markdown("#### 📈 Actual vs Predicted Plot")
                        chart_data = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
                        st.line_chart(chart_data.head(50))

# === تعديل جزء الـ Clustering النهائي والآمن ===
                    elif task_type == "Clustering":
                        numeric_df = df.select_dtypes(include=['number'])
                        if numeric_df.empty:
                            st.error("No numeric columns available for Clustering.")
                        else:
                            # 1. تشغيل موديل الكلاسترنج من الباك إيند
                            model, clusters = clf_cl.run_clustering(df, n_clusters)
                            
                            # 2. حساب وعرض الـ Metrics (Silhouette & Inertia)
                            try:
                                eval_results = eval_mod.evaluate_clustering(model, numeric_df, clusters)
                                formatted_text = eval_mod.format_clustering_results(eval_results)
                                
                                cl1, cl2 = st.columns(2)
                                cl1.metric("Silhouette Score", f"{eval_results['silhouette_score']:.4f}")
                                cl2.metric("Inertia (WCSS)", f"{eval_results['inertia']:.2f}")
                                
                                st.text_area("Detailed Clustering Summary", formatted_text, height=150)
                            except Exception:
                                from sklearn.metrics import silhouette_score
                                sil = silhouette_score(numeric_df, clusters)
                                st.metric("Silhouette Score", f"{sil:.4f}")
                                st.metric("Inertia", f"{model.inertia_:.2f}")
                            
                            # 3. الـ الرسمة البيانية الإلزامية للمجموعات (Visualizing Clusters)
                            st.markdown("#### 📊 Clusters Scatter Plot")
                            numeric_cols = numeric_df.columns.tolist()
                            
                            if len(numeric_cols) >= 2:
                                import matplotlib.pyplot as plt
                                import seaborn as sns
                                
                                fig, ax = plt.subplots(figsize=(8, 5))
                                # بنرسم أول عمودين رقميين في الداتا ونلون النقط بناءً على الـ clusters
                                sns.scatterplot(
                                    x=numeric_df[numeric_cols[0]], 
                                    y=numeric_df[numeric_cols[1]], 
                                    hue=clusters, 
                                    palette='Accent', # باليت ألوان شيك ومفصولة
                                    legend='full',
                                    ax=ax
                                )
                                ax.set_title(f"K-Means Clustering Groups ({n_clusters} Clusters)", fontsize=12, fontweight='bold')
                                ax.set_xlabel(numeric_cols[0])
                                ax.set_ylabel(numeric_cols[1])
                                st.pyplot(fig)
                            else:
                                st.warning("⚠️ Need at least 2 numeric columns to draw the scatter plot.")
                            
                            # 4. عرض الجدول في النهاية
                            st.markdown("#### 📌 Dataset Samples with Cluster Labels")
                            df_with_clusters = df.copy()
                            df_with_clusters['Cluster_Label'] = clusters
                            st.dataframe(df_with_clusters.head(10), use_container_width=True)
                    st.success(f"🎉 {task_type} Model trained and evaluated successfully!")

                except Exception as e:
                    st.error(f"❌ Error during training pipeline: {e}")
        else:
            st.info("Configure your machine learning task options on the left and click 'Train & Evaluate Model' to see results here instantly.")