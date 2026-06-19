import streamlit as st
import pandas as pd

from core.p_ml import (
    get_numeric_columns,
    handle_missing_values,
    clean_text,
    apply_label_encoding,
    apply_one_hot,
    remove_outliers_iqr,
    feature_selection,
    apply_standard_scaler,
    apply_minmax_scaler,
    apply_pca,
    handle_imbalance_smote,
    handle_missing_knn,
    handle_missing_iterative,
    remove_outliers_zscore,
    remove_outliers_winsorize,
    remove_outliers_clipping,
    apply_log_transformation,
    apply_boxcox_transformation,
    apply_power_transformation,
    apply_polynomial_features,
    apply_rfe,
    handle_imbalance_undersample,
)

st.set_page_config(page_title="ML Studio - Preprocessing", page_icon="⚙️", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #E67E22;'>⚙️ Data Preprocessing Studio</h1>
    <h3 style='text-align: center; color: #666;'>Step 3: Clean, Encode, and Transform Your Data</h3>
    <hr style='border-top: 2px solid #E67E22;'>
""", unsafe_allow_html=True)

if 'dataset' not in st.session_state or st.session_state['dataset'] is None:
    st.warning("⚠️ No data found! Please upload a dataset in the Upload page first.")
else:
    if 'processed_dataset' not in st.session_state or st.session_state['processed_dataset'] is None:
        st.session_state['processed_dataset'] = st.session_state['dataset'].copy()
        st.session_state['applied_steps'] = []
        st.session_state['encoders_dict'] = {}

    df_current = st.session_state['processed_dataset']

    if st.sidebar.button("🔄 Reset to Original Data"):
        st.session_state['processed_dataset'] = st.session_state['dataset'].copy()
        st.session_state['applied_steps'] = []
        st.session_state['encoders_dict'] = {}
        st.rerun()

    if st.session_state.get('applied_steps'):
        st.sidebar.markdown("### ✅ Applied Steps")
        for i, step in enumerate(st.session_state['applied_steps'], 1):
            st.sidebar.markdown(f"{i}. {step}")

    col_control, col_preview = st.columns([1, 1.2])

    with col_control:
        st.subheader("🛠️ Control Panel")

        # ============================================================
        # STEP 1 — HANDLE MISSING VALUES
        # ============================================================
        with st.expander("1️⃣ Handle Missing Values", expanded=False):
            missing_method = st.radio(
                "Choose Imputation Method:",
                ["Mean & Mode (Simple)", "KNN Imputer", "Iterative Imputer (MICE)"],
                key="rb_missing"
            )

            if missing_method == "Mean & Mode (Simple)":
                st.info("Fills numeric columns with Mean, categorical with Mode.")
                if st.button("Apply Simple Imputer", key="btn_mv"):
                    st.session_state['processed_dataset'] = handle_missing_values(df_current)
                    st.session_state['applied_steps'].append("Simple Imputer (Mean & Mode)")
                    st.success("Missing values handled!")
                    st.rerun()

            elif missing_method == "KNN Imputer":
                st.info("Fills missing numeric values based on K nearest neighbors.")
                knn_k = st.slider("Number of Neighbors (K)", 1, 15, 5, key="sl_knn_k")
                if st.button("Apply KNN Imputer", key="btn_knn"):
                    try:
                        st.session_state['processed_dataset'] = handle_missing_knn(df_current, n_neighbors=knn_k)
                        st.session_state['applied_steps'].append(f"KNN Imputer (K={knn_k})")
                        st.success("KNN Imputation applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            elif missing_method == "Iterative Imputer (MICE)":
                st.info("Advanced multivariate imputation using all other features.")
                if st.button("Apply Iterative Imputer", key="btn_iter"):
                    try:
                        st.session_state['processed_dataset'] = handle_missing_iterative(df_current)
                        st.session_state['applied_steps'].append("Iterative Imputer (MICE)")
                        st.success("Iterative Imputation applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # ============================================================
        # STEP 2 — TEXT CLEANING
        # ============================================================
        with st.expander("2️⃣ Text Cleaning", expanded=False):
            st.info("Lowercase, strip spaces, and remove special characters from text columns.")
            if st.button("Apply Text Cleaning", key="btn_tc"):
                st.session_state['processed_dataset'] = clean_text(df_current)
                st.session_state['applied_steps'].append("Cleaned Categorical Text")
                st.success("Text data cleaned!")
                st.rerun()

        # ============================================================
        # STEP 3 — CATEGORICAL ENCODING
        # ============================================================
        with st.expander("3️⃣ Categorical Encoding", expanded=False):
            st.info("Label Encoder for low-cardinality columns, One-Hot for the rest.")
            threshold = st.slider(
                "Unique Values Threshold for Label Encoding", 2, 20, 5,
                help="Columns with unique values ≤ threshold → Label Encoder. Others → One-Hot."
            )
            if st.button("Apply Categorical Encoding", key="btn_enc"):
                encoded_df, encoders = apply_label_encoding(df_current, threshold=threshold)
                final_enc_df = apply_one_hot(encoded_df, encoders.keys())
                st.session_state['processed_dataset'] = final_enc_df
                st.session_state['encoders_dict'] = encoders
                st.session_state['applied_steps'].append(f"Encoding: Label + One-Hot (threshold={threshold})")
                st.success("Encoding completed!")
                st.rerun()

        # ============================================================
        # STEP 4 — OUTLIER DETECTION & HANDLING
        # ============================================================
        with st.expander("4️⃣ Outlier Detection & Handling", expanded=False):
            outlier_method = st.radio(
                "Choose Outlier Method:",
                ["IQR (Remove Rows)", "Z-Score (Remove Rows)", "Winsorization (Cap Values)", "Clipping (Percentile Cap)"],
                key="rb_outlier"
            )

            if outlier_method == "IQR (Remove Rows)":
                st.info("Removes rows where any numeric value falls outside 1.5×IQR range.")
                if st.button("Remove Outliers (IQR)", key="btn_out"):
                    st.session_state['processed_dataset'] = remove_outliers_iqr(df_current)
                    st.session_state['applied_steps'].append("Outlier Removal: IQR")
                    st.success("IQR outlier removal applied!")
                    st.rerun()

            elif outlier_method == "Z-Score (Remove Rows)":
                st.info("Removes rows where any numeric value exceeds the Z-Score threshold.")
                zscore_thr = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.5, key="sl_zscore")
                if st.button("Remove Outliers (Z-Score)", key="btn_zscore"):
                    try:
                        st.session_state['processed_dataset'] = remove_outliers_zscore(df_current, threshold=zscore_thr)
                        st.session_state['applied_steps'].append(f"Outlier Removal: Z-Score (thr={zscore_thr})")
                        st.success("Z-Score outlier removal applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            elif outlier_method == "Winsorization (Cap Values)":
                st.info("Caps extreme values at the tail percentiles instead of removing rows.")
                wins_lim = st.slider("Winsorize Limit (each tail %)", 0.01, 0.20, 0.05, 0.01, key="sl_wins")
                if st.button("Apply Winsorization", key="btn_wins"):
                    try:
                        st.session_state['processed_dataset'] = remove_outliers_winsorize(df_current, limits=wins_lim)
                        st.session_state['applied_steps'].append(f"Outlier Handling: Winsorization (limit={wins_lim})")
                        st.success("Winsorization applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            elif outlier_method == "Clipping (Percentile Cap)":
                st.info("Clips values below and above specified percentiles.")
                clip_lower = st.slider("Lower Percentile", 0.01, 0.20, 0.05, 0.01, key="sl_clip_l")
                clip_upper = st.slider("Upper Percentile", 0.80, 0.99, 0.95, 0.01, key="sl_clip_u")
                if st.button("Apply Clipping", key="btn_clip"):
                    try:
                        st.session_state['processed_dataset'] = remove_outliers_clipping(df_current, lower_pct=clip_lower, upper_pct=clip_upper)
                        st.session_state['applied_steps'].append(f"Outlier Handling: Clipping ({clip_lower*100:.0f}%-{clip_upper*100:.0f}%)")
                        st.success("Clipping applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # ============================================================
        # STEP 5 — FEATURE TRANSFORMATION
        # ============================================================
        with st.expander("5️⃣ Feature Transformation", expanded=False):
            transform_method = st.radio(
                "Choose Transformation Method:",
                ["Log Transformation", "Box-Cox", "Power Transformation (Yeo-Johnson)", "Polynomial Features"],
                key="rb_transform"
            )

            if transform_method == "Log Transformation":
                st.info("Applies log1p to all positive numeric columns. Best for right-skewed data.")
                if st.button("Apply Log Transformation", key="btn_log"):
                    try:
                        st.session_state['processed_dataset'] = apply_log_transformation(df_current)
                        st.session_state['applied_steps'].append("Feature Transform: Log")
                        st.success("Log Transformation applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            elif transform_method == "Box-Cox":
                st.info("Box-Cox transformation. Requires all values to be strictly positive (> 0).")
                if st.button("Apply Box-Cox Transformation", key="btn_boxcox"):
                    try:
                        st.session_state['processed_dataset'] = apply_boxcox_transformation(df_current)
                        st.session_state['applied_steps'].append("Feature Transform: Box-Cox")
                        st.success("Box-Cox Transformation applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}. Ensure all numeric values are strictly positive.")

            elif transform_method == "Power Transformation (Yeo-Johnson)":
                st.info("Yeo-Johnson Power Transformation. Works with negative values too.")
                if st.button("Apply Power Transformation", key="btn_power"):
                    try:
                        st.session_state['processed_dataset'] = apply_power_transformation(df_current)
                        st.session_state['applied_steps'].append("Feature Transform: Power (Yeo-Johnson)")
                        st.success("Power Transformation applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            elif transform_method == "Polynomial Features":
                st.info("Generates interaction and polynomial features from numeric columns.")
                poly_degree = st.slider("Polynomial Degree", 2, 4, 2, key="sl_poly")
                if st.button("Apply Polynomial Features", key="btn_poly"):
                    try:
                        st.session_state['processed_dataset'] = apply_polynomial_features(df_current, degree=poly_degree)
                        st.session_state['applied_steps'].append(f"Feature Transform: Polynomial (degree={poly_degree})")
                        st.success(f"Polynomial Features (degree={poly_degree}) applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # ============================================================
        # STEP 6 — FEATURE SELECTION & DIMENSIONALITY REDUCTION
        # ============================================================
        with st.expander("6️⃣ Feature Selection & Dimensionality Reduction", expanded=False):
            selection_method = st.radio(
                "Choose Method:",
                ["Correlation Threshold", "RFE (Recursive Feature Elimination)", "PCA"],
                key="rb_selection"
            )

            if selection_method == "Correlation Threshold":
                st.info("Drops highly correlated numeric features above the threshold.")
                corr_threshold = st.slider("Correlation Threshold", 0.50, 0.99, 0.90, 0.05)
                if st.button("Apply Correlation Feature Selection", key="btn_fs"):
                    selected_df, dropped = feature_selection(df_current, threshold=corr_threshold)
                    st.session_state['processed_dataset'] = selected_df
                    st.session_state['applied_steps'].append(f"Feature Selection: Correlation (dropped {len(dropped)} cols)")
                    st.success(f"Applied! Dropped {len(dropped)} correlated columns.")
                    st.rerun()

            elif selection_method == "RFE (Recursive Feature Elimination)":
                st.info("Selects the most important features using a Logistic Regression estimator.")
                rfe_target = st.selectbox("Select Target Column", df_current.columns.tolist(), key="sb_rfe")
                rfe_n = st.slider("Number of Features to Keep", 1, min(20, df_current.shape[1] - 1), 5, key="sl_rfe")
                if st.button("Apply RFE", key="btn_rfe"):
                    try:
                        rfe_df, selected = apply_rfe(df_current, target_column=rfe_target, n_features=rfe_n)
                        st.session_state['processed_dataset'] = rfe_df
                        st.session_state['applied_steps'].append(f"Feature Selection: RFE ({rfe_n} features kept)")
                        st.success(f"RFE applied! Kept: {selected}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}. Make sure data is encoded and target is classification.")

            elif selection_method == "PCA":
                st.info("Reduces dimensions while retaining the selected explained variance ratio.")
                var_ratio = st.slider("Explained Variance Ratio to Retain", 0.50, 0.99, 0.95, 0.05)
                if st.button("Apply PCA", key="btn_pca"):
                    try:
                        pca_df, explained_var = apply_pca(df_current, variance_ratio=var_ratio)
                        st.session_state['processed_dataset'] = pca_df
                        st.session_state['applied_steps'].append(f"Dimensionality Reduction: PCA (var={explained_var:.2f})")
                        st.success(f"PCA applied! Explained Variance: {explained_var*100:.2f}%")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}. Ensure data is fully numeric/encoded before PCA.")

        # ============================================================
        # STEP 7 — FEATURE SCALING
        # ============================================================
        with st.expander("7️⃣ Feature Scaling", expanded=False):
            st.info("Apply after encoding and feature selection. Normalizes numeric column ranges.")
            scaler_choice = st.radio("Choose Scaling Method:", ["Standard Scaler", "MinMax Scaler"], key="rb_scaler")
            if st.button("Apply Scaling", key="btn_sc"):
                if scaler_choice == "Standard Scaler":
                    st.session_state['processed_dataset'] = apply_standard_scaler(df_current)
                else:
                    st.session_state['processed_dataset'] = apply_minmax_scaler(df_current)
                st.session_state['applied_steps'].append(f"Scaling: {scaler_choice}")
                st.success(f"Data scaled using {scaler_choice}!")
                st.rerun()

        # ============================================================
        # STEP 8 — HANDLE IMBALANCED DATA
        # ============================================================
        with st.expander("8️⃣ Handle Imbalanced Data", expanded=False):
            st.info("Balance target class distribution. Apply before training classification models.")
            imbalance_method = st.radio(
                "Choose Balancing Method:",
                ["SMOTE (Oversampling)", "Random Undersampling"],
                key="rb_imbalance"
            )
            imbalance_target = st.selectbox(
                "Select Target Column",
                df_current.columns.tolist(),
                key="sb_imbalance"
            )

            if imbalance_method == "SMOTE (Oversampling)":
                st.info("Generates synthetic minority class samples using SMOTE.")
                if st.button("Apply SMOTE", key="btn_smote"):
                    try:
                        balanced_df = handle_imbalance_smote(df_current, imbalance_target)
                        st.session_state['processed_dataset'] = balanced_df
                        st.session_state['applied_steps'].append(f"Imbalance: SMOTE on '{imbalance_target}'")
                        st.success(f"SMOTE applied! '{imbalance_target}' classes are now balanced.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}. Ensure data is fully encoded/numeric first.")

            elif imbalance_method == "Random Undersampling":
                st.info("Randomly removes majority class samples to balance classes.")
                if st.button("Apply Undersampling", key="btn_under"):
                    try:
                        balanced_df = handle_imbalance_undersample(df_current, imbalance_target)
                        st.session_state['processed_dataset'] = balanced_df
                        st.session_state['applied_steps'].append(f"Imbalance: Undersampling on '{imbalance_target}'")
                        st.success(f"Undersampling applied! '{imbalance_target}' classes are now balanced.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}. Ensure data is encoded/numeric first.")

    # ============================================================
    # RIGHT COLUMN — LIVE PREVIEW
    # ============================================================
    with col_preview:
        st.subheader("👁️ Live Data Preview")
        st.write(f"Current Shape: **{df_current.shape[0]}** rows, **{df_current.shape[1]}** columns")
        st.dataframe(df_current.head(10), use_container_width=True)

        st.subheader("📊 Remaining Missing Values")
        st.dataframe(pd.DataFrame(df_current.isna().sum(), columns=['Missing Count']), use_container_width=True)