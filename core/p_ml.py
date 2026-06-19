import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler ,LabelEncoder
from sklearn.decomposition import PCA 

#  Get Numeric Columns
def get_numeric_columns(df):
    return df.select_dtypes(include=['float64', 'int64']).columns


# 1. HANDLE MISSING VALUES

def handle_missing_values(df):
    df = df.copy()

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # numeric → mean
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    # categorical → mode
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df



# 2. TEXT CLEANING

def clean_text(df):
    df = df.copy()

    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.lower().str.strip()
        df[col] = df[col].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)

    return df



# 3. LABEL ENCODING

def apply_label_encoding(df, threshold=5):
    df = df.copy()
    label_encoders = {}

    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        if df[col].nunique() <= threshold:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    return df, label_encoders



# 4. ONE HOT ENCODING

def apply_one_hot(df, label_encoded_cols):
    df = df.copy()

    cat_cols = df.select_dtypes(include=['object']).columns

    # الأعمدة اللي لسه ما تعملهاش label encoding
    remaining_cols = [col for col in cat_cols if col not in label_encoded_cols]

    df = pd.get_dummies(df, columns=remaining_cols , drop_first=True)

    return df

# 5. IQR Cleaning 
def remove_outliers_iqr(df):
    numeric_cols = get_numeric_columns(df)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


# 6. Feature Selection 
def feature_selection(df, threshold=0.90):
    numeric_cols = get_numeric_columns(df)

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df = df.drop(columns=to_drop)

    return df, to_drop


# 7.Scaling
def apply_standard_scaler(df):
    numeric_cols = get_numeric_columns(df)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df



def apply_minmax_scaler(df):
    numeric_cols = get_numeric_columns(df)
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# 8.PCA 
def apply_pca(df, variance_ratio=0.95):
    numeric_cols = get_numeric_columns(df)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[numeric_cols])

    pca = PCA(n_components=variance_ratio)
    pca_data = pca.fit_transform(data_scaled)

    df_pca = pd.DataFrame(pca_data)

    return df_pca, pca.explained_variance_ratio_.sum()


# 10. Handle Imbalanced Data using SMOTE (إضافة ميزة موازنة البيانات)
def handle_imbalance_smote(df, target_column):
    from imblearn.over_sampling import SMOTE
    df = df.copy()
    
    # الـ SMOTE بيشتغل على الداتا الرقمية فقط، هنطرد أي نصوص باقية بالأمان
    X = df.drop(columns=[target_column]).select_dtypes(exclude=['object'])
    y = df[target_column]
    
    # تطبيق الألجوريزم لموازنة الفئات
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # إعادة تجميع الداتا المتزنة في DataFrame واحد جديد
    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_column] = y_resampled
    
    return df_balanced


# 9. full preprocessing

def full_preprocessing(df):
    print("Starting preprocessing...")

    df = handle_missing_values(df)
    df = clean_text(df)

    df, label_encoders = apply_label_encoding(df)

    df = apply_one_hot(df, label_encoders.keys())

    df = df.drop_duplicates()

    df =remove_outliers_iqr(df)

    df , dropped_cols = feature_selection(df)

    df = apply_standard_scaler(df)

    df_pca , variance = apply_pca(df)

    print("Preprocessing done!")

    return{
    "final data " : df,

     "encoders " :label_encoders,
     "pca_data " : df_pca,
     "variance " :variance,
     "dropped_columns ": dropped_cols
    }


# ============================================================
# ================= الإضافات الجديدة فقط ====================
# ============================================================

# A. KNN Imputer
def handle_missing_knn(df, n_neighbors=5):
    from sklearn.impute import KNNImputer
    df = df.copy()
    numeric_cols = get_numeric_columns(df)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


# B. Iterative Imputer
def handle_missing_iterative(df):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    df = df.copy()
    numeric_cols = get_numeric_columns(df)
    imputer = IterativeImputer(random_state=42, max_iter=10)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


# C. Z-Score Outlier Removal
def remove_outliers_zscore(df, threshold=3.0):
    from scipy import stats
    df = df.copy()
    numeric_cols = get_numeric_columns(df)
    z_scores = np.abs(stats.zscore(df[numeric_cols].dropna()))
    mask = (z_scores < threshold).all(axis=1)
    df = df[mask]
    return df


# D. Winsorization
def remove_outliers_winsorize(df, limits=0.05):
    from scipy.stats.mstats import winsorize
    df = df.copy()
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        df[col] = winsorize(df[col], limits=[limits, limits])
    return df


# E. Clipping
def remove_outliers_clipping(df, lower_pct=0.05, upper_pct=0.95):
    df = df.copy()
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        lower = df[col].quantile(lower_pct)
        upper = df[col].quantile(upper_pct)
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# F. Log Transformation
def apply_log_transformation(df):
    df = df.copy()
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        if (df[col] > 0).all():
            df[col] = np.log1p(df[col])
    return df


# G. Box-Cox Transformation
def apply_boxcox_transformation(df):
    from scipy.stats import boxcox
    df = df.copy()
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        if (df[col] > 0).all():
            df[col], _ = boxcox(df[col])
    return df


# H. Power Transformation (Yeo-Johnson)
def apply_power_transformation(df):
    from sklearn.preprocessing import PowerTransformer
    df = df.copy()
    numeric_cols = get_numeric_columns(df)
    pt = PowerTransformer(method='yeo-johnson')
    df[numeric_cols] = pt.fit_transform(df[numeric_cols])
    return df


# I. Polynomial Features
def apply_polynomial_features(df, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    df = df.copy()
    numeric_cols = get_numeric_columns(df)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_data = poly.fit_transform(df[numeric_cols])
    poly_cols = poly.get_feature_names_out(numeric_cols)
    df_poly = pd.DataFrame(poly_data, columns=poly_cols, index=df.index)
    non_numeric = df.select_dtypes(exclude=['float64', 'int64'])
    return pd.concat([non_numeric.reset_index(drop=True), df_poly.reset_index(drop=True)], axis=1)


# J. RFE - Recursive Feature Elimination
def apply_rfe(df, target_column, n_features=5):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    df = df.copy()
    numeric_cols = [c for c in get_numeric_columns(df) if c != target_column]
    X = df[numeric_cols].dropna()
    y = df[target_column].loc[X.index]
    estimator = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected = [col for col, sup in zip(numeric_cols, rfe.support_) if sup]
    return df[selected + [target_column]], selected


# K. Undersampling (Random)
def handle_imbalance_undersample(df, target_column):
    from imblearn.under_sampling import RandomUnderSampler
    df = df.copy()
    X = df.drop(columns=[target_column]).select_dtypes(exclude=['object'])
    y = df[target_column]
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_column] = y_resampled
    return df_balanced