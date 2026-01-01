import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# DATA GENERATION
# ============================================================
def generate_sample_data(n_samples=5000):
    data = {
        'order_id': [f'ORD{i:05d}' for i in range(n_samples)],
        'customer_id': np.random.randint(1000, 5000, n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'], n_samples),
        'product_price': np.random.uniform(10, 500, n_samples),
        'discount_percent': np.random.choice([0,5,10,15,20,25,30], n_samples),
        'payment_method': np.random.choice(['Credit Card','Debit Card','COD','UPI'], n_samples),
        'order_day': np.random.choice(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], n_samples),
        'order_hour': np.random.randint(0, 24, n_samples),
        'shipping_days': np.random.randint(1, 15, n_samples),
        'customer_reviews': np.random.randint(0, 50, n_samples),
        'customer_rating': np.random.uniform(1, 5, n_samples),
        'previous_returns': np.random.randint(0, 10, n_samples),
        'account_age_days': np.random.randint(30, 1500, n_samples)
    }

    df = pd.DataFrame(data)

    return_prob = (
        (df['payment_method'] == 'COD') * 0.3 +
        (df['discount_percent'] > 20) * 0.2 +
        (df['previous_returns'] > 3) * 0.3 +
        (df['shipping_days'] > 10) * 0.15 +
        (df['customer_rating'] < 3) * 0.2 +
        np.random.uniform(0, 0.2, n_samples)
    )

    df['returned'] = (return_prob > 0.5).astype(int)
    return df

# ============================================================
# EDA
# ============================================================
def perform_eda(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    df.groupby('product_category')['returned'].mean().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title("Return Rate by Category")

    df.groupby('payment_method')['returned'].mean().plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title("Return Rate by Payment")

    df.boxplot(column='product_price', by='returned', ax=axes[0,2])

    df.groupby('previous_returns')['returned'].mean().plot(ax=axes[1,0])
    axes[1,0].set_title("Previous Returns vs Return Rate")

    df.groupby('shipping_days')['returned'].mean().plot(ax=axes[1,1])
    axes[1,1].set_title("Shipping Days vs Return Rate")

    sns.heatmap(df.select_dtypes(include=np.number).corr(), ax=axes[1,2], cmap="coolwarm")

    plt.tight_layout()
    plt.savefig("eda_analysis.png", dpi=300)
    plt.close()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def feature_engineering(df):
    df['is_expensive'] = (df['product_price'] > df['product_price'].quantile(0.75)).astype(int)
    df['price_per_discount'] = df['product_price'] / (df['discount_percent'] + 1)
    df['return_rate'] = df['previous_returns'] / (df['account_age_days']/30 + 1)
    df['is_frequent_returner'] = (df['previous_returns'] > 2).astype(int)
    df['customer_engagement'] = df['customer_reviews'] * df['customer_rating']
    df['is_weekend'] = df['order_day'].isin(['Saturday','Sunday']).astype(int)
    df['is_peak_hour'] = df['order_hour'].isin([12,13,18,19,20]).astype(int)
    df['is_delayed_shipping'] = (df['shipping_days'] > 7).astype(int)
    df['high_discount'] = (df['discount_percent'] > 20).astype(int)
    df['is_cod'] = (df['payment_method'] == 'COD').astype(int)
    return df

# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_data(df):
    le_cat = LabelEncoder()
    le_pay = LabelEncoder()
    le_day = LabelEncoder()

    df['product_category_encoded'] = le_cat.fit_transform(df['product_category'])
    df['payment_method_encoded'] = le_pay.fit_transform(df['payment_method'])
    df['order_day_encoded'] = le_day.fit_transform(df['order_day'])

    features = [
        'product_price','discount_percent','order_hour','shipping_days',
        'customer_reviews','customer_rating','previous_returns','account_age_days',
        'product_category_encoded','payment_method_encoded','order_day_encoded',
        'is_expensive','price_per_discount','return_rate','is_frequent_returner',
        'customer_engagement','is_weekend','is_peak_hour','is_delayed_shipping',
        'high_discount','is_cod'
    ]

    X = df[features]
    y = df['returned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, features, le_cat, le_pay, le_day

# ============================================================
# TRAIN MODELS
# ============================================================
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    best_model = None
    best_auc = 0

    plt.figure(figsize=(8,6))

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)

        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    plt.plot([0,1],[0,1],'k--')
    plt.legend()
    plt.savefig("roc_curves.png", dpi=300)
    plt.close()

    return best_model

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = generate_sample_data()
    perform_eda(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test, scaler, features, le_cat, le_pay, le_day = preprocess_data(df)
    best_model = train_models(X_train, X_test, y_train, y_test)

    with open("return_risk_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": scaler,
            "features": features,
            "encoders": {
                "category": le_cat,
                "payment": le_pay,
                "day": le_day
            }
        }, f)

    print("✅ Model training complete & artifacts saved")
