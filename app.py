"""
Personal Spending Intelligence System
Phase 5 — Streamlit App

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings, calendar, random
from datetime import date, timedelta

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spending Intelligence",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Metric cards */
.metric-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 32px;
    font-weight: 600;
    font-family: 'DM Serif Display', serif;
    color: #f8fafc;
}
.metric-value.good  { color: #4ade80; }
.metric-value.warn  { color: #fb923c; }
.metric-value.bad   { color: #f87171; }
.metric-sub {
    font-size: 12px;
    color: #475569;
    margin-top: 4px;
}

/* Section header */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #0f172a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
    margin: 28px 0 16px;
}

/* Insight box */
.insight-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 14px;
    color: #1e3a5f;
    margin: 12px 0;
    line-height: 1.6;
}
.insight-box.warn {
    background: #fff7ed;
    border-left-color: #f97316;
    color: #7c2d12;
}
.insight-box.good {
    background: #f0fdf4;
    border-left-color: #22c55e;
    color: #14532d;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION  (Phase 1 logic — runs once and is cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_data(salary=65000):
    """Generate 12 months of synthetic spending data."""
    np.random.seed(42); random.seed(42)

    CATEGORIES = {
        "Rent":            {"avg":18000, "std":0,   "freq":1,  "vendors":["Housing Society","Landlord Transfer"],   "fixed_day":1},
        "Groceries":       {"avg":800,   "std":200, "freq":8,  "vendors":["BigBasket","Zepto","D-Mart","More"]},
        "Food & Dining":   {"avg":350,   "std":150, "freq":12, "vendors":["Zomato","Swiggy","Meghana Foods","CTR","Truffles"]},
        "Coffee & Snacks": {"avg":180,   "std":60,  "freq":14, "vendors":["Blue Tokai","Third Wave Coffee","Starbucks"]},
        "Transport":       {"avg":220,   "std":100, "freq":18, "vendors":["Uber","Ola","Rapido","Namma Metro"]},
        "Subscriptions":   {"avg":200,   "std":20,  "freq":4,  "vendors":["Netflix","Spotify","Amazon Prime"],       "fixed_day":1},
        "Shopping":        {"avg":1200,  "std":800, "freq":3,  "vendors":["Amazon","Flipkart","Myntra"]},
        "Health & Fitness":{"avg":600,   "std":300, "freq":3,  "vendors":["Cult.fit","PharmEasy","Apollo Pharmacy"]},
        "Utilities":       {"avg":900,   "std":150, "freq":2,  "vendors":["BESCOM","Airtel Broadband","Jio"]},
        "Entertainment":   {"avg":700,   "std":300, "freq":2,  "vendors":["BookMyShow","PVR","Steam"]},
    }

    def rand_date(year, month):
        last = calendar.monthrange(year, month)[1]
        return date(year, month, random.randint(1, last))

    def gen_amount(avg, std):
        a = np.random.normal(avg, std) if std > 0 else avg
        return round(max(a, avg * 0.2) / 10) * 10

    txns = []
    for month in range(1, 13):
        txns.append({"date": date(2024, month, 1), "description": "Salary Credit",
                     "amount": salary, "category": "Income", "type": "credit"})
        for cat, info in CATEGORIES.items():
            n = max(1, info["freq"] + random.randint(-1, 1))
            for _ in range(n):
                d = date(2024, month, 1) if info.get("fixed_day") == 1 else rand_date(2024, month)
                txns.append({"date": d, "description": random.choice(info["vendors"]),
                             "amount": gen_amount(info["avg"], info["std"]),
                             "category": cat, "type": "debit"})

    df = pd.DataFrame(txns)
    df['date']  = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    return df.sort_values('date').reset_index(drop=True)


@st.cache_data
def build_features(df):
    """Phase 3 feature engineering — cached so it only runs once."""
    expenses = df[df['type'] == 'debit'].copy()
    income   = df[df['type'] == 'credit'].copy()

    monthly_cat    = expenses.pivot_table(index='month', columns='category', values='amount', aggfunc='sum').fillna(0)
    monthly_income = income.groupby('month')['amount'].sum()

    mom_change         = (monthly_cat.pct_change() * 100).fillna(0).round(1)
    mom_change.columns = [f"{c}_mom_pct" for c in mom_change.columns]

    rolling_avg         = monthly_cat.rolling(window=3, min_periods=1).mean().round(0)
    rolling_avg.columns = [f"{c}_rolling3m" for c in rolling_avg.columns]

    cat_pct         = (monthly_cat.div(monthly_income, axis=0) * 100).round(2)
    cat_pct.columns = [f"{c}_pct_income" for c in cat_pct.columns]

    FIXED = ['Rent', 'Utilities', 'Subscriptions']
    DISC  = ['Food & Dining', 'Coffee & Snacks', 'Shopping', 'Entertainment', 'Health & Fitness', 'Transport']
    disc_ratio      = (monthly_cat[DISC].sum(axis=1) / monthly_cat[FIXED].sum(axis=1)).round(3)
    disc_ratio.name = 'disc_to_fixed_ratio'

    monthly_exp  = expenses.groupby('month')['amount'].sum()
    savings_rate = ((monthly_income - monthly_exp) / monthly_income * 100).round(2)
    savings_rate.name = 'savings_rate'

    feature_matrix = pd.concat([monthly_cat, mom_change, rolling_avg, cat_pct, disc_ratio, savings_rate], axis=1)
    feature_matrix = feature_matrix.iloc[1:].reset_index()

    X = feature_matrix.drop(columns=['savings_rate', 'month'])
    y = feature_matrix['savings_rate']
    return X, y, monthly_cat.iloc[1:], monthly_income.iloc[1:], savings_rate.iloc[1:]


@st.cache_resource
def train_models(salary):
    df      = generate_data(salary)
    X, y, monthly_cat, monthly_income, savings_rate = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    scaler      = StandardScaler()
    X_tr_sc     = scaler.fit_transform(X_train)
    X_te_sc     = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0).fit(X_tr_sc, y_train)
    rf    = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)

    return df, X, y, monthly_cat, monthly_income, savings_rate, ridge, rf, scaler, X_train, X_test


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    salary = st.slider("Monthly salary (₹)", 30000, 150000, 65000, step=5000,
                       help="Adjust to see how income changes your savings rate")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
This app is the end-to-end deliverable of the
**Personal Spending Intelligence System** project.

**Phases:**
1. Synthetic data generation
2. Exploratory data analysis
3. Feature engineering
4. ML models (Ridge + Random Forest)
5. This Streamlit app

Built to demonstrate applied ML on personal finance data.
    """)
    st.markdown("---")
    page = st.radio("Navigate", ["Dashboard", "EDA", "ML Models", "Raw Data"])


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

df, X, y, monthly_cat, monthly_income, savings_rate, ridge, rf, scaler, X_train, X_test = train_models(salary)
expenses = df[df['type'] == 'debit'].copy()
income_df = df[df['type'] == 'credit'].copy()

category_spend = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
total_spend    = expenses['amount'].sum()
total_income   = income_df['amount'].sum()
total_savings  = total_income - total_spend
avg_savings_rt = savings_rate.mean()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "Dashboard":
    st.markdown("# 💸 Personal Spending Intelligence")
    st.markdown("*12-month financial overview — 2024*")

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    rate_class = "good" if avg_savings_rt >= 20 else "warn" if avg_savings_rt >= 10 else "bad"

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Income</div>
            <div class="metric-value">₹{total_income/1e5:.2f}L</div>
            <div class="metric-sub">across 12 months</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Spent</div>
            <div class="metric-value warn">₹{total_spend/1e5:.2f}L</div>
            <div class="metric-sub">across 12 months</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Net Saved</div>
            <div class="metric-value good">₹{total_savings/1e3:.1f}K</div>
            <div class="metric-sub">income minus expenses</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Savings Rate</div>
            <div class="metric-value {rate_class}">{avg_savings_rt:.1f}%</div>
            <div class="metric-sub">target: 20%+</div>
        </div>""", unsafe_allow_html=True)

    # ── Insight ───────────────────────────────────────────────────────────────
    top_cat = category_spend.index[0]
    top_pct = category_spend.iloc[0] / total_spend * 100
    box_cls = "good" if avg_savings_rt >= 20 else "warn"
    st.markdown(f"""<div class="insight-box {box_cls}">
        <strong>Key insight:</strong> Your biggest cost driver is <strong>{top_cat}</strong>
        at {top_pct:.1f}% of total spending. Average monthly savings rate is
        <strong>{avg_savings_rt:.1f}%</strong>
        {'— healthy! Keep it above 20%.' if avg_savings_rt >= 20 else '— below the 20% target. Discretionary cuts would help most.'}
    </div>""", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown('<div class="section-header">Monthly: Income vs Expenses</div>', unsafe_allow_html=True)
        months_str     = [str(m) for m in savings_rate.index]
        monthly_exp_s  = expenses.groupby('month')['amount'].sum().iloc[1:]
        monthly_inc_s  = income_df.groupby('month')['amount'].sum().iloc[1:]

        fig, ax = plt.subplots(figsize=(9, 3.8))
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#f8fafc')
        ax.plot(months_str, monthly_inc_s.values, 'o-', color='#22c55e', lw=2, ms=6, label='Income')
        ax.plot(months_str, monthly_exp_s.values, 'o-', color='#ef4444', lw=2, ms=6, label='Expenses')
        ax.fill_between(months_str, monthly_exp_s.values, monthly_inc_s.values, alpha=0.1, color='#22c55e')
        ax.set_xlabel('Month', fontsize=10)
        ax.set_ylabel('Amount (₹)', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">Spend by Category</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
        fig2.patch.set_facecolor('#f8fafc')
        ax2.set_facecolor('#f8fafc')
        colors_pie = sns.color_palette('Set2', len(category_spend))
        wedges, texts, autotexts = ax2.pie(
            category_spend.values, labels=None,
            autopct='%1.0f%%', startangle=90,
            colors=colors_pie, pctdistance=0.82,
            wedgeprops=dict(width=0.55)   # donut style
        )
        for at in autotexts: at.set_fontsize(8)
        ax2.legend(wedges, category_spend.index, loc='center left',
                   bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Savings rate bar ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Monthly Savings Rate</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(11, 3))
    fig3.patch.set_facecolor('#f8fafc')
    ax3.set_facecolor('#f8fafc')
    bar_colors = ['#22c55e' if v >= 20 else '#f97316' if v >= 10 else '#ef4444'
                  for v in savings_rate.values]
    bars = ax3.bar(months_str, savings_rate.values, color=bar_colors, width=0.6)
    ax3.axhline(y=20, color='#3b82f6', linestyle='--', alpha=0.7, lw=1.5, label='20% target')
    for bar, val in zip(bars, savings_rate.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=8)
    ax3.set_ylabel('Savings rate (%)')
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.spines[['top','right']].set_visible(False)
    ax3.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════

elif page == "EDA":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("*The 5 EDA questions answered visually*")

    # Q1: Distribution
    st.markdown('<div class="section-header">Q1 — How is spending distributed?</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
    colors = sns.color_palette('Blues_r', len(category_spend))
    bars = ax.bar(category_spend.index, category_spend.values, color=colors)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                f'₹{bar.get_height()/1000:.0f}K', ha='center', fontsize=8)
    ax.set_ylabel('Total spend (₹)'); ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1000:.0f}K'))
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Q2: Summary stats
    st.markdown('<div class="section-header">Q2 — Summary statistics per category</div>', unsafe_allow_html=True)
    stats = expenses.groupby('category')['amount'].describe().round(0).astype(int)
    st.dataframe(stats, use_container_width=True)

    # Q3: Heatmap
    st.markdown('<div class="section-header">Q3 — When do I spend on what?</div>', unsafe_allow_html=True)
    pivot = expenses.pivot_table(index='category', columns='month', values='amount', aggfunc='sum').fillna(0)
    pivot.columns = [str(m) for m in pivot.columns]
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.patch.set_facecolor('#f8fafc')
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f',
                linewidths=0.4, ax=ax2, cbar_kws={'label': '₹'})
    ax2.set_title('Category × Month spending heatmap', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Q4: Outliers
    st.markdown('<div class="section-header">Q4 — Outliers (IQR method)</div>', unsafe_allow_html=True)
    Q1 = expenses['amount'].quantile(0.25)
    Q3 = expenses['amount'].quantile(0.75)
    IQR = Q3 - Q1
    fence = Q3 + 1.5 * IQR
    outliers = expenses[expenses['amount'] > fence]
    st.markdown(f"""<div class="insight-box">
        Outlier fence: <strong>₹{fence:,.0f}</strong> (Q3 + 1.5×IQR).
        Found <strong>{len(outliers)}</strong> transactions above this threshold.
    </div>""", unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(11, 4))
    fig3.patch.set_facecolor('#f8fafc'); ax3.set_facecolor('#f8fafc')
    no_rent = expenses[expenses['category'] != 'Rent']
    order = no_rent.groupby('category')['amount'].median().sort_values(ascending=False).index
    sns.boxplot(data=no_rent, x='category', y='amount', palette='Set2', order=order, ax=ax3)
    ax3.set_title('Transaction distribution per category (excl. Rent)', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, labelsize=9)
    ax3.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # Q5: Savings rate
    st.markdown('<div class="section-header">Q5 — Savings rate month by month</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({'Month': [str(m) for m in savings_rate.index],
                      'Savings Rate (%)': savings_rate.values}).set_index('Month'),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ML Models":
    st.markdown("# 🤖 Machine Learning Models")
    st.markdown("*Predicting monthly savings rate from spending features*")

    from sklearn.metrics import mean_absolute_error, r2_score

    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    ridge_preds = ridge.predict(X_test_sc)
    rf_preds    = rf.predict(X_test)
    y_test      = y.iloc[len(X_train):]

    col1, col2 = st.columns(2)

    for col, name, preds in [(col1, "Ridge Regression", ridge_preds),
                              (col2, "Random Forest",    rf_preds)]:
        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        r2c = "good" if r2 >= 0.7 else "warn" if r2 >= 0.4 else "bad"
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value {r2c}">R² = {r2:.2f}</div>
                <div class="metric-sub">MAE = {mae:.2f} percentage points</div>
            </div>""", unsafe_allow_html=True)

    # Predictions chart
    st.markdown('<div class="section-header">Predicted vs Actual savings rate</div>', unsafe_allow_html=True)
    test_x = list(range(1, len(y_test)+1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#f8fafc')
    for ax, name, preds, col in zip(axes,
        ["Ridge Regression", "Random Forest"],
        [ridge_preds, rf_preds],
        ['#3b82f6', '#22c55e']):
        ax.set_facecolor('#f8fafc')
        ax.plot(test_x, y_test.values, 'o-', color='#0f172a', lw=2, ms=7, label='Actual')
        ax.plot(test_x, preds, 's--', color=col, lw=2, ms=7, label='Predicted')
        ax.fill_between(test_x, y_test.values, preds, alpha=0.1, color=col)
        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        ax.set_title(f'{name}\nMAE={mae:.2f}  R²={r2:.2f}', fontweight='bold', fontsize=10)
        ax.set_xlabel('Test month'); ax.set_ylabel('Savings rate (%)')
        ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top12 = importances.sort_values(ascending=False).head(12)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.patch.set_facecolor('#f8fafc'); ax2.set_facecolor('#f8fafc')
    ax2.barh(range(len(top12)), top12.values,
             color=['#3b82f6' if 'pct' in f else '#64748b' for f in top12.index])
    ax2.set_yticks(range(len(top12)))
    ax2.set_yticklabels(top12.index, fontsize=9)
    ax2.set_xlabel('Importance score')
    ax2.set_title('Top 12 features driving savings rate predictions', fontweight='bold')
    ax2.invert_yaxis(); ax2.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("""<div class="insight-box">
        <strong>How to read feature importance:</strong> Each bar shows how much that feature
        reduces prediction error across all 100 Random Forest trees.
        Features highlighted in blue contain <em>percentage of income</em> — confirming that
        normalised spend (relative to income) predicts savings rate better than raw amounts.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Raw Data":
    st.markdown("# 📋 Raw Transaction Data")
    st.markdown(f"*{len(df)} transactions — Jan 2024 to Dec 2024*")

    cat_filter = st.multiselect("Filter by category",
                                 options=sorted(df['category'].unique()),
                                 default=sorted(df['category'].unique()))
    filtered = df[df['category'].isin(cat_filter)]

    st.dataframe(
        filtered[['date','description','category','amount','type']].sort_values('date'),
        use_container_width=True, height=500
    )
    st.caption(f"Showing {len(filtered)} of {len(df)} transactions")
