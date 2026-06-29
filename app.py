import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="🚚",
    layout="wide"
)

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:700;color:#2c3e50;text-align:center;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('supply_chain_processed.csv')
    fraud = pd.read_csv('fraud_results.csv')
    # Parse dates properly
    if 'order date (DateOrders)' in df.columns:
        df['order date (DateOrders)'] = pd.to_datetime(
            df['order date (DateOrders)'], errors='coerce')
        df['order_year'] = df['order date (DateOrders)'].dt.year
        df['order_month'] = df['order date (DateOrders)'].dt.month
        df['order_dayofweek'] = df['order date (DateOrders)'].dt.dayofweek
    # delivery delay
    if 'Days for shipping (real)' in df.columns:
        df['delivery_delay'] = (
            df['Days for shipping (real)'] - 
            df['Days for shipment (scheduled)']
        )
    return df, fraud

@st.cache_resource
def load_models():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

df, fraud_df = load_data()
best_model = load_models()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/delivery.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Executive Overview",
    "🚚 Delivery Performance",
    "💰 Profitability Analysis",
    "🔍 Fraud Detection",
    "🤖 Late Delivery Predictor"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** DataCo Global Supply Chain")
st.sidebar.markdown(f"**Records:** {len(df):,} orders")
st.sidebar.markdown("**Period:** 2015–2018")

# ══════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════
if page == "🏠 Executive Overview":
    st.markdown('<p class="main-header">🚚 Supply Chain Analytics Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown("### DataCo Global — 110K+ Orders Analyzed")
    st.markdown("---")

    total_orders = len(df)
    total_sales = df['Sales'].sum()
    total_profit = df['Order Profit Per Order'].sum()
    late_rate = df['Late_delivery_risk'].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{total_orders:,}")
    c2.metric("Total Sales", f"${total_sales:,.0f}")
    c3.metric("Total Profit", f"${total_profit:,.0f}")
    c4.metric("Late Delivery Risk", f"{late_rate:.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        sales_market = df.groupby('Market')['Sales'].sum().reset_index()
        fig = px.bar(
            sales_market.sort_values('Sales', ascending=False),
            x='Market', y='Sales', color='Market',
            title='Total Sales by Market', text='Sales'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.pie(
            df, names='Delivery Status',
            title='Delivery Status Distribution',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly trend — safely
    try:
        monthly = df.dropna(subset=['order_year', 'order_month'])
        monthly = monthly[
            (monthly['order_year'] >= 2015) &
            (monthly['order_month'] >= 1) &
            (monthly['order_month'] <= 12)
        ]
        monthly = monthly.groupby(
            ['order_year', 'order_month']
        ).size().reset_index(name='orders')
        monthly['date'] = pd.to_datetime(
            monthly['order_year'].astype(int).astype(str) + '-' +
            monthly['order_month'].astype(int).astype(str).str.zfill(2) + '-01'
        )
        monthly = monthly.sort_values('date')
        fig3 = px.line(
            monthly, x='date', y='orders',
            title='Monthly Order Volume Trend', markers=True
        )
        fig3.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.warning(f"Trend chart unavailable: {e}")

# ══════════════════════════════════════════════════
# PAGE 2 — DELIVERY PERFORMANCE
# ══════════════════════════════════════════════════
elif page == "🚚 Delivery Performance":
    st.title("🚚 Delivery Performance Analysis")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        late_ship = df.groupby('Shipping Mode')['Late_delivery_risk'].mean().reset_index()
        late_ship['Late %'] = (late_ship['Late_delivery_risk'] * 100).round(1)
        fig = px.bar(
            late_ship, x='Shipping Mode', y='Late %',
            color='Shipping Mode',
            title='Late Delivery Risk % by Shipping Mode',
            text=late_ship['Late %'].astype(str) + '%'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        late_region = df.groupby('Order Region')['Late_delivery_risk'].mean().reset_index()
        late_region['Late %'] = (late_region['Late_delivery_risk'] * 100).round(1)
        late_region = late_region.sort_values('Late %', ascending=True)
        fig2 = px.bar(
            late_region, x='Late %', y='Order Region',
            orientation='h',
            title='Late Delivery Risk % by Region',
            color='Late %',
            color_continuous_scale='RdYlGn_r'
        )
        fig2.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

    if 'delivery_delay' in df.columns:
        fig3 = px.histogram(
            df, x='delivery_delay',
            title='Delivery Delay Distribution',
            color_discrete_sequence=['#3498db'],
            nbins=30
        )
        fig3.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig3, use_container_width=True)

    on_time_rate = (1 - df['Late_delivery_risk'].mean()) * 100
    st.info(f"**Key Insight:** On-Time Delivery Rate is **{on_time_rate:.1f}%** — "
            f"Industry benchmark is 95%+!")

# ══════════════════════════════════════════════════
# PAGE 3 — PROFITABILITY ANALYSIS
# ══════════════════════════════════════════════════
elif page == "💰 Profitability Analysis":
    st.title("💰 Profitability Analysis")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        cat_profit = df.groupby('Category Name')['Order Profit Per Order'] \
                       .mean().reset_index()
        cat_profit = cat_profit.sort_values(
            'Order Profit Per Order', ascending=False).head(10)
        fig = px.bar(
            cat_profit, x='Order Profit Per Order', y='Category Name',
            orientation='h',
            title='Top 10 Categories by Avg Profit',
            color='Order Profit Per Order',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(plot_bgcolor='white',
                          yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        seg = df.groupby('Customer Segment').agg({
            'Sales': 'sum',
            'Order Profit Per Order': 'mean'
        }).reset_index()
        fig2 = px.bar(
            seg, x='Customer Segment', y='Sales',
            color='Customer Segment',
            title='Total Sales by Customer Segment',
            text='Sales'
        )
        fig2.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig2.update_layout(showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

    dept = df.groupby('Department Name').agg({
        'Sales': 'sum',
        'Order Profit Per Order': 'mean',
        'Order Item Quantity': 'sum'
    }).reset_index()
    dept.columns = ['Department', 'Total Sales', 'Avg Profit', 'Total Quantity']
    dept = dept.sort_values('Total Sales', ascending=False)
    st.subheader("Department-wise Performance")
    st.dataframe(dept.style.format({
        'Total Sales': '${:,.0f}',
        'Avg Profit': '${:,.2f}',
        'Total Quantity': '{:,.0f}'
    }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════
# PAGE 4 — FRAUD DETECTION
# ══════════════════════════════════════════════════
elif page == "🔍 Fraud Detection":
    st.title("🔍 Fraud Detection Analysis")
    st.markdown("---")

    if 'fraud_flag' in fraud_df.columns:
        fraud_count = fraud_df['fraud_flag'].value_counts()
        suspicious = fraud_count.get('Suspicious', 0)
        fraud_rate = suspicious / len(fraud_df) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Orders Analyzed", f"{len(fraud_df):,}")
        c2.metric("Suspicious Orders", f"{suspicious:,}")
        c3.metric("Fraud Rate", f"{fraud_rate:.1f}%")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            fig = px.pie(
                values=fraud_count.values,
                names=fraud_count.index,
                title='Normal vs Suspicious Orders',
                color=fraud_count.index,
                color_discrete_map={
                    'Normal': '#2ecc71',
                    'Suspicious': '#e74c3c'
                },
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.scatter(
                fraud_df.sample(min(5000, len(fraud_df))),
                x='Sales', y='Order Item Profit Ratio',
                color='fraud_flag',
                color_discrete_map={
                    'Normal': '#2ecc71',
                    'Suspicious': '#e74c3c'
                },
                title='Fraud Pattern — Sales vs Profit Ratio',
                opacity=0.6
            )
            fig2.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig2, use_container_width=True)

        if 'Category Name' in fraud_df.columns:
            fraud_cat = fraud_df[fraud_df['fraud_flag'] == 'Suspicious'][
                'Category Name'].value_counts().head(10).reset_index()
            fraud_cat.columns = ['Category', 'Count']
            fig3 = px.bar(
                fraud_cat, x='Count', y='Category',
                orientation='h',
                title='Top 10 Categories with Suspicious Orders',
                color='Count',
                color_continuous_scale='Reds'
            )
            fig3.update_layout(plot_bgcolor='white',
                               yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig3, use_container_width=True)

        st.warning(f"**Alert:** {suspicious:,} suspicious orders detected "
                   f"({fraud_rate:.1f}% of total)!")
    else:
        st.error("fraud_flag column not found in fraud_results.csv!")

# ══════════════════════════════════════════════════
# PAGE 5 — LATE DELIVERY PREDICTOR
# ══════════════════════════════════════════════════
elif page == "🤖 Late Delivery Predictor":
    st.title("🤖 Late Delivery Risk Predictor")
    st.markdown("Predict whether an order will be delivered late!")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        shipping_mode = st.selectbox("Shipping Mode",
            ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
        customer_segment = st.selectbox("Customer Segment",
            ['Consumer', 'Corporate', 'Home Office'])
        market = st.selectbox("Market",
            ['Europe', 'LATAM', 'Pacific Asia', 'USCA', 'Africa'])
        order_region = st.selectbox("Order Region",
            ['Western Europe', 'Central America', 'Oceania',
             'Eastern Asia', 'South America', 'Eastern Europe',
             'West Africa', 'South Asia', 'North America'])

    with c2:
        scheduled_days = st.slider("Scheduled Shipping Days", 1, 7, 4)
        quantity = st.slider("Order Quantity", 1, 10, 2)
        discount_rate = st.slider("Discount Rate", 0.0, 1.0, 0.1)
        profit_ratio = st.slider("Profit Ratio", -1.0, 1.0, 0.3)
        category = st.selectbox("Product Category",
            sorted(df['Category Name'].dropna().unique().tolist()))
        department = st.selectbox("Department",
            sorted(df['Department Name'].dropna().unique().tolist()))

    if st.button("🔍 Predict Delivery Risk", type="primary"):
        from sklearn.preprocessing import LabelEncoder

        le_map = {}
        cat_cols = ['Type', 'Shipping Mode', 'Customer Segment',
                    'Market', 'Order Region', 'Category Name', 'Department Name']

        ml_df = df.copy()
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(ml_df[col].astype(str))
            le_map[col] = le

        def safe_encode(encoder, value):
            try:
                return encoder.transform([str(value)])[0]
            except Exception:
                return 0

        input_data = pd.DataFrame({
            'Days for shipment (scheduled)': [scheduled_days],
            'Shipping Mode': [safe_encode(le_map['Shipping Mode'], shipping_mode)],
            'Customer Segment': [safe_encode(le_map['Customer Segment'], customer_segment)],
            'Market': [safe_encode(le_map['Market'], market)],
            'Order Region': [safe_encode(le_map['Order Region'], order_region)],
            'Category Name': [safe_encode(le_map['Category Name'], category)],
            'Order Item Quantity': [quantity],
            'Order Item Discount Rate': [discount_rate],
            'Order Item Profit Ratio': [profit_ratio],
            'delivery_delay': [0],
            'order_month': [6],
            'order_dayofweek': [1],
            'Type': [safe_encode(le_map['Type'], 'DEBIT')]
        })

        prediction = best_model.predict(input_data)[0]
        proba = best_model.predict_proba(input_data)[0]

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        if prediction == 1:
            c1.error("⚠️ HIGH LATE DELIVERY RISK!")
            c2.metric("Risk Probability", f"{proba[1]*100:.1f}%")
            c3.metric("Model Confidence", f"{max(proba)*100:.1f}%")
            st.error("**Recommendation:** Consider upgrading shipping mode!")
        else:
            c1.success("✅ LOW LATE DELIVERY RISK!")
            c2.metric("On-Time Probability", f"{proba[0]*100:.1f}%")
            c3.metric("Model Confidence", f"{max(proba)*100:.1f}%")
            st.success("**Recommendation:** Order likely to be delivered on time!")

        fig = px.bar(
            x=['On Time', 'Late Delivery'],
            y=[proba[0]*100, proba[1]*100],
            color=['On Time', 'Late Delivery'],
            color_discrete_map={
                'On Time': '#2ecc71',
                'Late Delivery': '#e74c3c'
            },
            title='Prediction Probability',
            text=[f'{proba[0]*100:.1f}%', f'{proba[1]*100:.1f}%']
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            yaxis_range=[0, 110]
        )
        st.plotly_chart(fig, use_container_width=True)