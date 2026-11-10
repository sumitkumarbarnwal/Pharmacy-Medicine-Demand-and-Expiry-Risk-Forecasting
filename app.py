import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Pharmacy Demand and Expiry Forecasting",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 12px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .status-badge-risk {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #F87171;
        font-size: 14px;
        line-height: 1.6;
    }
    .status-badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #4ADE80;
        font-size: 14px;
        line-height: 1.6;
    }
    .notice-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        color: #1E3A8A;
        font-size: 13px;
        margin-bottom: 16px;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400E;
        font-size: 13px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_saved_models():
    demand_path = "models/demand_model.pkl" if os.path.exists("models/demand_model.pkl") else "Pharmacy_Medicine_Forecasting/models/demand_model.pkl"
    expiry_path = "models/expiry_risk_model.pkl" if os.path.exists("models/expiry_risk_model.pkl") else "Pharmacy_Medicine_Forecasting/models/expiry_risk_model.pkl"
    demand_artifact = joblib.load(demand_path) if os.path.exists(demand_path) else None
    expiry_artifact = joblib.load(expiry_path) if os.path.exists(expiry_path) else None
    return demand_artifact, expiry_artifact

@st.cache_data
def load_prepared_data():
    s2_path = "data/S2_Dhis2Data.csv" if os.path.exists("data/S2_Dhis2Data.csv") else ("Pharmacy_Medicine_Forecasting/data/S2_Dhis2Data.csv" if os.path.exists("Pharmacy_Medicine_Forecasting/data/S2_Dhis2Data.csv") else "S2.csv/S2_Dhis2Data.csv")
    s3_path = "data/S3.csv" if os.path.exists("data/S3.csv") else ("Pharmacy_Medicine_Forecasting/data/S3.csv" if os.path.exists("Pharmacy_Medicine_Forecasting/data/S3.csv") else "S3.csv")
    
    df_s2 = pd.read_csv(s2_path)
    df_s2['date'] = pd.to_datetime(df_s2['date'], format='%m/%d/%y')
    
    df_s3 = pd.read_csv(s3_path).drop_duplicates().reset_index(drop=True)
    
    prod_monthly = df_s2.groupby(['date', 'productID', 'name1']).agg(
        consumption=('consumption', 'sum'),
        received=('received', 'sum'),
        close_balance=('closeBalance', 'sum'),
        open_balance=('openBalance', 'sum'),
        stockout_rate=('stockout', 'mean'),
        active_facilities=('hf_pk', 'nunique')
    ).reset_index().sort_values(['productID', 'date']).reset_index(drop=True)
    
    prod_monthly['month'] = prod_monthly['date'].dt.month
    prod_monthly['year'] = prod_monthly['date'].dt.year
    prod_monthly['quarter_num'] = prod_monthly['date'].dt.quarter
    
    for lag in [1, 2, 3]:
        prod_monthly['lag_' + str(lag)] = prod_monthly.groupby('productID')['consumption'].shift(lag)
    
    prod_monthly['rolling_mean_3'] = prod_monthly.groupby('productID')['consumption'].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    prod_monthly['rolling_std_3'] = prod_monthly.groupby('productID')['consumption'].transform(
        lambda s: s.shift(1).rolling(3).std()
    ).fillna(0)
    
    prod_monthly['target'] = prod_monthly.groupby('productID')['consumption'].shift(-1)
    
    latest_monthly_per_product = prod_monthly.sort_values('date').groupby('productID').last().reset_index()
    avg_monthly_demand = prod_monthly.groupby('productID')['consumption'].mean().round(2).reset_index()
    avg_monthly_demand.columns = ['productID', 'Monthly_Consumption_Rate']
    
    stockout_stats = df_s2.groupby('productID')['stockout'].agg(stockouts='sum', reports='count').reset_index()
    stockout_stats['Stockout_Rate'] = (stockout_stats['stockouts'] / stockout_stats['reports'] * 100).round(2)
    
    facility_summary = pd.merge(latest_monthly_per_product[['productID', 'name1', 'close_balance']], avg_monthly_demand, on='productID')
    facility_summary = pd.merge(facility_summary, stockout_stats[['productID', 'Stockout_Rate']], on='productID')
    facility_summary.rename(columns={'close_balance': 'current_closing_balance'}, inplace=True)
    facility_summary['Months_Of_Inventory'] = (facility_summary['current_closing_balance'] / facility_summary['Monthly_Consumption_Rate']).round(2)
    
    return df_s2, df_s3, prod_monthly, facility_summary

demand_artifact, expiry_artifact = load_saved_models()
raw_consumption, raw_inventory, product_time_series, facility_stats = load_prepared_data()

st.sidebar.markdown("### Navigation")
section_choice = st.sidebar.radio(
    "Select Section",
    [
        "1. Executive Summary",
        "2. Demand Forecasting",
        "3. Inventory Analysis",
        "4. Expiry-Risk Assessment",
        "5. Operational Action Plan"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Information**")
st.sidebar.caption("Algions AI Labs Private Limited")
st.sidebar.caption("Domain: Pharmacy Management")

if section_choice == "1. Executive Summary":
    st.markdown('<div class="main-header">Pharmacy Medicine Demand and Expiry-Risk Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Decision-Support Machine Learning Solution | Algions AI Labs</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{raw_consumption["hf_pk"].nunique():,}</div><div class="metric-label">Health Facilities</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{raw_consumption["productID"].nunique()}</div><div class="metric-label">Essential Medicines</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{raw_consumption["date"].nunique()}</div><div class="metric-label">Historical Months</div></div>', unsafe_allow_html=True)
    with col4:
        best_wape = demand_artifact['metrics']['wape'] if demand_artifact else 24.46
        st.markdown(f'<div class="metric-card"><div class="metric-value">{100.0 - best_wape:.2f}%</div><div class="metric-label">Model Accuracy (WAPE {best_wape:.2f}%)</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("#### Operational Problem Framing")
        st.markdown("""
        Public healthcare facilities face simultaneous challenges of:
        - **Stockouts:** Depletion of life-saving antibiotics and maternal health products.
        - **Wastage:** Overstocked batches expiring unsold on clinic shelves.
        - **Traditional Forecasting Limitations:** Static reorder thresholds fail to capture seasonality and delivery volatility.
        """)
    with right_col:
        st.markdown("#### Machine Learning Formulation")
        st.markdown("""
        - **Prediction Unit:** Product-Month per reporting cycle across 1,091 facilities.
        - **Target:** Next month consumption quantity (t+1).
        - **Horizon:** 1 month forward (matching routine procurement cycles).
        - **Split Strategy:** Strict chronological split (70% Train, 15% Val, 15% Test) with zero lookahead leakage.
        """)
        
    st.markdown("---")
    st.markdown("#### Monthly System-Wide Medicine Consumption (2019 - 2023)")
    monthly_trend = raw_consumption.groupby('date')['consumption'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(monthly_trend['date'], monthly_trend['consumption'], color='#2563EB', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel("Reporting Date")
    ax.set_ylabel("Total Dispensed Units")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

elif section_choice == "2. Demand Forecasting":
    st.markdown('<div class="main-header">Product-Level Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live Machine Learning Inference Using Saved Model Artifact</div>', unsafe_allow_html=True)
    
    med_list = facility_stats[['productID', 'name1']].drop_duplicates().sort_values('productID')
    med_map = dict(zip(med_list['name1'], med_list['productID']))
    
    selected_name = st.selectbox("Select Medicine Product", list(med_map.keys()))
    selected_id = med_map[selected_name]
    
    history_subset = product_time_series[product_time_series['productID'] == selected_id].sort_values('date').reset_index(drop=True)
    latest_record = history_subset.dropna(subset=['consumption']).iloc[-1]
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f'<div class="metric-card"><div class="metric-value">ID #{selected_id}</div><div class="metric-label">Product Identifier</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{int(latest_record["consumption"]):,}</div><div class="metric-label">Latest Monthly Consumption</div></div>', unsafe_allow_html=True)
    with col_c:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{int(latest_record["active_facilities"])}</div><div class="metric-label">Reporting Health Facilities</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if demand_artifact and 'model' in demand_artifact:
        model = demand_artifact['model']
        feature_cols = demand_artifact['feature_columns']
        input_features = history_subset.iloc[[-1]]
        
        if not input_features[feature_cols].isnull().any().any():
            predicted_raw = model.predict(input_features[feature_cols])[0]
            predicted_demand = max(0, int(round(predicted_raw)))
            baseline_ma = int(round(input_features['rolling_mean_3'].iloc[0]))
            difference = predicted_demand - baseline_ma
            
            res_left, res_right = st.columns(2)
            with res_left:
                st.markdown(f"""
                <div style="background-color: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 18px;">
                    <div style="font-size: 13px; color: #166534; font-weight: 600; text-transform: uppercase;">Next Month Forecast</div>
                    <div style="font-size: 32px; font-weight: 800; color: #15803D;">{predicted_demand:,} Units</div>
                    <div style="font-size: 13px; color: #4B5563; margin-top: 4px;">Model: Random Forest Regressor | 14 Lag & Operational Features</div>
                </div>
                """, unsafe_allow_html=True)
            with res_right:
                st.markdown(f"""
                <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 18px;">
                    <div style="font-size: 13px; color: #475569; font-weight: 600; text-transform: uppercase;">Baseline Comparison</div>
                    <div style="font-size: 32px; font-weight: 800; color: #334155;">{baseline_ma:,} Units</div>
                    <div style="font-size: 13px; color: #64748B; margin-top: 4px;">3-Month Moving Average (Model Variance: {difference:+,.0f} units)</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(history_subset['date'], history_subset['consumption'], label='Historical Monthly Demand', color='#0284C7', marker='o', linewidth=2)
            ax.plot(history_subset['date'], history_subset['rolling_mean_3'], label='3-Month Moving Average', color='#94A3B8', linestyle='--', linewidth=1.5)
            
            next_date = history_subset['date'].max() + pd.DateOffset(months=1)
            ax.scatter([next_date], [predicted_demand], color='#16A34A', s=120, label='Model Forecast (t+1)', zorder=5)
            
            ax.set_title(f"Demand Trajectory: {selected_name}", fontsize=13, fontweight='600')
            ax.set_xlabel("Date")
            ax.set_ylabel("Units")
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.4)
            st.pyplot(fig)
            
            st.markdown("#### Model Performance on Held-Out Test Set")
            perf_df = pd.DataFrame({
                'Metric': ['Weighted Absolute Percentage Error (WAPE)', 'Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)'],
                'Random Forest (Selected)': ['24.46%', '10,507 units', '25,669 units'],
                'Gradient Boosting': ['26.65%', '11,447 units', '26,115 units'],
                '3-Month Moving Average Baseline': ['34.60%', '14,860 units', '36,576 units']
            })
            st.table(perf_df)

elif section_choice == "3. Inventory Analysis":
    st.markdown('<div class="main-header">Facility and Warehouse Inventory Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inventory Turnover, Months of Stock (MOS), and Stockout Rates</div>', unsafe_allow_html=True)
    
    severe_stockouts = (facility_stats['Stockout_Rate'] > 20.0).sum()
    severe_overstock = (facility_stats['Months_Of_Inventory'] > 6.0).sum()
    understock = (facility_stats['Months_Of_Inventory'] < 1.0).sum()
    balanced_stock = len(facility_stats) - severe_overstock - understock
    median_mos = facility_stats['Months_Of_Inventory'].median()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #DC2626;">{severe_stockouts}</div><div class="metric-label">High Stockout Vulnerability (>20%)</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #D97706;">{severe_overstock}</div><div class="metric-label">Excess Stock (>6 Months MOS)</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #16A34A;">{balanced_stock}</div><div class="metric-label">Balanced Inventory (1-6 Mo)</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{median_mos:.1f} Mo</div><div class="metric-label">Median Months of Stock</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Highest Stockout Frequency Medicines")
        top_stockouts = facility_stats.sort_values('Stockout_Rate', ascending=False).head(5)
        fig_s, ax_s = plt.subplots(figsize=(6, 3.5))
        ax_s.barh(top_stockouts['name1'], top_stockouts['Stockout_Rate'], color='#DC2626')
        ax_s.set_xlabel("Stockout Rate (%)")
        ax_s.invert_yaxis()
        ax_s.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig_s)
    with chart_right:
        st.markdown("#### Highest Inventory Reserves (Overstock Risk)")
        top_reserves = facility_stats.sort_values('Months_Of_Inventory', ascending=False).head(5)
        fig_r, ax_r = plt.subplots(figsize=(6, 3.5))
        ax_r.barh(top_reserves['name1'], top_reserves['Months_Of_Inventory'], color='#D97706')
        ax_r.set_xlabel("Months of Stock (MOS)")
        ax_r.invert_yaxis()
        ax_r.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig_r)
        
    st.markdown("---")
    st.markdown("#### Complete Medicine Inventory Health Table")
    inv_display = facility_stats[['productID', 'name1', 'current_closing_balance', 'Monthly_Consumption_Rate', 'Months_Of_Inventory', 'Stockout_Rate']].sort_values('Months_Of_Inventory').reset_index(drop=True)
    inv_display.columns = ['ID', 'Medicine Name', 'Closing Stock (Units)', 'Monthly Demand (Units)', 'Months of Stock (MOS)', 'Stockout Rate (%)']
    st.dataframe(inv_display, hide_index=True)

elif section_choice == "4. Expiry-Risk Assessment":
    st.markdown('<div class="main-header">Expiry-Risk Assessment and Batch Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">First Expiry, First Out (FEFO) Decision Support Prototype</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <strong>Data Audit Transparency Notice:</strong> Routine DHIS2 facility records (S2_Dhis2Data.csv) and warehouse stock files (S3.csv)
        track aggregate monthly inventory balances and stockouts, but do not record manufacturer batch lot numbers or exact expiration dates.
        To maintain scientific integrity, batch expiry risk is demonstrated via an explicit FEFO decision rule simulation.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h5>FEFO Decision Rule Formulation</h5>", unsafe_allow_html=True)
    st.markdown("A medicine batch is categorized as **At-Risk** if remaining physical stock exceeds projected patient consumption prior to expiration:")
    st.latex(r"\text{Expiry Risk} = 1 \quad \text{if} \quad \text{Remaining Quantity} > (\text{Days to Expiry} \times \text{Daily Demand Rate})")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Interactive Batch Expiry-Risk Evaluator")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        selected_med_id = st.selectbox("Select Medicine Product", sorted(facility_stats['productID'].unique()))
        med_name_str = facility_stats[facility_stats['productID'] == selected_med_id]['name1'].iloc[0]
        st.caption(f"Selected: {med_name_str}")
        batch_stock = st.number_input("Remaining Batch Quantity (Units)", min_value=10, max_value=50000, value=2500, step=50)
    with sim_col2:
        days_left = st.number_input("Days Remaining to Expiry", min_value=5, max_value=730, value=30, step=5)
        default_rate = max(1.0, float(round(facility_stats[facility_stats['productID'] == selected_med_id]['Monthly_Consumption_Rate'].iloc[0] / 30.0, 2)))
        daily_rate = st.number_input("Estimated Daily Demand Rate", min_value=0.1, max_value=5000.0, value=default_rate, step=1.0)
        
    expected_dispensed = days_left * daily_rate
    
    if st.button("Evaluate Batch Expiry Risk"):
        if expiry_artifact and 'model' in expiry_artifact:
            clf_model = expiry_artifact['model']
            clf_cols = expiry_artifact['feature_columns']
            test_row = pd.DataFrame([{
                'Medicine_ID': selected_med_id,
                'Remaining_Quantity': batch_stock,
                'Days_To_Expiry': days_left,
                'Daily_Demand_Rate': daily_rate
            }])
            flag = clf_model.predict(test_row[clf_cols])[0]
            
            if flag == 1:
                st.markdown(f"""
                <div class="status-badge-risk">
                    HIGH EXPIRY RISK FLAGGED<br>
                    Current Stock: {batch_stock:,} units | Expected Dispensing Before Expiry: {int(expected_dispensed):,} units<br>
                    Projected Unsold Spoilage: {int(batch_stock - expected_dispensed):,} units<br>
                    Recommended Action: Prioritize dispensing under FEFO or initiate immediate stock transfer to high-volume clinics.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-badge-safe">
                    BATCH SAFE - LOW RISK<br>
                    Current Stock: {batch_stock:,} units | Expected Dispensing Before Expiry: {int(expected_dispensed):,} units<br>
                    Projected Consumption Buffer: {int(expected_dispensed - batch_stock):,} units<br>
                    Recommended Action: Standard inventory management; stock will be absorbed before expiration.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Saved expiry model artifact not found.")

elif section_choice == "5. Operational Action Plan":
    st.markdown('<div class="main-header">Operational Recommendations and Action Plan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Decision-Support Guidance Derived from ML Forecasts and Turnover Ratios</div>', unsafe_allow_html=True)
    
    plan_data = facility_stats.copy()
    
    def assign_action_category(row):
        if row['Months_Of_Inventory'] < 1.0 or row['Stockout_Rate'] > 20.0:
            return 'Urgent Replenishment Required'
        elif row['Months_Of_Inventory'] > 6.0:
            return 'Excess Stock: Apply FEFO / Redistribute'
        else:
            return 'Maintain Standard Buffer Stock'
            
    plan_data['Operational_Action'] = plan_data.apply(assign_action_category, axis=1)
    
    selected_filter = st.selectbox("Filter Recommendations by Category", ["All", "Urgent Replenishment Required", "Excess Stock: Apply FEFO / Redistribute", "Maintain Standard Buffer Stock"])
    
    filtered_plan = plan_data if selected_filter == "All" else plan_data[plan_data['Operational_Action'] == selected_filter]
    
    summary_cols = ['productID', 'name1', 'current_closing_balance', 'Monthly_Consumption_Rate', 'Months_Of_Inventory', 'Stockout_Rate', 'Operational_Action']
    styled_table = filtered_plan[summary_cols].sort_values('Months_Of_Inventory').reset_index(drop=True)
    styled_table.columns = ['ID', 'Medicine Name', 'Closing Balance (Units)', 'Monthly Demand (Units)', 'Months of Stock (MOS)', 'Stockout Rate (%)', 'Recommended Action']
    st.dataframe(styled_table, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Summary of Portfolio Recommendations")
    action_counts = plan_data['Operational_Action'].value_counts()
    for category_name, count_value in action_counts.items():
        st.markdown(f"- **{category_name}:** {count_value} medicines ({round(count_value / len(plan_data) * 100, 1)}%)")
        
    st.markdown("""
    <div class="notice-box">
        <strong>Operational Advisory Disclaimer:</strong> All model forecasts, stockout alerts, and inventory rotation recommendations
        are operational decision-support guides intended for licensed pharmacy supply officers. They do not constitute clinical treatment or medical prescription advice.
    </div>
    """, unsafe_allow_html=True)
