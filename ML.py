import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
# Set up the Streamlit page configuration
st.set_page_config(page_title="Financial Dashboard and Analysis Tool", layout="wide")

# Title of the dashboard
st.title("Company Financial Dashboard and Analysis Tool")

# Sidebar menu for navigation
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Select an option", [
    "Dashboard",
    "Budget Forecasting Tool", 
    "Revenue Analysis Tool", 
    "Cost Analysis Tool",
    "Scenario Analysis Tool"
])

if app_mode == "Dashboard":
    # File uploader for the dashboard
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the CSV file, assuming the first column as the index (attributes)
            df = pd.read_csv(uploaded_file, index_col=0)
            
            # Transpose the DataFrame so that attributes become columns and years become rows
            df = df.T

            # Clean up column names by removing whitespace and converting to lowercase
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

            # Convert all data to numeric, forcing non-numeric values to NaN
            df = df.apply(pd.to_numeric, errors='coerce')

            # Display the cleaned data
            st.write("### Cleaned Data")
            st.dataframe(df)

            # Generate additional metrics if necessary
            df['tax_rate'] = df['tax_expense'] / df['profit_before_tax'] * 100
            df['net_profit_margin'] = df['profit_after_tax'] / df['total_revenue'] * 100

            # Create and display different types of charts

            # Cluster Chart for Total Revenue and Profit Before Tax
            cluster_chart = alt.Chart(df.reset_index()).mark_point(size=100).encode(
                x=alt.X('total_revenue:Q', title="Total Revenue"),
                y=alt.Y('profit_before_tax:Q', title="Profit Before Tax"),
                color=alt.Color('index:N', title='Financial Year'),
                tooltip=['index', 'total_revenue', 'profit_before_tax']
            ).properties(
                title="Cluster Chart: Total Revenue vs. Profit Before Tax"
            )

            # Line Chart for Profit Before Tax
            profit_chart = alt.Chart(df.reset_index()).mark_line(point=True).encode(
                x=alt.X('index:O', title="Financial Year"),
                y=alt.Y('profit_before_tax:Q', title="Profit Before Tax", axis=alt.Axis(format=",.0f")),
                color=alt.Color('index:N', title='Financial Year'),
                tooltip=['index', 'profit_before_tax']
            ).properties(
                title="Profit Before Tax Over Years"
            )

            # Line Chart for Tax Rate
            tax_rate_chart = alt.Chart(df.reset_index()).mark_line(point=True).encode(
                x=alt.X('index:O', title="Financial Year"),
                y=alt.Y('tax_rate:Q', title="Tax Rate (%)", axis=alt.Axis(format=",.2f")),
                color=alt.Color('index:N', title='Financial Year'),
                tooltip=['index', 'tax_rate']
            ).properties(
                title="Tax Rate Over Years"
            )

            # Bar Chart for Net Profit Margin
            net_profit_margin_chart = alt.Chart(df.reset_index()).mark_bar().encode(
                x=alt.X('index:O', title="Financial Year"),
                y=alt.Y('net_profit_margin:Q', title="Net Profit Margin (%)", axis=alt.Axis(format=",.2f")),
                color=alt.Color('index:N', title='Financial Year'),
                tooltip=['index', 'net_profit_margin']
            ).properties(
                title="Net Profit Margin Over Years"
            )

            # Bubble Chart for Tax Expense vs. Profit After Tax
            bubble_chart = alt.Chart(df.reset_index()).mark_circle(size=100).encode(
                x=alt.X('tax_expense:Q', title="Tax Expense"),
                y=alt.Y('profit_after_tax:Q', title="Profit After Tax"),
                color=alt.Color('index:N', title='Financial Year'),
                tooltip=['index', 'tax_expense', 'profit_after_tax']
            ).properties(
                title="Bubble Chart: Tax Expense vs. Profit After Tax"
            )

            # Display the charts in a two-column layout
            col1, col2 = st.columns(2)
            col1.altair_chart(cluster_chart, use_container_width=True)
            col2.altair_chart(profit_chart, use_container_width=True)

            col3, col4 = st.columns(2)
            col3.altair_chart(tax_rate_chart, use_container_width=True)
            col4.altair_chart(net_profit_margin_chart, use_container_width=True)

            st.altair_chart(bubble_chart, use_container_width=True)

            # Display Key Metrics
            st.write("### Key Metrics")
            st.metric(label="Latest Revenue", value=f"{df['total_revenue'].iloc[-1]:,}")
            st.metric(label="Latest Profit Before Tax", value=f"{df['profit_before_tax'].iloc[-1]:,}")
            st.metric(label="Latest Profit After Tax", value=f"{df['profit_after_tax'].iloc[-1]:,}")
            st.metric(label="Retained Earnings", value=f"{df['balance_at_the_end_of_the_year'].iloc[-1]:,}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.warning("Please upload a CSV file to see the dashboard.")

elif app_mode == "Budget Forecasting Tool":
    st.title("Budget Forecasting Tool")

    # File uploader in Streamlit with unique key
    uploaded_file = st.file_uploader("Upload your CSV file for Budget Forecasting", type=["csv"], key="budget_forecasting_uploader")

    if uploaded_file is not None:
        # Read the data
        data = pd.read_csv(uploaded_file)
        
        # Transpose and clean the data
        transposed_data = data.T
        transposed_data.columns = transposed_data.iloc[0]
        transposed_data = transposed_data.drop(transposed_data.index[0])
        transposed_data.index.name = 'Year'
        transposed_data.index = pd.to_numeric(transposed_data.index)
        st.write("Transposed Data:")
        st.dataframe(transposed_data)

        # List of columns for forecasting
        forecastable_columns = [
            'Sales Turnover',
            'Net Sales',
            'Total Income',
            'Operating Profit',
            'Reported Net Profit',
            'Earning Per Share (Rs)'
        ]

        # User input for the metric to forecast
        metric = st.selectbox("Select the financial metric for forecasting", forecastable_columns)

        # Function to forecast using ARIMA
        def forecast_with_arima(df, column_name, forecast_years=5):
            data = df[column_name].dropna().astype(float)
            model = ARIMA(data, order=(5, 1, 0))  # Adjust order as necessary
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_years)
            forecast_years = [data.index[-1] + i for i in range(1, forecast_years + 1)]
            forecast_df = pd.DataFrame({'Year': forecast_years, f'Forecasted {column_name}': forecast})
            return forecast_df

        # Perform forecasting
        forecast_df = forecast_with_arima(transposed_data, metric)

        # Display results
        st.write(f"Forecasting Results for {metric}:")
        st.dataframe(forecast_df)

        # Plot the results
        plt.figure(figsize=(14, 7))
        plt.plot(transposed_data.index, transposed_data[metric].astype(float), label='Historical Data', marker='o')
        plt.plot(forecast_df['Year'], forecast_df[f'Forecasted {metric}'], label='Forecasted Data', linestyle='--', marker='o')
        plt.xlabel('Year')
        plt.ylabel(metric)
        plt.title(f'Forecasting of {metric}')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

elif app_mode == "Revenue Analysis Tool":
    st.title("Revenue Analysis Tool")

    # File uploader in Streamlit with unique key
    uploaded_file = st.file_uploader("Upload your CSV file for Revenue Analysis", type=["csv"], key="revenue_analysis_uploader")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Transpose the data
        transposed_data = data.T
        transposed_data.columns = transposed_data.iloc[0]
        transposed_data = transposed_data.drop(transposed_data.index[0])
        transposed_data.index.name = 'Year'
        transposed_data.index = pd.to_numeric(transposed_data.index)
        st.write("Transposed Data:")
        st.dataframe(transposed_data)

        # Identify revenue-related columns
        revenue_columns = [
            'Total Share Capital',
            'Total Reserves and Surplus',
            'Total Shareholders\' Funds',
            'Total Current Assets',
            'FOB Value Of Goods',
            'Other Earnings'
        ]
        
        # Check if these columns exist in the dataset
        available_columns = [col for col in revenue_columns if col in transposed_data.columns]

        # Function to calculate year-over-year revenue growth
        def calculate_revenue_growth(df, column_name):
            df[f'{column_name} Growth'] = df[column_name].pct_change() * 100
            return df[[column_name, f'{column_name} Growth']]

        # User input for the revenue metric to analyze
        metric = st.selectbox("Select the revenue metric for analysis", available_columns)

        # Perform analysis
        growth_df = calculate_revenue_growth(transposed_data, metric)

        # Display results
        st.write(f"Revenue Growth Analysis for {metric}:")
        st.dataframe(growth_df)

        # Plot the results
        plt.figure(figsize=(14, 7))
        plt.plot(growth_df.index, growth_df[metric].astype(float), label='Revenue', marker='o')
        plt.plot(growth_df.index, growth_df[f'{metric} Growth'], label='Growth Rate (%)', linestyle='--', marker='o')
        plt.xlabel('Year')
        plt.ylabel(f'{metric} and Growth Rate (%)')
        plt.title(f'Revenue Growth Analysis for {metric}')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

elif app_mode == "Cost Analysis Tool":
    st.title("Cost Analysis Tool")

    # File uploader in Streamlit with unique key
    uploaded_file = st.file_uploader("Upload your CSV file for Cost Analysis", type=["csv"], key="cost_analysis_uploader")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the original data
    st.write("Original Data:")
    st.write(data.head())

    # Feature Selection
    selected_features = st.multiselect("Select features for clustering:", data.columns.tolist())
    if selected_features:
        selected_data = data[selected_features]

        # Handling Categorical and Numerical Data
        categorical_cols = selected_data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = selected_data.select_dtypes(include=['number']).columns.tolist()

        if categorical_cols:
            selected_data = pd.get_dummies(selected_data, columns=categorical_cols)

        if numerical_cols:
            scaler = StandardScaler()
            selected_data[numerical_cols] = scaler.fit_transform(selected_data[numerical_cols])

        st.write("Preprocessed Data:")
        st.dataframe(selected_data)

        # Number of Clusters
        num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)

        # Agglomerative Clustering
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
        cluster_labels = clustering_model.fit_predict(selected_data)

        # Add cluster labels to original data
        data['Cluster'] = cluster_labels

        # Display the clustered data in table form
        st.write("Clustered Data with Labels:")
        st.dataframe(data)

        # Silhouette Score
        silhouette_avg = silhouette_score(selected_data, cluster_labels)
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")

        # PCA for Dimensionality Reduction
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(selected_data)

        # Plotting the Clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=cluster_labels, palette="viridis")
        plt.title("Clusters Visualized with PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Cluster")
        st.pyplot(plt)

elif app_mode == "Scenario Analysis Tool":
    st.title("Scenario Analysis Tool")

    uploaded_file = st.file_uploader("Upload your CSV file for Scenario Analysis", type=["csv"], key="scenario_analysis_uploader")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Transpose the data
    transposed_data = data.T
    transposed_data.columns = transposed_data.iloc[0]  # Set the first row as header
    transposed_data = transposed_data.drop(transposed_data.index[0])  # Drop the old header row
    transposed_data.reset_index(inplace=True)
    transposed_data.rename(columns={'index': 'Year'}, inplace=True)
    
    # Display the transposed data
    st.write("Transposed Data:")
    st.dataframe(transposed_data)
    
    # Define revenue, cost, and profit variables
    revenue_vars = ['Sales Turnover', 'Net Sales', 'Total Income', 'Other Income']
    cost_vars = ['Raw Materials', 'Power & Fuel Cost', 'Employee Cost', 'Selling and Admin Expenses', 'Miscellaneous Expenses']
    profit_vars = ['Operating Profit', 'PBDIT', 'Profit Before Tax', 'Reported Net Profit', 'Earnings Per Share (EPS)']
    
    # Create a dropdown to select scenario
    scenario = st.selectbox("Select Scenario", ["Best-Case", "Worst-Case", "Most-Likely"])
    
    # Define scenario assumptions
    if scenario == "Best-Case":
        revenue_increase = 0.1  # 10% increase
        cost_decrease = 0.05  # 5% decrease
    elif scenario == "Worst-Case":
        revenue_increase = -0.1  # 10% decrease
        cost_decrease = 0.1  # 10% increase
    else:  # Most-Likely
        revenue_increase = 0.03  # 3% increase
        cost_decrease = 0.02  # 2% decrease
    
    # Calculate new values based on the selected scenario
    scenario_data = transposed_data.copy()
    
    # Apply scenario changes to revenue
    for col in revenue_vars:
        if col in scenario_data.columns:
            scenario_data[col] = scenario_data[col].astype(float) * (1 + revenue_increase)
    
    # Apply scenario changes to costs
    for col in cost_vars:
        if col in scenario_data.columns:
            scenario_data[col] = scenario_data[col].astype(float) * (1 - cost_decrease)
    
    # Recalculate profit metrics
    if 'Operating Profit' in scenario_data.columns:
        scenario_data['Operating Profit'] = (
            scenario_data['Total Income'].astype(float) - scenario_data[['Raw Materials', 'Power & Fuel Cost', 'Employee Cost', 'Selling and Admin Expenses', 'Miscellaneous Expenses']].astype(float).sum(axis=1)
        )
    if 'Reported Net Profit' in scenario_data.columns:
        scenario_data['Reported Net Profit'] = (
            scenario_data['Operating Profit'].astype(float) - scenario_data[['Interest', 'Depreciation', 'Tax']].astype(float).sum(axis=1)
        )
    
    # Display the scenario data
    st.write(f"Data for {scenario} Scenario:")
    st.dataframe(scenario_data)
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in revenue_vars + profit_vars:
        if col in scenario_data.columns:
            ax.plot(scenario_data['Year'], scenario_data[col], label=col)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title(f'{scenario} Scenario Analysis')
    ax.legend()
    ax.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(fig)