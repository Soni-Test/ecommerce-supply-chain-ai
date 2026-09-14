#Importing 
import os
import pandas as pd
from datetime import timedelta
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from dotenv import load_dotenv 

# Load credentials from your .env file
load_dotenv()

def run_pipeline():
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("OmniSegmentForecasting") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.2.18") \
        .getOrCreate()

    print("Loading and aggregating datasets...")

    # Load all datasets
    orders_df = pd.read_csv("olist_orders_dataset.csv")
    order_items_df = pd.read_csv("olist_order_items_dataset.csv")
    reviews_df = pd.read_csv("olist_order_reviews_dataset.csv")
    payments_df = pd.read_csv("olist_order_payments_dataset.csv")
    products_df = pd.read_csv("olist_products_dataset.csv")
    translation_df = pd.read_csv("product_category_name_translation.csv")
    sellers_df = pd.read_csv("olist_sellers_dataset.csv")
    geo_df = pd.read_csv("olist_geolocation_datasets.csv")

    # Merging the files to create a comprehensive dataset for analysis
    merged_df = orders_df.merge(order_items_df, on="order_id", how="left")
    merged_df = merged_df.merge(payments_df, on="order_id", how="left")
    merged_df = merged_df.merge(reviews_df, on="order_id", how="left")
    merged_df = merged_df.merge(products_df, on="product_id", how="left")
    merged_df = merged_df.merge(translation_df, on="product_category_name", how="left")
    merged_df = merged_df.merge(sellers_df, on="seller_id", how="left")

    # Define DB connection variables for both exports
    db_url = f"jdbc:postgresql://{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    db_properties = {
        "user": os.getenv('DB_USER'),
        "password": os.getenv('DB_PASSWORD'),
        "driver": "org.postgresql.Driver"
    }

    # --- NEW: Export master dataset to PostgreSQL for Power BI ---
    print("Exporting master dataset to PostgreSQL for Power BI...")
    # Clean the dataframe to avoid PySpark schema errors with NaNs/mixed types
    merged_clean = merged_df.fillna("Unknown").astype(str)
    
    # Convert to Spark and write the massive master table to pgAdmin
    spark_master_df = spark.createDataFrame(merged_clean)
    spark_master_df.write \
        .jdbc(url=db_url, table="ecommerce_master", mode="overwrite", properties=db_properties)
    # -------------------------------------------------------------

    print("Processing daily sales for forecasting...")
    merged_df["order_purchase_date"] = pd.to_datetime(merged_df["order_purchase_timestamp"]).dt.date
    daily_sales = merged_df.groupby("order_purchase_date")["price"].sum().reset_index()
    daily_sales.columns = ["date", "total_sales"]
    daily_sales["date"] = pd.to_datetime(daily_sales["date"])
    daily_sales.set_index("date", inplace=True)
    daily_sales = daily_sales.asfreq("D").fillna(0)

    print("Fitting SARIMA Model for time-series forecasting...")
    model = SARIMAX(daily_sales["total_sales"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(disp=False)

    print("Forecasting future demand for the next 30 days...")
    forecast = model_fit.forecast(steps=30)

    # Structure the historical and forecasted data together
    hist_df = daily_sales.reset_index()
    hist_df["type"] = "Historical"

    last_date = daily_sales.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "total_sales": forecast.values,
        "type": "Forecast"
    })

    combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
    spark_combined_df = spark.createDataFrame(combined_df)

    print(f"Exporting forecast results to PostgreSQL database: {os.getenv('DB_NAME')}...")
    # Write directly to pgAdmin
    spark_combined_df.select("date", "total_sales", "type") \
        .write \
        .jdbc(url=db_url, table="sales_predictions", mode="overwrite", properties=db_properties)
    
    print("Pipeline execution complete. Data is ready for the AI Agent and Power BI.")

if __name__ == "__main__":
    run_pipeline()

    