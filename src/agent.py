#Importing the library
import os
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables (API Key and DB credentials)
load_dotenv()

def get_forecasted_sales(days_ahead: int) -> str:
    """Queries the PostgreSQL database to get the forecasted sales for the upcoming days."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
        )
        cursor = conn.cursor()
        
        query = f"SELECT date, total_sales FROM sales_predictions WHERE type = 'Forecast' ORDER BY date ASC LIMIT {days_ahead};"
        cursor.execute(query)
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            return "No forecast data found."
            
        result = "Date | Forecasted Sales\n"
        for row in records:
            result += f"{row[0]} | ${row[1]:.2f}\n"
        return result
        
    except Exception as e:
        return f"Database error: {str(e)}"

def get_top_categories(limit: int) -> str:
    """Queries the master dataset to find the top-selling product categories by order volume."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
        )
        cursor = conn.cursor()
        
        query = f"""
            SELECT product_category_name, COUNT(order_id) as total_orders 
            FROM ecommerce_master 
            WHERE product_category_name != 'Unknown' 
            GROUP BY product_category_name 
            ORDER BY total_orders DESC 
            LIMIT {limit};
        """
        cursor.execute(query)
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            return "No category data found."
            
        result = "Category | Total Orders\n"
        for row in records:
            result += f"{row[0]} | {row[1]}\n"
        return result
    except Exception as e:
        return f"Database error: {str(e)}"

def run_agent():
    print("Initializing Supply Chain AI Agent with Extended Data Access...")
    client = genai.Client() 

    chat = client.chats.create(
        model="gemini-3.6-flash", 
        config=types.GenerateContentConfig(
            tools=[get_forecasted_sales, get_top_categories],
            temperature=0.2,
            system_instruction=(
                "You are an autonomous Supply Chain Analyst. Use your tools to fetch sales forecasts "
                "and top product categories from the database. Combine this data to recommend specific "
                "inventory, procurement, or marketing actions."
            )
        )
    )

    print("\nAgent is ready. Type 'exit' to quit.")
    while True:
        user_prompt = input("\nAsk the Agent a question: ")
        if user_prompt.lower() == 'exit':
            break
            
        print("\nAgent is analyzing...")
        response = chat.send_message(user_prompt)
        print("-" * 50)
        print(response.text)
        print("-" * 50)

if __name__ == "__main__":
    run_agent()
