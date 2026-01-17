import json
import logging
from app.utils import embedding
from langchain_community.vectorstores import Chroma
from app.core.config import settings
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def _generate_recommendation(prediction, confidence, quantity, current_stock, 
                            avg_daily_sales, days_until_stockout, product_name):
    """Generate human-readable recommendation with robust error handling"""
    
    try:
        # Sanitize inputs - handle NaN, inf, and None values
        quantity = int(quantity) if pd.notna(quantity) else 1
        current_stock = int(current_stock) if pd.notna(current_stock) else 0
        avg_daily_sales = float(avg_daily_sales) if pd.notna(avg_daily_sales) and np.isfinite(avg_daily_sales) else 0
        days_until_stockout = float(days_until_stockout) if pd.notna(days_until_stockout) and np.isfinite(days_until_stockout) else 0
        confidence = float(confidence) if pd.notna(confidence) and np.isfinite(confidence) else 0
        
        # Check if quantity can be fulfilled
        can_fulfill = current_stock >= quantity
        
        # Base recommendations by stock status
        if prediction == 'High Stock':
            if confidence >= 0.8:
                if can_fulfill:
                    return f"✅ {product_name} is readily available with HIGH STOCK levels. We have {current_stock} units in stock, and you're requesting {quantity}. You can safely purchase now. Expected availability: {days_until_stockout:.0f}+ days."
                else:
                    shortage = quantity - current_stock
                    return f"⚠️ {product_name} has HIGH STOCK overall ({current_stock} units), but you're requesting {quantity} units which exceeds current inventory. We recommend ordering {current_stock} units now and placing a backorder for the remaining {shortage} units."
            else:
                return f"✅ {product_name} appears to be in stock (confidence: {confidence:.0%}). Current inventory: {current_stock} units. Safe to proceed with your order of {quantity} unit(s)."
        
        elif prediction == 'Medium Stock':
            if confidence >= 0.8:
                if can_fulfill:
                    return f"⚠️ {product_name} has MEDIUM STOCK levels. Current inventory: {current_stock} units (you need {quantity}). Recent demand is moderate ({avg_daily_sales:.1f} units/day). We recommend purchasing soon as stock may run low in {days_until_stockout:.0f} days. Consider ordering within the next few days."
                else:
                    return f"⚠️ {product_name} has MEDIUM STOCK but cannot fully fulfill your order of {quantity} units (only {current_stock} available). We recommend: 1) Order {current_stock} units now, 2) Check back in 3-5 days for restock, or 3) Consider a similar alternative product."
            else:
                return f"⚠️ {product_name} has MEDIUM STOCK levels (confidence: {confidence:.0%}). Inventory status is uncertain. We recommend checking with our support team before placing a large order of {quantity} units."
        
        else:  # Low Stock
            if confidence >= 0.8:
                if can_fulfill and quantity <= 3:
                    return f"🔴 URGENT: {product_name} is experiencing LOW STOCK. Only {current_stock} units available, and demand is high ({avg_daily_sales:.1f} units/day). Stock may run out in {days_until_stockout:.0f} days. Your order of {quantity} unit(s) can be fulfilled, but we STRONGLY recommend purchasing immediately to ensure availability."
                else:
                    fulfillment_status = 'cannot be fully fulfilled' if not can_fulfill else 'would deplete most of our inventory'
                    return f"🔴 ALERT: {product_name} is in LOW STOCK ({current_stock} units) and experiencing high demand. Your request of {quantity} units {fulfillment_status}. Recommendations: 1) Purchase immediately if critical, 2) Consider backorder (3-7 day wait), 3) Check alternative products, or 4) Contact sales for bulk order options."
            else:
                return f"🔴 {product_name} may be out of stock or very low inventory (confidence: {confidence:.0%}). Current status uncertain. We strongly recommend contacting our sales team before placing your order of {quantity} units to verify availability and delivery timeline."
    
    except Exception as e:
        # Fallback recommendation if anything fails
        logger.error(f"Error generating recommendation: {str(e)}")
        return f"Product: {product_name}. Current stock: {current_stock} units. Requested: {quantity} units. Status: {prediction}. Please contact support for detailed availability information."


def retrieve_document(query: str) -> str:
    """Retrieve relevant documents from Vector DB based on user query.
    
    Args:
        query: User's search query
        
    Returns:
        JSON string with search results or error
    """
    scope = "general"
    function_description = "Retrieve product information from the vector database based on user query."
    query_description = "Search query to find relevant products"
    
    try:
        logger.info(f"Retrieving documents for query: {query}")
        db = Chroma(persist_directory=settings.vectorDBPath, embedding_function=embedding)
        results = db.similarity_search_with_score(query, k=5)
        return str(results)
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return json.dumps({"error": str(e)})
    

def check_availability(product_name: str, quantity: int):
    """Check product availability using ML model and inventory data.
    
    Args:
        product_name: Name of the product to check
        quantity: Number of units requested (default: 1)
        
    Returns:
        JSON string with availability status and recommendations
    """
    scope = "general"
    function_description = "The function used to check the availability of the product, when ever user plans to buy a product or ask for the product availability this function will be used to provide the availability status using ML model"
    product_name_description = "Product name for which availability needs to be checked"
    quantity_description = "Number of items to check availability for the product give by user. Need to ask user every time"

    try:
        # CRITICAL: Convert quantity to int (LLM may pass it as string)
        quantity = int(quantity) #if quantity else 1
        product_name = str(product_name).strip()
        
        print(product_name, quantity)
        # ========================================================================
        # STEP 1: Load Required Data
        # ========================================================================
        
        print(f"\n{'='*70}")
        print(f"CHECKING AVAILABILITY: {product_name}")
        print(f"{'='*70}")
        
        # Load product catalog
        catalog_df = pd.read_csv(r'D:\RND\workshop2\tai3_workshop\data\product_catalog_real.csv')
        
        # Load sales history
        sales_df = pd.read_csv(r'D:\RND\workshop2\tai3_workshop\data\sales_history_real.csv')
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        
        # Load current inventory
        inventory_df = pd.read_csv(r'D:\RND\workshop2\tai3_workshop\data\current_inventory_real.csv')
        
        # Load trained ML model
        model_artifacts = joblib.load(r'D:\RND\workshop2\tai3_workshop\app\models\demand_forecast_model.pkl')
        model = model_artifacts['model']
        scaler = model_artifacts['scaler']
        feature_names = model_artifacts['feature_names']
        
        print("✅ Data and model loaded")
        
        # ========================================================================
        # STEP 2: Find Product in Catalog
        # ========================================================================
        
        print(f"\n🔍 Searching for product: '{product_name}'")
        
        # Search by name (case-insensitive, partial match)
        product_match = catalog_df[
            catalog_df['product_name'].str.lower().str.contains(product_name.lower(), na=False)
        ]
        
        if product_match.empty:
            return json.dumps({
                'status': 'error',
                'message': f"Product '{product_name}' not found in catalog",
                'suggestion': "Please check the product name and try again",
                'available_products': catalog_df['product_name'].head(10).tolist()
            })
        
        # Get best match (first result)
        product = product_match.iloc[0]
        product_id = product['product_id']
        full_product_name = product['product_name']
        product_price = product['price']
        
        print(f"✅ Found: {full_product_name} (ID: {product_id})")
        print(f"   Price: ${product_price:.2f}")
        
        # ========================================================================
        # STEP 3: Get Recent Sales Data and Engineer Features
        # ========================================================================
        
        print(f"\n📊 Analyzing recent sales patterns...")
        
        # Get product sales history
        product_sales = sales_df[sales_df['product_id'] == product_id].copy()
        
        if len(product_sales) < 30:
            return json.dumps({
                'status': 'error',
                'message': f"Insufficient sales history for {full_product_name}",
                'suggestion': "This product may be new or have limited sales data"
            })
        
        # Sort by date
        product_sales = product_sales.sort_values('date')
        
        # Add time features if not present (for feature engineering)
        if 'day_of_week' not in product_sales.columns:
            product_sales['day_of_week'] = product_sales['date'].dt.dayofweek
        if 'month' not in product_sales.columns:
            product_sales['month'] = product_sales['date'].dt.month
        if 'year' not in product_sales.columns:
            product_sales['year'] = product_sales['date'].dt.year
        if 'is_weekend' not in product_sales.columns:
            product_sales['is_weekend'] = product_sales['day_of_week'].isin([5, 6]).astype(int)
        if 'is_holiday_season' not in product_sales.columns:
            product_sales['is_holiday_season'] = product_sales['month'].isin([11, 12]).astype(int)
        
        # Get last 30 days
        latest_date = product_sales['date'].max()
        last_30_days = product_sales[product_sales['date'] >= (latest_date - timedelta(days=30))]
        
        # ========================================================================
        # STEP 4: Engineer Features (Same as Training)
        # ========================================================================
        
        print(f"🔧 Engineering features...")
        
        features = {}
        
        # Rolling statistics
        for window in [7, 14, 30]:
            last_window = product_sales.tail(window)
            features[f'sales_mean_{window}d'] = last_window['daily_sales'].mean()
            features[f'sales_std_{window}d'] = last_window['daily_sales'].std() if len(last_window) > 1 else 0
            features[f'sales_sum_{window}d'] = last_window['daily_sales'].sum()
            features[f'sales_max_{window}d'] = last_window['daily_sales'].max()
            features[f'sales_min_{window}d'] = last_window['daily_sales'].min()
            features[f'revenue_sum_{window}d'] = last_window['daily_revenue'].sum()
        
        # Lag features
        for lag in [1, 7, 14]:
            lag_value = product_sales['daily_sales'].iloc[-lag] if len(product_sales) >= lag else 0
            features[f'sales_lag_{lag}'] = lag_value
        
        # Trend features
        features['sales_velocity'] = features['sales_mean_7d'] / (features['sales_mean_30d'] + 0.1)
        features['sales_trend'] = features['sales_mean_7d'] - features['sales_mean_14d']
        features['sales_acceleration'] = 0  # Simplified
        
        # Variability
        features['sales_cv'] = features['sales_std_30d'] / (features['sales_mean_30d'] + 0.1)
        features['sales_range_30d'] = features['sales_max_30d'] - features['sales_min_30d']
        
        # Zero sales streak
        zero_streak = 0
        for sale in product_sales['daily_sales'].tail(7).values[::-1]:
            if sale == 0:
                zero_streak += 1
            else:
                break
        features['zero_sales_streak'] = zero_streak
        
        # Days since last sale
        sales_dates = product_sales[product_sales['daily_sales'] > 0]['date']
        if len(sales_dates) > 0:
            features['days_since_sale'] = (latest_date - sales_dates.max()).days
        else:
            features['days_since_sale'] = 30
        
        # Product features
        features['product_price'] = product_price
        features['product_popularity'] = product['order_count']
        features['category_encoded'] = 0  # Simplified
        
        # Fill NaN
        for key in features:
            if pd.isna(features[key]):
                features[key] = 0
        
        # Only keep features that exist in the model's feature_names
        # Create a complete feature dict with all required features
        final_features = {}
        for fname in feature_names:
            if fname in features:
                final_features[fname] = features[fname]
            else:
                final_features[fname] = 0  # Default value for missing features
        
        print(f"✅ Features engineered")
        
        # ========================================================================
        # STEP 5: Make ML Prediction
        # ========================================================================
        
        print(f"\n🤖 Running ML prediction...")
        
        # Create feature vector using final_features (ensure correct order)
        X = pd.DataFrame([final_features])[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Map probabilities to classes
        classes = ['Low Stock', 'Medium Stock', 'High Stock']
        prob_dict = {
            'low_stock': float(probabilities[0]),
            'medium_stock': float(probabilities[1]),
            'high_stock': float(probabilities[2])
        }
        
        confidence = float(probabilities.max())
        
        print(f"✅ Prediction: {prediction} (Confidence: {confidence:.2%})")
        
        # ========================================================================
        # STEP 6: Get Current Inventory Status
        # ========================================================================
        
        print(f"\n📦 Checking current inventory...")
        
        inventory_info = inventory_df[inventory_df['product_id'] == product_id]
        
        if not inventory_info.empty:
            inventory_info = inventory_info.iloc[0]
            current_stock = int(inventory_info['current_stock'])
            avg_daily_sales = float(inventory_info['avg_daily_sales'])
            days_until_stockout = float(inventory_info['days_until_stockout'])
        else:
            # Estimate if not in inventory
            current_stock = int(features['sales_mean_30d'] * 10)
            avg_daily_sales = float(features['sales_mean_7d'])
            days_until_stockout = current_stock / (avg_daily_sales + 0.1)
        
        # Sanitize values before passing to recommendation
        current_stock = int(current_stock) if pd.notna(current_stock) else 0
        avg_daily_sales = float(avg_daily_sales) if pd.notna(avg_daily_sales) and np.isfinite(avg_daily_sales) else 0
        days_until_stockout = float(days_until_stockout) if pd.notna(days_until_stockout) and np.isfinite(days_until_stockout) else 0
        
        print(f"   Current stock: {current_stock} units")
        print(f"   Avg daily sales: {avg_daily_sales:.1f} units")
        print(f"   Days until stockout: {days_until_stockout:.1f} days")
        
        # ========================================================================
        # STEP 7: Generate Recommendation
        # ========================================================================
        
        print(f"\n💡 Generating recommendation...")
        
        recommendation = _generate_recommendation(
            prediction=prediction,
            confidence=confidence,
            quantity=quantity,
            current_stock=current_stock,
            avg_daily_sales=avg_daily_sales,
            days_until_stockout=days_until_stockout,
            product_name=full_product_name
        )
        
        print(f"✅ Recommendation generated: {recommendation[:100]}...")
        
        # ========================================================================
        # STEP 8: Build Response
        # ========================================================================
        
        result = {
            'status': 'success',
            'product': {
                'id': str(product_id),
                'name': str(full_product_name),
                'price': float(product_price),
                'category': str(product.get('category', 'N/A'))
            },
            'availability': {
                'status': str(prediction),
                'confidence': float(confidence),
                'probabilities': prob_dict
            },
            'inventory': {
                'current_stock': int(current_stock),
                'can_fulfill': bool(current_stock >= quantity),
                'units_requested': int(quantity),
                'avg_daily_sales': float(avg_daily_sales),
                'days_until_stockout': float(days_until_stockout)
            },
            'demand_insights': {
                'recent_7d_sales': int(final_features['sales_sum_7d']),
                'recent_30d_sales': int(final_features['sales_sum_30d']),
                'sales_velocity': float(final_features['sales_velocity']),
                'trend': 'increasing' if final_features['sales_trend'] > 0 else 'stable/decreasing'
            },
            'recommendation': str(recommendation),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n{'='*70}")
        print(f"✅ AVAILABILITY CHECK COMPLETE")
        print(f"{'='*70}")
        print(f"Status: {prediction}")
        print(f"Recommendation: {recommendation}")
        
        # Return JSON string
        return json.dumps(result, ensure_ascii=False)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return json.dumps({
            'status': 'error',
            'message': f'Required file not found: {str(e)}',
            'suggestion': 'Ensure all data files and model are in the current directory'
        })
    
    except Exception as e:
        logger.error(f"Error in check_availability: {str(e)}", exc_info=True)
        return json.dumps({
            'status': 'error',
            'message': f'Error checking availability: {str(e)}',
            'suggestion': 'Please try again or contact support'
        })