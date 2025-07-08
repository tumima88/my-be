import uvicorn
import firebase_admin
import requests
import os
import google.generativeai as genai
import base64
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
db = None

# --- Firebase Initialization (for Deployment) ---
try:
    # Get the base64 encoded string from environment variables
    firebase_creds_b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_BASE64")
    if firebase_creds_b64:
        # Decode the base64 string to bytes, then decode bytes to a normal string
        creds_json_str = base64.b64decode(firebase_creds_b64).decode('utf-8')
        # Load the string as a dictionary
        creds_dict = json.loads(creds_json_str)
        # Initialize with the dictionary
        cred = credentials.Certificate(creds_dict)
    else:
        # Fallback for local development
        cred = credentials.Certificate("serviceAccountKey.json")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    print("✅ Firebase connection successful.")
except Exception as e:
    print(f"🔥 Firebase connection failed: {e}")

# Service API Keys
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET")
PAYPAL_API_BASE = os.getenv("PAYPAL_API_BASE")
PAYPAL_SANDBOX_EMAIL = os.getenv("PAYPAL_SANDBOX_EMAIL")
OPEN_EXCHANGE_RATES_API_KEY = os.getenv("OPEN_EXCHANGE_RATES_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini AI
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini AI client configured.")
    except Exception as e:
        print(f"🔥 Gemini AI configuration failed: {e}")
else:
    print("⚠️ Gemini AI key not found. Proxy will not function.")


# --- Pydantic Models ---
class CartItem(BaseModel):
    name: str
    quantity: int
    price: str

class OrderCaptureRequest(BaseModel):
    order_id: str

class GeminiRequest(BaseModel):
    prompt: str


# --- PayPal Helper Functions ---
def get_paypal_access_token():
    auth = (PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET)
    headers = {"Accept": "application/json", "Accept-Language": "en_US"}
    data = {"grant_type": "client_credentials"}
    try:
        response = requests.post(f"{PAYPAL_API_BASE}/v1/oauth2/token", auth=auth, headers=headers, data=data)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.HTTPError as err:
        raise HTTPException(status_code=500, detail=f"Failed to get PayPal token: {err.response.text}")

def capture_paypal_order(order_id: str):
    access_token = get_paypal_access_token()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
    try:
        response = requests.post(f"{PAYPAL_API_BASE}/v2/checkout/orders/{order_id}/capture", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        raise HTTPException(status_code=500, detail=f"Failed to capture PayPal order: {err.response.text}")

# --- FastAPI App ---
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Server is running"}

@app.get("/api/products")
def get_products():
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        docs = db.collection('products').stream()
        return [{"id": doc.id, **doc.to_dict()} for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch products: {e}")

@app.post("/api/paypal/create-order")
def create_paypal_order(cart_items: List[CartItem]):
    access_token = get_paypal_access_token()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
    cart_items_dict = [item.dict() for item in cart_items]
    total_value = sum(float(item['price']) * int(item['quantity']) for item in cart_items_dict)
    
    purchase_units = [{
        "amount": { "currency_code": "USD", "value": f"{total_value:.2f}" },
        "payee": { "email_address": PAYPAL_SANDBOX_EMAIL }
    }]
    payload = { "intent": "CAPTURE", "purchase_units": purchase_units, "application_context": { "return_url": "https://example.com/return", "cancel_url": "https://example.com/cancel" } }
    
    try:
        response = requests.post(f"{PAYPAL_API_BASE}/v2/checkout/orders", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        raise HTTPException(status_code=500, detail=f"Failed to create PayPal order: {err.response.text}")

@app.post("/api/paypal/capture-order")
def capture_order(request: OrderCaptureRequest):
    return capture_paypal_order(request.order_id)

@app.get("/api/convert-currency")
def get_exchange_rate(from_currency: str, to_currency: str):
    if from_currency == to_currency:
        return {"rate": 1}
    try:
        url = f"https://open.er-api.com/v6/latest/{from_currency}?apikey={OPEN_EXCHANGE_RATES_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        rates = response.json().get("rates", {})
        if to_currency not in rates:
            raise HTTPException(status_code=404, detail=f"Target currency '{to_currency}' not found.")
        return {"from": from_currency, "to": to_currency, "rate": rates[to_currency]}
    except requests.exceptions.HTTPError as err:
        raise HTTPException(status_code=500, detail=f"Failed to get exchange rate: {err.response.text}")

@app.post("/api/gemini/generate-email")
def generate_gemini_email(request: GeminiRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini AI client not configured on backend.")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(request.prompt)
        return {"success": True, "emailContent": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate email content: {e}")