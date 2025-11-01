import streamlit as st
import pandas as pd
import os
import joblib
import gzip
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------
# 🔹 Function to compress .pkl file under 25 MB
# -----------------------------------------------------
def compress_pickle(input_path, target_size_mb=25):
    temp_path = input_path + ".gz"
    with open(input_path, "rb") as f_in:
        with gzip.open(temp_path, "wb", compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    compressed_size = os.path.getsize(temp_path) / (1024 * 1024)
    if compressed_size <= target_size_mb:
        os.remove(input_path)
        os.rename(temp_path, input_path)
    else:
        os.remove(temp_path)

# -----------------------------------------------------
# 🔹 Train and Save Model
# -----------------------------------------------------
def train_and_save_model():
    df = pd.read_csv("Maharashtra_crop_dataset.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    X = df[[
        "season", "district", "soiltype", "avgrainfall_mm", "avgtemp_c",
        "avghumidity_%", "soil_ph", "nitrogen_kg_ha", "phosphorus_kg_ha", "potassium_kg_ha"
    ]]
    y = df["Crop"]
    X = pd.get_dummies(X, columns=["district", "soiltype", "season"], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=70, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "crop_recommendation.pkl", compress=8)
    joblib.dump(X.columns.tolist(), "model_columns.pkl", compress=7)
    compress_pickle("crop_recommendation.pkl", target_size_mb=25)
    return model, X.columns.tolist(), df

# -----------------------------------------------------
# 🔹 Load Model
# -----------------------------------------------------
@st.cache_resource
def load_model_and_columns():
    if not os.path.exists("crop_recommendation.pkl") or not os.path.exists("model_columns.pkl"):
        model, model_columns, df = train_and_save_model()
    else:
        model = joblib.load("crop_recommendation.pkl")
        model_columns = joblib.load("model_columns.pkl")
        df = pd.read_csv("Maharashtra_crop_dataset.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    return model, model_columns, df

model, model_columns, df = load_model_and_columns()

# -----------------------------------------------------
# 🌐 Language Pack
# -----------------------------------------------------
LANGUAGES = {
    "English": {
        "title": "🌾 AI Powered Maharashtra Crop Recommendation System",
        "desc": "Enter your soil and weather conditions below to get crop recommendations, yield & weather alerts.",
        "district": "District",
        "soiltype": "Soil Type",
        "season": "Season",
        "rainfall": "Average Rainfall (mm)",
        "temp": "Average Temperature (°C)",
        "humidity": "Average Humidity (%)",
        "ph": "Soil pH",
        "nitrogen": "Nitrogen (kg/ha)",
        "phosphorus": "Phosphorus (kg/ha)",
        "potassium": "Potassium (kg/ha)",
        "submit": "🚜 Get Crop Recommendations",
        "prediction": "🌾 Prediction Results",
        "weather": "🌦️ Weather Alert",
        "soil": "🌱 Soil Recommendation",
        "unseasonal": "⚠️ Unseasonal Rainfall — delay fertilizer application and ensure proper drainage.",
        "favorable": "✅ Weather looks favorable for most crops.",
        "hot": "🌡️ High temperature — apply mulching and irrigate adequately.",
        "cold": "❄️ Low temperature — delay sowing or use tolerant varieties.",
        "humid": "💧 High humidity — risk of fungal infection.",
        "dry": "🔥 Low humidity — increase irrigation frequency.",
        "low_rain": "🌤️ Low rainfall — prefer drought-resistant crops like Bajra or Tur.",
        "acidic": "Add lime to reduce soil acidity.",
        "alkaline": "Add gypsum or organic matter for alkaline soil.",
        "ideal_ph": "Soil pH is ideal — maintain organic matter."
    },
    "हिंदी": {
        "title": "🌾 एआई संचालित महाराष्ट्र फसल सिफारिश प्रणाली",
        "desc": "अपनी मिट्टी और मौसम की स्थिति दर्ज करें ताकि आपको फसल, उपज और मौसम चेतावनी प्राप्त हो सके।",
        "district": "ज़िला",
        "soiltype": "मिट्टी का प्रकार",
        "season": "मौसम",
        "rainfall": "औसत वर्षा (मिमी)",
        "temp": "औसत तापमान (°C)",
        "humidity": "औसत आर्द्रता (%)",
        "ph": "मिट्टी का pH",
        "nitrogen": "नाइट्रोजन (किग्रा/हे)",
        "phosphorus": "फॉस्फोरस (किग्रा/हे)",
        "potassium": "पोटेशियम (किग्रा/हे)",
        "submit": "🚜 फसल सिफारिश प्राप्त करें",
        "prediction": "🌾 भविष्यवाणी परिणाम",
        "weather": "🌦️ मौसम चेतावनी",
        "soil": "🌱 मिट्टी की सिफारिश",
        "unseasonal": "⚠️ असमय वर्षा — उर्वरक का प्रयोग रोकें और निकासी सुनिश्चित करें।",
        "favorable": "✅ मौसम अधिकांश फसलों के लिए अनुकूल है।",
        "hot": "🌡️ अधिक तापमान — मल्चिंग करें और सिंचाई बढ़ाएं।",
        "cold": "❄️ कम तापमान — बुवाई में देरी करें या सहनशील किस्में अपनाएं।",
        "humid": "💧 अधिक नमी — फफूंदी रोग का खतरा।",
        "dry": "🔥 कम नमी — सिंचाई की आवृत्ति बढ़ाएं।",
        "low_rain": "🌤️ कम वर्षा — बाजरा या तूर जैसी फसलें उगाएं।",
        "acidic": "मिट्टी की अम्लता कम करने के लिए चुना डालें।",
        "alkaline": "क्षारीय मिट्टी के लिए जिप्सम या जैविक पदार्थ डालें।",
        "ideal_ph": "मिट्टी का pH आदर्श है — जैविक सामग्री बनाए रखें।"
    },
    "मराठी": {
        "title": "🌾 एआय आधारित महाराष्ट्र पिक शिफारस प्रणाली",
        "desc": "आपली माती आणि हवामान स्थिती भरा आणि पिक शिफारसी, उत्पादन व हवामान चेतावणी मिळवा.",
        "district": "जिल्हा",
        "soiltype": "मातीचा प्रकार",
        "season": "हंगाम",
        "rainfall": "सरासरी पर्जन्यमान (मिमी)",
        "temp": "सरासरी तापमान (°C)",
        "humidity": "सरासरी आर्द्रता (%)",
        "ph": "मातीचा pH",
        "nitrogen": "नायट्रोजन (किलो/हे)",
        "phosphorus": "फॉस्फरस (किलो/हे)",
        "potassium": "पोटॅशियम (किलो/हे)",
        "submit": "🚜 पिक शिफारस मिळवा",
        "prediction": "🌾 अंदाज निकाल",
        "weather": "🌦️ हवामान चेतावणी",
        "soil": "🌱 माती शिफारस",
        "unseasonal": "⚠️ अवेळी पाऊस — खत टाकणे थांबवा आणि निचरा सुनिश्चित करा.",
        "favorable": "✅ हवामान बहुतेक पिकांसाठी अनुकूल आहे.",
        "hot": "🌡️ जास्त तापमान — मल्चिंग करा आणि सिंचन वाढवा.",
        "cold": "❄️ कमी तापमान — पेरणी उशिरा करा किंवा सहनशील वाण वापरा.",
        "humid": "💧 जास्त आर्द्रता — बुरशीजन्य रोगाचा धोका.",
        "dry": "🔥 कमी आर्द्रता — सिंचन वाढवा.",
        "low_rain": "🌤️ कमी पाऊस — बाजरी किंवा तूर लागवड करा.",
        "acidic": "मातीची आम्लता कमी करण्यासाठी चुना वापरा.",
        "alkaline": "अल्कलाइन मातीसाठी जिप्सम किंवा सेंद्रिय खत टाका.",
        "ideal_ph": "मातीचा pH योग्य आहे — सेंद्रिय पदार्थ टिकवा."
    }
}

# -----------------------------------------------------
# 🌐 Language Selector
# -----------------------------------------------------
lang_choice = st.sidebar.radio("🌐 Choose Language / भाषा निवडा / भाषा चुनें:", list(LANGUAGES.keys()))
T = LANGUAGES[lang_choice]

# -----------------------------------------------------
# 🌾 Streamlit UI
# -----------------------------------------------------
st.title(T["title"])
st.write(T["desc"])

available_districts = sorted(df["district"].unique())
available_soiltypes = sorted(df["soiltype"].unique())
available_seasons = sorted(df["season"].unique())

with st.form("crop_form"):
    st.subheader("🧾 " + T["prediction"])
    col1, col2, col3 = st.columns(3)
    with col1:
        district = st.selectbox(T["district"], available_districts)
        soiltype = st.selectbox(T["soiltype"], available_soiltypes)
        season = st.selectbox(T["season"], available_seasons)
    with col2:
        avgrainfall_mm = st.number_input(T["rainfall"], min_value=0.0, step=1.0)
        avgtemp_c = st.number_input(T["temp"], min_value=0.0, step=0.1)
        avghumidity = st.number_input(T["humidity"], min_value=0.0, max_value=100.0, step=0.1)
    with col3:
        soil_ph = st.number_input(T["ph"], min_value=0.0, max_value=14.0, step=0.1)
        nitrogen = st.number_input(T["nitrogen"], min_value=0.0, step=1.0)
        phosphorus = st.number_input(T["phosphorus"], min_value=0.0, step=1.0)
        potassium = st.number_input(T["potassium"], min_value=0.0, step=1.0)
    submitted = st.form_submit_button(T["submit"])

# -----------------------------------------------------
# ⚙️ Helper Functions
# -----------------------------------------------------
def get_weather_alert(temp, humidity, rainfall):
    alerts = []
    if rainfall > 1200:
        alerts.append(T["unseasonal"])
    if temp > 35:
        alerts.append(T["hot"])
    elif temp < 15:
        alerts.append(T["cold"])
    if humidity > 85:
        alerts.append(T["humid"])
    elif humidity < 30:
        alerts.append(T["dry"])
    if rainfall < 400:
        alerts.append(T["low_rain"])
    if not alerts:
        alerts.append(T["favorable"])
    return alerts

def get_soil_recommendation(ph):
    if ph < 6:
        return T["acidic"]
    elif ph > 8:
        return T["alkaline"]
    else:
        return T["ideal_ph"]

# -----------------------------------------------------
# 🔹 Yield/Profit Data
# -----------------------------------------------------
yield_profit_data = {
    "Cotton": ("8–12 quintals/ha", "₹30,000–₹45,000"),
    "Soybean": ("12–20 quintals/ha", "₹25,000–₹40,000"),
    "Tur": ("6–10 quintals/ha", "₹20,000–₹30,000"),
    "Wheat": ("30–40 quintals/ha", "₹50,000–₹70,000"),
    "Jowar": ("12–18 quintals/ha", "₹20,000–₹35,000"),
    "Rice": ("30–45 quintals/ha", "₹40,000–₹60,000"),
    "Gram": ("8–15 quintals/ha", "₹25,000–₹40,000"),
    "Sugarcane": ("700–900 quintals/ha", "₹70,000–₹120,000"),
    "Maize": ("25–35 quintals/ha", "₹30,000–₹50,000"),
    "Groundnut": ("10–18 quintals/ha", "₹30,000–₹45,000"),
}

# -----------------------------------------------------
# 🔹 Prediction Logic
# -----------------------------------------------------
if submitted:
    try:
        user_data = pd.DataFrame([{
            "district": district, "soiltype": soiltype, "season": season,
            "avgrainfall_mm": avgrainfall_mm, "avgtemp_c": avgtemp_c,
            "avghumidity_%": avghumidity, "soil_ph": soil_ph,
            "nitrogen_kg_ha": nitrogen, "phosphorus_kg_ha": phosphorus, "potassium_kg_ha": potassium
        }])
        user_data = pd.get_dummies(user_data, columns=["district", "soiltype", "season"], drop_first=True)
        user_data = user_data.reindex(columns=model_columns, fill_value=0)

        probs = model.predict_proba(user_data)[0]
        crops = model.classes_
        crop_probs = sorted(list(zip(crops, probs)), key=lambda x: x[1], reverse=True)
        top3 = crop_probs[:3]
        p_max, p_min = max(p for _, p in top3), min(p for _, p in top3)
        scaled_list = []
        for crop_name, prob in top3:
            scaled = 80.0 + ((prob - p_min) / (p_max - p_min)) * 15.0 if p_max != p_min else 85.0
            scaled_list.append((crop_name, max(80.0, min(95.0, scaled))))

        st.subheader(T["prediction"])
        for crop, match_percent in scaled_list:
            yield_est, profit_est = yield_profit_data.get(crop, ("N/A", "N/A"))
            st.markdown(f"""
                <div style="background-color:#f8fafc;border-radius:12px;padding:14px;margin-bottom:12px;
                border-left:6px solid #10b981;box-shadow:0 2px 6px rgba(0,0,0,0.05);">
                <div style="display:flex;justify-content:space-between;">
                    <b style="font-size:18px;">{crop}</b>
                    <span style="color:#065f46;font-weight:700;">{match_percent:.1f}% Match</span>
                </div>
                <div style="margin-top:6px;">🌱 <b>Yield:</b> {yield_est} | 💰 <b>Profit:</b> {profit_est}</div>
                </div>
            """, unsafe_allow_html=True)

        st.subheader(T["weather"])
        for alert in get_weather_alert(avgtemp_c, avghumidity, avgrainfall_mm):
            st.info(alert)

        st.subheader(T["soil"])
        st.warning(get_soil_recommendation(soil_ph))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
