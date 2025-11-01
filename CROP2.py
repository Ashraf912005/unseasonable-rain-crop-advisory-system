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
        "title": "🌾 AI-Powered Maharashtra Crop Recommendation System",
        "desc": "Enter your soil and weather conditions below to get crop recommendations, yield & weather alerts.",
        "form_title": "🧾 Enter Farm Details",
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
        "prediction": "🌾 Crop Recommendations",
        "weather": "🌦️ Weather Alerts",
        "soil": "🌱 Soil Recommendation",
        "alerts": {
            "unseasonal": "⚠️ Unseasonal Rain — delay irrigation and fertilizer use, ensure drainage, and protect harvest.",
            "favorable": "✅ Weather looks favorable for most crops.",
            "hot": "🌡️ High temperature — apply mulching and irrigate adequately.",
            "cold": "❄️ Low temperature — delay sowing or use tolerant varieties.",
            "humid": "💧 High humidity — risk of fungal infection.",
            "dry": "🔥 Low humidity — increase irrigation frequency.",
            "low_rain": "🌤️ Low rainfall — prefer drought-resistant crops like Bajra or Tur."
        },
        "soil_text": {
            "acidic": "Add lime to reduce soil acidity.",
            "alkaline": "Add gypsum or organic matter for alkaline soil.",
            "ideal": "Soil pH is ideal — maintain organic content."
        }
    },
    "हिंदी": {
        "title": "🌾 एआई संचालित महाराष्ट्र फसल सिफारिश प्रणाली",
        "desc": "अपनी मिट्टी और मौसम की जानकारी दर्ज करें और फसल सुझाव, उपज व मौसम चेतावनी प्राप्त करें।",
        "form_title": "🧾 खेत का विवरण दर्ज करें",
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
        "prediction": "🌾 फसल सिफारिशें",
        "weather": "🌦️ मौसम चेतावनी",
        "soil": "🌱 मिट्टी सिफारिश",
        "alerts": {
            "unseasonal": "⚠️ असमय वर्षा — उर्वरक और सिंचाई रोकें, निकासी सुनिश्चित करें और फसल को नमी से बचाएं।",
            "favorable": "✅ मौसम अधिकांश फसलों के लिए अनुकूल है।",
            "hot": "🌡️ अधिक तापमान — मल्चिंग करें और सिंचाई बढ़ाएं।",
            "cold": "❄️ कम तापमान — बुवाई में देरी करें या सहनशील किस्में अपनाएं।",
            "humid": "💧 अधिक नमी — फफूंदी रोग का खतरा।",
            "dry": "🔥 कम नमी — सिंचाई की आवृत्ति बढ़ाएं।",
            "low_rain": "🌤️ कम वर्षा — बाजरा या तूर जैसी फसलें लगाएं।"
        },
        "soil_text": {
            "acidic": "मिट्टी की अम्लता कम करने के लिए चुना डालें।",
            "alkaline": "क्षारीय मिट्टी के लिए जिप्सम या जैविक पदार्थ डालें।",
            "ideal": "मिट्टी का pH आदर्श है — जैविक सामग्री बनाए रखें।"
        }
    },
    "मराठी": {
        "title": "🌾 एआय आधारित महाराष्ट्र पिक शिफारस प्रणाली",
        "desc": "आपली माती व हवामान माहिती भरा आणि पिक सल्ला, उत्पादन व हवामान सूचना मिळवा.",
        "form_title": "🧾 शेताची माहिती भरा",
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
        "prediction": "🌾 पिक शिफारसी",
        "weather": "🌦️ हवामान चेतावणी",
        "soil": "🌱 माती सल्ला",
        "alerts": {
            "unseasonal": "⚠️ अवकाळी पाऊस — खत टाकणे आणि सिंचन थांबवा, निचरा करा आणि पिकाचे संरक्षण करा.",
            "favorable": "✅ हवामान बहुतेक पिकांसाठी अनुकूल आहे.",
            "hot": "🌡️ जास्त तापमान — मल्चिंग करा आणि सिंचन वाढवा.",
            "cold": "❄️ कमी तापमान — पेरणी उशिरा करा किंवा सहनशील वाण वापरा.",
            "humid": "💧 जास्त आर्द्रता — बुरशीजन्य रोगाचा धोका.",
            "dry": "🔥 कमी आर्द्रता — सिंचन वाढवा.",
            "low_rain": "🌤️ कमी पाऊस — बाजरी किंवा तूर लागवड करा."
        },
        "soil_text": {
            "acidic": "मातीची आम्लता कमी करण्यासाठी चुना वापरा.",
            "alkaline": "अल्कलाइन मातीसाठी जिप्सम किंवा सेंद्रिय खत वापरा.",
            "ideal": "मातीचा pH योग्य आहे — सेंद्रिय पदार्थ टिकवा."
        }
    }
}

# -----------------------------------------------------
# 🌐 Language Selector
# -----------------------------------------------------
lang_choice = st.sidebar.radio("Choose Language / भाषा चुनें / भाषा निवडा:", list(LANGUAGES.keys()))
T = LANGUAGES[lang_choice]

# -----------------------------------------------------
# 🌾 Multilingual Mapping
# -----------------------------------------------------
district_map = {
    "English": {"Hingoli": "Hingoli", "Nashik": "Nashik"},
    "हिंदी": {"हिंगोली": "Hingoli", "नासिक": "Nashik"},
    "मराठी": {"हिंगोली": "Hingoli", "नाशिक": "Nashik"}
}

soil_map = {
    "English": {"Black Soil": "Black Soil", "Alluvial Soil": "Alluvial Soil"},
    "हिंदी": {"काली मिट्टी": "Black Soil", "जलोढ़ मिट्टी": "Alluvial Soil"},
    "मराठी": {"काळी माती": "Black Soil", "आलुवीयल माती": "Alluvial Soil"}
}

season_map = {
    "English": {"Kharif": "Kharif", "Rabi": "Rabi"},
    "हिंदी": {"खरीफ": "Kharif", "रबी": "Rabi"},
    "मराठी": {"खरीफ": "Kharif", "रबी": "Rabi"}
}

# -----------------------------------------------------
# 🌾 Streamlit UI
# -----------------------------------------------------
st.title(T["title"])
st.write(T["desc"])

with st.form("crop_form"):
    st.subheader(T["form_title"])
    col1, col2, col3 = st.columns(3)
    with col1:
        district_display = st.selectbox(T["district"], list(district_map[lang_choice].keys()))
        district = district_map[lang_choice][district_display]

        soil_display = st.selectbox(T["soiltype"], list(soil_map[lang_choice].keys()))
        soiltype = soil_map[lang_choice][soil_display]

        season_display = st.selectbox(T["season"], list(season_map[lang_choice].keys()))
        season = season_map[lang_choice][season_display]

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
        alerts.append(T["alerts"]["🌡️ High temperature — heat stress risk."])
    elif temp < 15:
        alerts.append(T["alerts"]["❄️ Low temperature — slow growth expected."])
    if humidity > 85:
        alerts.append(T["alerts"]["💧 High humidity — possible fungal risk."])
    elif humidity < 30:
        alerts.append(T["alerts"]["🔥 Low humidity — frequent irrigation needed."])
    if rainfall < 400:
        alerts.append(T["alerts"]["🌤️ Low rainfall — use drought-tolerant crops."])
    if not alerts:
        alerts.append(T["alerts"]["✅ Weather looks favorable for most crops."])
    return alerts

def get_soil_recommendation(ph):
    if ph < 6:
        return T["soil_text"]["acidic"]
    elif ph > 8:
        return T["soil_text"]["alkaline"]
    else:
        return T["soil_text"]["ideal"]

# -----------------------------------------------------
# 🔹 Prediction Logic
# -----------------------------------------------------
if submitted:
    try:
        user_data = pd.DataFrame([{
            "district": district,
            "soiltype": soiltype,
            "season": season,
            "avgrainfall_mm": avgrainfall_mm,
            "avgtemp_c": avgtemp_c,
            "avghumidity_%": avghumidity,
            "soil_ph": soil_ph,
            "nitrogen_kg_ha": nitrogen,
            "phosphorus_kg_ha": phosphorus,
            "potassium_kg_ha": potassium
        }])
        user_data = pd.get_dummies(user_data, columns=["district", "soiltype", "season"], drop_first=True)
        user_data = user_data.reindex(columns=model_columns, fill_value=0)

        probs = model.predict_proba(user_data)[0]
        crops = model.classes_
        crop_probs = sorted(list(zip(crops, probs)), key=lambda x: x[1], reverse=True)
        top3 = crop_probs[:3]
        p_max, p_min = max(p for _, p in top3), min(p for _, p in top3)
        scaled_list = [(c, 80 + (p - p_min) / (p_max - p_min) * 15 if p_max != p_min else 85) for c, p in top3]

        st.subheader(T["prediction"])
        for crop, match_percent in scaled_list:
            st.markdown(
                f"<div style='background:#f1f5f9;padding:12px;border-left:5px solid #10b981;border-radius:8px;margin:8px 0;'>"
                f"<b>{crop}</b> — {match_percent:.1f}% match</div>", unsafe_allow_html=True)

        st.subheader(T["weather"])
        for alert in get_weather_alert(avgtemp_c, avghumidity, avgrainfall_mm):
            st.info(alert)

        st.subheader(T["soil"])
        st.warning(get_soil_recommendation(soil_ph))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
