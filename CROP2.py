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


district_map = {
    "English": {
        "Ahmednagar": "Ahmednagar", "Akola": "Akola", "Amravati": "Amravati", "Aurangabad": "Aurangabad",
        "Beed": "Beed", "Bhandara": "Bhandara", "Buldhana": "Buldhana", "Chandrapur": "Chandrapur",
        "Dhule": "Dhule", "Gadchiroli": "Gadchiroli", "Gondia": "Gondia", "Hingoli": "Hingoli",
        "Jalgaon": "Jalgaon", "Jalna": "Jalna", "Kolhapur": "Kolhapur", "Latur": "Latur",
        "Mumbai City": "Mumbai City", "Mumbai Suburban": "Mumbai Suburban", "Nagpur": "Nagpur", "Nanded": "Nanded",
        "Nandurbar": "Nandurbar", "Nashik": "Nashik", "Osmanabad": "Osmanabad", "Palghar": "Palghar",
        "Parbhani": "Parbhani", "Pune": "Pune", "Raigad": "Raigad", "Ratnagiri": "Ratnagiri",
        "Sangli": "Sangli", "Satara": "Satara", "Sindhudurg": "Sindhudurg", "Solapur": "Solapur",
        "Thane": "Thane", "Wardha": "Wardha", "Washim": "Washim", "Yavatmal": "Yavatmal"
    },
    "हिंदी": {
        "अहमदनगर": "Ahmednagar", "अकोला": "Akola", "अमरावती": "Amravati", "औरंगाबाद": "Aurangabad",
        "बीड़": "Beed", "भंडारा": "Bhandara", "बुलढाणा": "Buldhana", "चंद्रपुर": "Chandrapur",
        "धुले": "Dhule", "गडचिरोली": "Gadchiroli", "गोंदिया": "Gondia", "हिंगोली": "Hingoli",
        "जलगांव": "Jalgaon", "जालना": "Jalna", "कोल्हापुर": "Kolhapur", "लातूर": "Latur",
        "मुंबई सिटी": "Mumbai City", "मुंबई उपनगर": "Mumbai Suburban", "नागपुर": "Nagpur", "नांदेड़": "Nanded",
        "नंदुरबार": "Nandurbar", "नासिक": "Nashik", "उस्मानाबाद": "Osmanabad", "पालघर": "Palghar",
        "परभणी": "Parbhani", "पुणे": "Pune", "रायगढ़": "Raigad", "रत्नागिरी": "Ratnagiri",
        "सांगली": "Sangli", "सातारा": "Satara", "सिंधुदुर्ग": "Sindhudurg", "सोलापुर": "Solapur",
        "ठाणे": "Thane", "वर्धा": "Wardha", "वाशीम": "Washim", "यवतमाल": "Yavatmal"
    },
    "मराठी": {
        "अहमदनगर": "Ahmednagar", "अकोला": "Akola", "अमरावती": "Amravati", "औरंगाबाद": "Aurangabad",
        "बीड": "Beed", "भंडारा": "Bhandara", "बुलढाणा": "Buldhana", "चंद्रपूर": "Chandrapur",
        "धुळे": "Dhule", "गडचिरोली": "Gadchiroli", "गोंदिया": "Gondia", "हिंगोली": "Hingoli",
        "जळगाव": "Jalgaon", "जालना": "Jalna", "कोल्हापूर": "Kolhapur", "लातूर": "Latur",
        "मुंबई सिटी": "Mumbai City", "मुंबई उपनगर": "Mumbai Suburban", "नागपूर": "Nagpur", "नांदेड": "Nanded",
        "नंदुरबार": "Nandurbar", "नाशिक": "Nashik", "उस्मानाबाद": "Osmanabad", "पालघर": "Palghar",
        "परभणी": "Parbhani", "पुणे": "Pune", "रायगड": "Raigad", "रत्नागिरी": "Ratnagiri",
        "सांगली": "Sangli", "सातारा": "Satara", "सिंधुदुर्ग": "Sindhudurg", "सोलापूर": "Solapur",
        "ठाणे": "Thane", "वर्धा": "Wardha", "वाशीम": "Washim", "यवतमाळ": "Yavatmal"
    }
}

soil_map = {
    "English": {
        "Black Soil": "Black Soil",
        "Alluvial Soil": "Alluvial Soil",
        "Red Soil": "Red Soil",
        "Laterite Soil": "Laterite Soil",
        "Sandy Soil": "Sandy Soil"
    },
    "हिंदी": {
        "काली मिट्टी": "Black Soil",
        "जलोढ़ मिट्टी": "Alluvial Soil",
        "लाल मिट्टी": "Red Soil",
        "लेटेराइट मिट्टी": "Laterite Soil",
        "रेतीली मिट्टी": "Sandy Soil"
    },
    "मराठी": {
        "काळी माती": "Black Soil",
        "आलुवीयल माती": "Alluvial Soil",
        "लाल माती": "Red Soil",
        "लेटेराइट माती": "Laterite Soil",
        "वालुकामय माती": "Sandy Soil"
    }
}

season_map = {
    "English": {"Kharif": "Kharif", "Rabi": "Rabi", "Zaid": "Zaid"},
    "हिंदी": {"खरीफ": "Kharif", "रबी": "Rabi", "जायद": "Zaid"},
    "मराठी": {"खरीफ": "Kharif", "रबी": "Rabi", "जायद": "Zaid"}
}


# -----------------------------------------------------
# 🌾 Streamlit UI with Seasonal Rainfall Guide
# -----------------------------------------------------
st.title(T["title"])
st.write(T["desc"])

# Layout: form on left, rainfall info card on right
col_form, col_info = st.columns([2, 1])

with col_form:
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
# 🌧️ General Rainfall Reference (Dynamic Language)
# -----------------------------------------------------
rainfall_reference = {
    "English": {
        "Kharif": [
            ("Low", "Below 400 mm"),
            ("Moderate", "400 – 1200 mm"),
            ("High", "Above 1200 mm")
        ],
        "Rabi": [
            ("Low", "Below 50 mm"),
            ("Moderate", "50 – 150 mm"),
            ("High", "Above 150 mm (Unseasonal)")
        ],
        "Zaid": [
            ("Low", "Below 25 mm"),
            ("Moderate", "25 – 100 mm"),
            ("High", "Above 100 mm (Unseasonal)")
        ],
        "title": "💧 Rainfall Reference (per season)",
        "note": "Typical rainfall range for Maharashtra regions."
    },
    "हिंदी": {
        "खरीफ": [
            ("कम", "400 मिमी से कम"),
            ("मध्यम", "400 – 1200 मिमी"),
            ("ज्यादा", "1200 मिमी से अधिक")
        ],
        "रबी": [
            ("कम", "50 मिमी से कम"),
            ("मध्यम", "50 – 150 मिमी"),
            ("ज्यादा", "150 मिमी से अधिक (असमय वर्षा)")
        ],
        "जायद": [
            ("कम", "25 मिमी से कम"),
            ("मध्यम", "25 – 100 मिमी"),
            ("ज्यादा", "100 मिमी से अधिक (असमय वर्षा)")
        ],
        "title": "💧 वर्षा संदर्भ (प्रति मौसम)",
        "note": "महाराष्ट्र के लिए सामान्य वर्षा सीमा।"
    },
    "मराठी": {
        "खरीफ": [
            ("कमी", "400 मिमी पेक्षा कमी"),
            ("मध्यम", "400 – 1200 मिमी"),
            ("जास्त", "1200 मिमी पेक्षा अधिक")
        ],
        "रबी": [
            ("कमी", "50 मिमी पेक्षा कमी"),
            ("मध्यम", "50 – 150 मिमी"),
            ("जास्त", "150 मिमी पेक्षा अधिक (अवकाळी पाऊस)")
        ],
        "जायद": [
            ("कमी", "25 मिमी पेक्षा कमी"),
            ("मध्यम", "25 – 100 मिमी"),
            ("जास्त", "100 मिमी पेक्षा अधिक (अवकाळी पाऊस)")
        ],
        "title": "💧 पर्जन्यमान संदर्भ (प्रति हंगाम)",
        "note": "महाराष्ट्रातील सामान्य पर्जन्यमान श्रेणी."
    }
}

# -----------------------------------------------------
# 💧 Rainfall Info Card beside the input form
# -----------------------------------------------------
with col_info:
    ref = rainfall_reference[lang_choice]
    
    # Detect season keys dynamically based on language
    if lang_choice == "English":
        kharif_key, rabi_key, zaid_key = "Kharif", "Rabi", "Zaid"
    elif lang_choice == "हिंदी":
        kharif_key, rabi_key, zaid_key = "खरीफ", "रबी", "जायद"
    else:  # मराठी
        kharif_key, rabi_key, zaid_key = "खरीफ", "रबी", "जायद"

    st.markdown(f"""
        <div style="
            background-color:#000000;
            padding:16px;
            border-radius:12px;
            box-shadow:0 2px 6px rgba(0,0,0,0.08);
            border-left:6px solid #0ea5e9;
            width: 105%;  /* ⬅️ slightly wider than container */
            ">
            <h4 style="margin-bottom:10px;">{ref['title']}</h4>
            <div style="font-size:14px; margin-bottom:8px;">
                <b>🌾 {kharif_key}:</b><br>
                • {ref[kharif_key][0][0]} – {ref[kharif_key][0][1]}<br>
                • {ref[kharif_key][1][0]} – {ref[kharif_key][1][1]}<br>
                • {ref[kharif_key][2][0]} – {ref[kharif_key][2][1]}<br><br>
                <b>🌾 {rabi_key}:</b><br>
                • {ref[rabi_key][0][0]} – {ref[rabi_key][0][1]}<br>
                • {ref[rabi_key][1][0]} – {ref[rabi_key][1][1]}<br>
                • {ref[rabi_key][2][0]} – {ref[rabi_key][2][1]}<br><br>
                <b>🌾 {zaid_key}:</b><br>
                • {ref[zaid_key][0][0]} – {ref[zaid_key][0][1]}<br>
                • {ref[zaid_key][1][0]} – {ref[zaid_key][1][1]}<br>
                • {ref[zaid_key][2][0]} – {ref[zaid_key][2][1]}
            </div>
            <p style="font-size:12px;color:#ffffff;margin-top:10px;">ℹ️ {ref['note']}</p>
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------
# ⚙️ Helper Functions
# -----------------------------------------------------
def get_weather_alert(season, temp, humidity, rainfall):
    alerts = []
    # Seasonal rainfall expectations (approx)
    seasonal_rain_limits = {
        "Kharif": (400, 1200),   # typical Kharif rain range
        "Rabi": (0, 150),        # Rabi is mostly dry
        "Zaid": (0, 100)      
    }

    # Get season-specific range
    min_rain, max_rain = seasonal_rain_limits.get(season, (0, 1200))

    # ✅ Unseasonal rain condition: too much rain for a dry season
    if rainfall > max_rain + 50:  # Allow a small margin
        alerts.append(T["alerts"]["unseasonal"])

    # Temperature alerts
    if temp > 35:
        alerts.append(T["alerts"]["hot"])
    elif temp < 15:
        alerts.append(T["alerts"]["cold"])

    # Humidity alerts
    if humidity > 85:
        alerts.append(T["alerts"]["humid"])
    elif humidity < 30:
        alerts.append(T["alerts"]["dry"])

    # Low rainfall (for Kharif mostly)
    if rainfall < min_rain:
        alerts.append(T["alerts"]["low_rain"])

    # Default message if no major issues
    if not alerts:
        alerts.append(T["alerts"]["favorable"])

    return alerts



def get_soil_recommendation(ph):
    if ph < 6:
        return T["soil_text"]["acidic"]
    elif ph > 8:
        return T["soil_text"]["alkaline"]
    else:
        return T["soil_text"]["ideal"]
    
yield_profit_data = {
    "Cotton": ("8-12 quintals/hectare", "₹30,000-₹45,000"),        # realistic maybe 8-12
    "Soybean": ("15-25 quintals/hectare", "₹30,000-₹50,000"),     # revise down a bit
    "Tur": ("6-12 quintals/hectare", "₹20,000-₹30,000"),
    "Wheat": ("30-40 quintals/hectare", "₹50,000-₹70,000"),
    "Jowar": ("10-20 quintals/hectare", "₹20,000-₹35,000"),
    "Rice": ("30-45 quintals/hectare", "₹40,000-₹60,000"),
    "Gram": ("8-15 quintals/hectare", "₹25,000-₹40,000"),
    "Sugarcane": ("700-900 quintals/hectare", "₹70,000-₹120,000"),
    "Maize": ("25-35 quintals/hectare", "₹30,000-₹50,000"),
    "Groundnut": ("10-18 quintals/hectare", "₹30,000-₹45,000"),
}

# -----------------------------------------------------
# 🌾 Multilingual Crop Names
# -----------------------------------------------------
crop_names = {
    "English": {
        "Cotton": "Cotton",
        "Soybean": "Soybean",
        "Tur": "Tur (Pigeon Pea)",
        "Wheat": "Wheat",
        "Jowar": "Jowar (Sorghum)",
        "Rice": "Rice",
        "Gram": "Gram (Chickpea)",
        "Sugarcane": "Sugarcane",
        "Maize": "Maize (Corn)",
        "Groundnut": "Groundnut (Peanut)"
    },
    "हिंदी": {
        "Cotton": "कपास",
        "Soybean": "सोयाबीन",
        "Tur": "तूर (अरहर)",
        "Wheat": "गेहूं",
        "Jowar": "ज्वार",
        "Rice": "चावल",
        "Gram": "चना",
        "Sugarcane": "गन्ना",
        "Maize": "मक्का",
        "Groundnut": "मूंगफली"
    },
    "मराठी": {
        "Cotton": "कापूस",
        "Soybean": "सोयाबीन",
        "Tur": "तूर (अरहर)",
        "Wheat": "गहू",
        "Jowar": "ज्वारी",
        "Rice": "तांदूळ",
        "Gram": "हरभरा",
        "Sugarcane": "ऊस",
        "Maize": "मका",
        "Groundnut": "शेंगदाणे"
    }
}


# -----------------------------------------------------
# 🔹 Prediction Logic
# -----------------------------------------------------
if submitted:
    try:
        # Prepare user input
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

        # Preprocess for model
        user_data = pd.get_dummies(user_data, columns=["district", "soiltype", "season"], drop_first=True)
        user_data = user_data.reindex(columns=model_columns, fill_value=0)

        # Predict
        probs = model.predict_proba(user_data)[0]
        crops = model.classes_
        crop_probs = sorted(list(zip(crops, probs)), key=lambda x: x[1], reverse=True)

# 🎯 Filter predictions by season
        valid_crops_by_season = {
            "Kharif": ["Cotton", "Soybean", "Tur", "Jowar", "Rice", "Maize", "Groundnut"],
            "Rabi": ["Wheat", "Gram", "Jowar", "Tur"],
            "Zaid": ["Maize", "Groundnut", "Sugarcane"]
        }

        valid_crops = valid_crops_by_season.get(season, [])
        filtered_top3 = [(crop, prob) for crop, prob in crop_probs if crop in valid_crops]

        if not filtered_top3:
            filtered_top3 = crop_probs[:3]

        top3 = filtered_top3[:3]

        # Scale probabilities into readable match %
        raw_vals = [p for _, p in top3]
        p_max, p_min = max(raw_vals), min(raw_vals)
        scaled_list = []
        if p_max == p_min:
            fixed = [90.0, 85.0, 80.0]
            for (cp, _), s in zip(top3, fixed):
                scaled_list.append((cp, s))
        else:
            for crop_name, prob in top3:
                scaled = 80.0 + ((prob - p_min) / (p_max - p_min)) * 10.0
                scaled = max(80.0, min(95.0, scaled))
                scaled_list.append((crop_name, scaled))

        # -----------------------------------------------------
        # 🌾 Multilingual Prediction Results Section
        # -----------------------------------------------------
        result_title = {
            "English": "🌾 Crop Recommendations",
            "हिंदी": "🌾 फसल सिफारिशें",
            "मराठी": "🌾 पिक शिफारसी"
        }[lang_choice]

        yield_label = {
            "English": "Expected Yield",
            "हिंदी": "अनुमानित उपज",
            "मराठी": "अपेक्षित उत्पादन"
        }[lang_choice]

        profit_label = {
            "English": "Estimated Profit",
            "हिंदी": "अनुमानित लाभ",
            "मराठी": "अंदाजे नफा"
        }[lang_choice]

        match_label = {
            "English": "% Match",
            "हिंदी": "% मेल",
            "मराठी": "% जुळणारे"
        }[lang_choice]

        # Show multilingual prediction results
        st.subheader(result_title)
        for i, (crop, match_percent) in enumerate(scaled_list):
            crop_display = crop_names[lang_choice].get(crop, crop)
            yield_est, profit_est = yield_profit_data.get(crop, ("N/A", "N/A"))
            card_bg = "#ffffff" if i % 2 == 0 else "#FEFDFD"
            st.markdown(f"""
                <div style="background-color:{card_bg};
                    border-radius:10px;padding:14px;margin-bottom:12px;
                    border-left:6px solid #10b981;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
                    color: #0f172a;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-size:18px;font-weight:600;">{crop_display}</div>
                        <div style="font-size:16px;color:#065f46;font-weight:700;">{match_percent:.1f}{match_label}</div>
                    </div>
                    <div style="margin-top:8px;font-size:14px;color:#0f172a;">
                        🌾 <b>{yield_label}:</b> {yield_est} &nbsp;&nbsp; | &nbsp;&nbsp; 💰 <b>{profit_label}:</b> {profit_est}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # -----------------------------------------------------
        # 🌦️ Weather Alerts (Multilingual)
        # -----------------------------------------------------
        st.subheader(T["weather"])
        alerts = get_weather_alert(season, avgtemp_c, avghumidity, avgrainfall_mm)
        for alert in alerts:
            st.info(alert)

        # -----------------------------------------------------
        # 🌱 Soil Recommendation (Multilingual)
        # -----------------------------------------------------
        st.subheader(T["soil"])
        st.warning(get_soil_recommendation(soil_ph))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
