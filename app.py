import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import date


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flight_delay_model.pkl"
AIRPORTS_PATH = BASE_DIR / "data" / "airports.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid rgba(120, 120, 120, 0.25);
        border-radius: 18px;
        margin-bottom: 1.8rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.6rem;
    }

    .hero p {
        margin-top: 0.7rem;
        margin-bottom: 0;
        opacity: 0.75;
        font-size: 1.05rem;
    }

    .hero-tags {
        margin-top: 1.1rem;
    }

    .tag {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
        border-radius: 20px;
        border: 1px solid rgba(120, 120, 120, 0.3);
        font-size: 0.82rem;
        opacity: 0.85;
    }

    .route-card {
        padding: 1rem 1.3rem;
        border: 1px solid rgba(120, 120, 120, 0.25);
        border-radius: 14px;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
    }

    .route-code {
        font-size: 1.6rem;
        font-weight: 700;
    }

    .route-details {
        opacity: 0.7;
        margin-top: 0.2rem;
    }

    .result-card {
        padding: 1.5rem 1.7rem;
        border-radius: 16px;
        border: 1px solid rgba(120, 120, 120, 0.25);
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }

    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .result-subtitle {
        opacity: 0.75;
        font-size: 1rem;
    }

    .footer {
        text-align: center;
        opacity: 0.55;
        font-size: 0.85rem;
        padding-top: 1.5rem;
    }

    div.stButton > button {
        height: 3.2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:

    package = load_model()

    model = package["model"]

    airline_delay_map = package["airline_delay_rates"]
    origin_delay_map = package["origin_delay_rates"]
    destination_delay_map = package["destination_delay_rates"]

    global_rate = package["global_delay_rate"]

    origin_traffic_map = package["origin_traffic"]
    destination_traffic_map = package["destination_traffic"]
    route_frequency_map = package["route_frequency"]

    numeric_features = package["numeric_features"]
    categorical_features = package["categorical_features"]

except Exception as e:

    st.error("The prediction model could not be loaded.")
    st.exception(e)
    st.stop()


# ============================================================
# LOAD AIRPORT INFORMATION
# ============================================================

@st.cache_data
def load_airports():

    df = pd.read_csv(AIRPORTS_PATH)

    df["IATA_CODE"] = (
        df["IATA_CODE"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return df


try:

    airports_df = load_airports()

    airport_info = (
        airports_df
        .set_index("IATA_CODE")[
            ["AIRPORT", "CITY", "STATE"]
        ]
        .to_dict("index")
    )

except Exception as e:

    st.error("Airport information could not be loaded.")
    st.exception(e)
    st.stop()


# ============================================================
# AIRLINE INFORMATION
# ============================================================

AIRLINE_NAMES = {
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "EV": "Atlantic Southeast Airlines",
    "F9": "Frontier Airlines",
    "HA": "Hawaiian Airlines",
    "MQ": "American Eagle",
    "NK": "Spirit Airlines",
    "OO": "SkyWest Airlines",
    "UA": "United Airlines",
    "US": "US Airways",
    "VX": "Virgin America",
    "WN": "Southwest Airlines"
}


airline_options = sorted(
    str(x)
    for x in airline_delay_map.keys()
)


origin_options = sorted([
    str(x).upper()
    for x in origin_traffic_map.keys()
    if str(x).isalpha()
    and len(str(x)) == 3
])


destination_options = sorted([
    str(x).upper()
    for x in destination_traffic_map.keys()
    if str(x).isalpha()
    and len(str(x)) == 3
])


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_airline(code):

    return (
        f"{code} — "
        f"{AIRLINE_NAMES.get(code, code)}"
    )


def format_airport(code):

    info = airport_info.get(code)

    if info is None:
        return code

    city = info.get("CITY", "")
    airport = info.get("AIRPORT", "")

    return (
        f"{code} — {city} | {airport}"
    )


def airport_city(code):

    info = airport_info.get(code, {})

    return info.get(
        "CITY",
        code
    )


# ============================================================
# FEATURE HELPERS
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Spring"

    if month in [6, 7, 8]:
        return "Summer"

    return "Autumn"


def get_time_of_day(hour):

    if 5 <= hour < 12:
        return "Morning"

    if 12 <= hour < 17:
        return "Afternoon"

    if 17 <= hour < 21:
        return "Evening"

    return "Night"


def get_distance_category(distance):

    if distance <= 500:
        return "Short"

    if distance <= 1000:
        return "Medium"

    if distance <= 2000:
        return "Long"

    return "Very Long"


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
<h1> Flight Delay Predictor</h1>
<p>Estimate the likelihood of a flight delay using schedule, route, airport traffic, historical patterns and optional weather information.</p>
<div class="hero-tags">
<span class="tag">Advanced XGBoost</span>
<span class="tag">294K+ flights</span>
<span class="tag">40 predictive features</span>
<span class="tag">Weather-aware</span>
</div>
</div>
""",
    unsafe_allow_html=True
)

# ============================================================
# FLIGHT DETAILS
# ============================================================

st.header("Flight Details")

st.caption(
    "Enter the scheduled flight information to calculate "
    "its estimated delay risk."
)


col1, col2, col3 = st.columns(3)


with col1:

    airline = st.selectbox(
        "Airline",
        airline_options,
        format_func=format_airline
    )


with col2:

    origin_airport = st.selectbox(
        "Origin Airport",
        origin_options,
        format_func=format_airport
    )


with col3:

    destination_airport = st.selectbox(
        "Destination Airport",
        destination_options,
        format_func=format_airport,
        index=(
            1
            if len(destination_options) > 1
            else 0
        )
    )


col1, col2, col3 = st.columns(3)


with col1:

    flight_date = st.date_input(
        "Flight Date",
        value=date.today()
    )


with col2:

    departure_time = st.time_input(
        "Scheduled Departure"
    )


with col3:

    distance = st.number_input(
        "Flight Distance (miles)",
        min_value=1,
        max_value=6000,
        value=1000,
        step=10
    )


# ============================================================
# ROUTE SUMMARY
# ============================================================

origin_city = airport_city(
    origin_airport
)

destination_city = airport_city(
    destination_airport
)


st.markdown(
    f"""
<div class="route-card">
<div class="route-code">{origin_airport} &nbsp; → &nbsp; {destination_airport}</div>
<div class="route-details">{origin_city} → {destination_city} &nbsp; • &nbsp; {distance:,} miles &nbsp; • &nbsp; {departure_time.strftime("%H:%M")}</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# WEATHER
# ============================================================

with st.expander(
    " Add weather information (optional)"
):

    st.caption(
        "Weather data can provide additional context. "
        "The model can still make a prediction without it."
    )

    weather_available = st.checkbox(
        "Use weather information"
    )


    temperature = None
    humidity = None
    pressure = None
    wind_speed = None
    weather_description = "Unknown"


    if weather_available:

        col1, col2 = st.columns(2)


        with col1:

            temperature_c_input = (
                st.number_input(
                    "Temperature (°C)",
                    min_value=-50.0,
                    max_value=60.0,
                    value=20.0,
                    step=0.5
                )
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0
            )


        with col2:

            pressure = st.number_input(
                "Pressure",
                min_value=800.0,
                max_value=1100.0,
                value=1013.0,
                step=1.0
            )

            wind_speed = st.number_input(
                "Wind Speed",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.5
            )


        weather_description = st.selectbox(
            "Weather Condition",
            [
                "clear sky",
                "few clouds",
                "scattered clouds",
                "broken clouds",
                "overcast clouds",
                "mist",
                "fog",
                "haze",
                "light rain",
                "moderate rain",
                "heavy rain",
                "drizzle",
                "snow",
                "thunderstorm"
            ]
        )


        # Training temperature data uses Kelvin.

        temperature = (
            temperature_c_input
            + 273.15
        )


# ============================================================
# DATE FEATURES
# ============================================================

month = flight_date.month
day = flight_date.day

day_of_week = (
    flight_date.weekday()
    + 1
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_flight_delay():

    departure_hour = (
        departure_time.hour
    )

    departure_minute = (
        departure_time.minute
    )


    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    quarter = (
        ((month - 1) // 3)
        + 1
    )

    weekend = int(
        day_of_week in [6, 7]
    )

    peak_hour = int(
        departure_hour in [
            6, 7, 8, 9,
            16, 17, 18, 19
        ]
    )

    season = get_season(
        month
    )

    time_of_day = get_time_of_day(
        departure_hour
    )

    distance_category = (
        get_distance_category(
            distance
        )
    )


    # --------------------------------------------------------
    # TRAFFIC
    # --------------------------------------------------------

    origin_traffic = (
        origin_traffic_map.get(
            str(origin_airport),
            0
        )
    )

    destination_traffic = (
        destination_traffic_map.get(
            str(destination_airport),
            0
        )
    )


    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    route = (
        str(origin_airport)
        + "_"
        + str(destination_airport)
    )

    route_frequency = (
        route_frequency_map.get(
            route,
            0
        )
    )


    # --------------------------------------------------------
    # HISTORICAL DELAY RATES
    # --------------------------------------------------------

    airline_delay_rate = (
        airline_delay_map.get(
            str(airline),
            global_rate
        )
    )

    origin_delay_rate = (
        origin_delay_map.get(
            str(origin_airport),
            global_rate
        )
    )

    destination_delay_rate = (
        destination_delay_map.get(
            str(destination_airport),
            global_rate
        )
    )


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    weather_flag = int(
        weather_available
    )


    temperature_value = (
        np.nan
        if temperature is None
        else float(temperature)
    )


    humidity_value = (
        np.nan
        if humidity is None
        else float(humidity)
    )


    pressure_value = (
        np.nan
        if pressure is None
        else float(pressure)
    )


    wind_value = (
        np.nan
        if wind_speed is None
        else float(wind_speed)
    )


    temperature_c = (
        np.nan
        if np.isnan(temperature_value)
        else temperature_value - 273.15
    )


    freezing = int(
        not np.isnan(temperature_c)
        and temperature_c <= 0
    )


    very_hot = int(
        not np.isnan(temperature_c)
        and temperature_c >= 35
    )


    high_wind = int(
        not np.isnan(wind_value)
        and wind_value >= 10
    )


    weather_text = (
        str(weather_description)
        .lower()
    )


    low_visibility = int(
        any(
            word in weather_text
            for word in [
                "fog",
                "mist",
                "haze"
            ]
        )
    )


    storm_weather = int(
        any(
            word in weather_text
            for word in [
                "storm",
                "thunder",
                "squall"
            ]
        )
    )


    rain_snow = int(
        any(
            word in weather_text
            for word in [
                "rain",
                "snow",
                "drizzle"
            ]
        )
    )


    # --------------------------------------------------------
    # INTERACTION FEATURES
    # --------------------------------------------------------

    peak_origin_traffic = (
        peak_hour
        * origin_traffic
    )

    peak_destination_traffic = (
        peak_hour
        * destination_traffic
    )

    distance_origin_traffic = (
        distance
        * origin_traffic
    )

    distance_destination_traffic = (
        distance
        * destination_traffic
    )

    hour_squared = (
        departure_hour ** 2
    )

    weekend_peak = (
        weekend
        * peak_hour
    )


    # --------------------------------------------------------
    # BUILD INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "MONTH": month,
        "DAY": day,
        "DAY_OF_WEEK": day_of_week,

        "Quarter": quarter,
        "Weekend": weekend,

        "AIRLINE": str(airline),

        "ORIGIN_AIRPORT":
            str(origin_airport),

        "DESTINATION_AIRPORT":
            str(destination_airport),

        "DISTANCE":
            float(distance),

        "Departure_Hour":
            departure_hour,

        "Departure_Minute":
            departure_minute,

        "Peak_Hour":
            peak_hour,

        "Time_of_Day":
            time_of_day,

        "Season":
            season,

        "Distance_Category":
            distance_category,

        "Origin_Traffic":
            origin_traffic,

        "Destination_Traffic":
            destination_traffic,

        "Route_Frequency":
            route_frequency,

        "Temperature":
            temperature_value,

        "Humidity":
            humidity_value,

        "Pressure":
            pressure_value,

        "Wind_Speed":
            wind_value,

        "Weather_Description":
            weather_description,

        "Weather_Available":
            weather_flag,

        "Airline_Delay_Rate":
            airline_delay_rate,

        "Origin_Delay_Rate":
            origin_delay_rate,

        "Destination_Delay_Rate":
            destination_delay_rate,

        "Peak_Origin_Traffic":
            peak_origin_traffic,

        "Peak_Destination_Traffic":
            peak_destination_traffic,

        "Distance_Origin_Traffic":
            distance_origin_traffic,

        "Distance_Destination_Traffic":
            distance_destination_traffic,

        "High_Wind":
            high_wind,

        "Low_Visibility_Weather":
            low_visibility,

        "Storm_Weather":
            storm_weather,

        "Rain_Snow":
            rain_snow,

        "Temperature_C":
            temperature_c,

        "Freezing":
            freezing,

        "Very_Hot":
            very_hot,

        "Hour_Squared":
            hour_squared,

        "Weekend_Peak":
            weekend_peak

    }])


    # --------------------------------------------------------
    # EXACT TRAINING FEATURE ORDER
    # --------------------------------------------------------

    expected_features = (
        numeric_features
        + categorical_features
    )


    missing_features = [
        feature
        for feature in expected_features
        if feature not in input_data.columns
    ]


    if missing_features:

        raise ValueError(
            "Missing model features: "
            + str(missing_features)
        )


    input_data = input_data[
        expected_features
    ]


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(
            input_data
        )[0, 1]
    )


    prediction = int(
        probability >= 0.50
    )


    return (
        prediction,
        probability
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "Predict Flight Delay",
    type="primary",
    use_container_width=True
)


# ============================================================
# RESULTS
# ============================================================

if predict_button:

    if (
        origin_airport
        == destination_airport
    ):

        st.error(
            "Please choose different origin "
            "and destination airports."
        )

    else:

        try:

            prediction, probability = (
                predict_flight_delay()
            )


            delay_percent = (
                probability * 100
            )

            on_time_percent = (
                100 - delay_percent
            )


            # ------------------------------------------------
            # RISK CATEGORY
            # ------------------------------------------------

            if probability < 0.35:

                risk = "Low"
                risk_icon = "🟢"

            elif probability < 0.60:

                risk = "Moderate"
                risk_icon = "🟡"

            else:

                risk = "High"
                risk_icon = "🔴"


            st.divider()

            st.header(
                "Prediction"
            )


            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            if prediction == 1:

                result_title = (
                    "Delay Likely"
                )

                result_text = (
                    f"The model estimates a "
                    f"{delay_percent:.1f}% probability "
                    f"of delay for this flight."
                )

            else:

                result_title = (
                    " Likely On Time"
                )

                result_text = (
                    f"The model estimates a "
                    f"{on_time_percent:.1f}% probability "
                    f"that this flight will be on time."
                )


            st.markdown(
                f"""
            <div class="result-card">
            <div class="result-title">{result_title}</div>
            <div class="result-subtitle"><strong>{origin_airport} → {destination_airport}</strong> &nbsp; • &nbsp; {AIRLINE_NAMES.get(airline, airline)}<br><br>{result_text}</div>
            </div>
            """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            col1, col2, col3 = (
                st.columns(3)
            )


            with col1:

                st.metric(
                    "Delay Probability",
                    f"{delay_percent:.1f}%"
                )


            with col2:

                st.metric(
                    "On-Time Probability",
                    f"{on_time_percent:.1f}%"
                )


            with col3:

                st.metric(
                    "Risk Level",
                    f"{risk_icon} {risk}"
                )


            # ------------------------------------------------
            # PROBABILITY BAR
            # ------------------------------------------------

            st.caption(
                "Estimated delay probability"
            )

            st.progress(
                int(
                    min(
                        max(
                            delay_percent,
                            0
                        ),
                        100
                    )
                )
            )


            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            if risk == "Low":

                st.info(
                    "This flight currently shows a relatively "
                    "low predicted delay risk based on the "
                    "available information."
                )


            elif risk == "Moderate":

                st.warning(
                    "This flight has a moderate predicted delay "
                    "risk. Schedule, route traffic or weather "
                    "conditions may increase uncertainty."
                )


            else:

                st.error(
                    "This flight has a relatively high predicted "
                    "delay risk. Consider checking the airline's "
                    "latest operational status before departure."
                )


        except Exception as e:

            st.error(
                "The prediction could not be generated."
            )

            st.exception(e)


# ============================================================
# ABOUT MODEL
# ============================================================

st.divider()


with st.expander(
    "About the Model"
):

    st.markdown(
        f"""
### Advanced XGBoost Flight Delay Model

The application uses a machine-learning model trained on
historical flight information together with engineered
schedule, route, airport-traffic and weather features.

**Model performance**

- Accuracy: `{package["metrics"]["accuracy"]:.3f}`
- Precision: `{package["metrics"]["precision"]:.3f}`
- Recall: `{package["metrics"]["recall"]:.3f}`
- F1 Score: `{package["metrics"]["f1"]:.3f}`
- ROC-AUC: `{package["metrics"]["roc_auc"]:.3f}`
- PR-AUC: `{package["metrics"]["pr_auc"]:.3f}`

**Information considered**

Flight date and departure time, airline, origin and
destination airports, flight distance, historical airport
traffic, route frequency, historical delay patterns,
temporal interactions and optional weather information.

The output is a **probabilistic machine-learning estimate**,
not a real-time airline status or a guarantee that a flight
will or will not be delayed.
        """
    )
# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

with st.expander(" Model Performance & Training Results"):

    st.markdown("### Model Comparison")

    st.caption(
        "Performance of the models evaluated during development. "
        "The Advanced XGBoost model is used for predictions."
    )

    performance_data = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Base XGBoost",
            "Tuned XGBoost",
            "Advanced XGBoost"
        ],
        "Accuracy": [0.6088, 0.6541, 0.6942, 0.6967],
        "Precision": [0.2660, 0.2990, 0.3245, 0.3266],
        "Recall": [0.6239, 0.6362, 0.5914, 0.5905],
        "F1 Score": [0.3729, 0.4068, 0.4190, 0.4206],
        "ROC-AUC": [0.6549, 0.7017, 0.7124, 0.7118],
        "PR-AUC": [0.2935, 0.3620, 0.3835, 0.3805]
    })

    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )
    st.markdown("### Performance Comparison")

    chart_data = performance_data.set_index("Model")[
        ["F1 Score", "ROC-AUC", "PR-AUC"]
    ]

    st.bar_chart(chart_data)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Flight Delay Predictor • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)