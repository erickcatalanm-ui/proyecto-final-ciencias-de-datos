import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- Definición de Constantes ---
TARGET_VARIABLE = 'log_rentability'

NUMERICAL_FEATURES_INITIAL = [
    'accommodates', 'review_scores_rating',
    'calculated_host_listings_count', 'minimum_nights',
    'number_of_reviews'
]

CATEGORICAL_FEATURES_INITIAL = [
    'host_is_superhost', 'room_type', 'property_type'
]

FINAL_R2 = 0.6898
FINAL_RMSE = 1.7111

FEATURE_IMPORTANCE_DATA = {
    'Feature': [
        'number_of_reviews',
        'calculated_host_listings_count',
        'review_scores_rating',
        'minimum_nights',
        'neighbourhood_rentability_mean'
    ],
    'Importance': [0.630, 0.065, 0.058, 0.045, 0.040]
}
IMPORTANCE_DF = pd.DataFrame(FEATURE_IMPORTANCE_DATA)


# -------------------------------------------------------------------
# --- 1. CARGA DE DATOS ---
@st.cache_data
def load_data():

    try:
        df = pd.read_csv('listings.csv')
    except FileNotFoundError:
        st.error("No se encontró listings.csv en el directorio.")
        return pd.DataFrame(), {}

    # Limpieza de precios
    df['price'] = (
        df['price'].astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(float)
    )

    # Limpieza de puntuaciones
    median_score = df['review_scores_rating'].median()
    df['review_scores_rating'].fillna(median_score, inplace=True)

    score_cols = [
        'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value'
    ]
    for col in score_cols:
        df[col].fillna(median_score, inplace=True)

    df.dropna(subset=['price', 'review_scores_rating'], inplace=True)

    # Rentabilidad Proxy Mejorada
    df['occupancy_rate'] = df['estimated_occupancy_l365d'] / 365
    df['rentability_proxy'] = (
        df['price'] *
        df['review_scores_rating'] *
        df['occupancy_rate']
    )

    df[TARGET_VARIABLE] = np.log1p(df['rentability_proxy'])

    # Clipping
    lower = df[TARGET_VARIABLE].quantile(0.01)
    upper = df[TARGET_VARIABLE].quantile(0.99)
    df[TARGET_VARIABLE] = np.clip(df[TARGET_VARIABLE], lower, upper)

    # Target Encoding (rentabilidad media por barrio)
    rentability_mean_map = df.groupby(
        'neighbourhood_cleansed'
    )[TARGET_VARIABLE].mean()

    df['neighbourhood_rentability_mean'] = df[
        'neighbourhood_cleansed'
    ].map(rentability_mean_map)

    return df, rentability_mean_map.to_dict()


# -------------------------------------------------------------------
# --- 2. ENTRENAMIENTO DEL MODELO ---
@st.cache_resource
def train_model(df_clean):

    X_cols = NUMERICAL_FEATURES_INITIAL + CATEGORICAL_FEATURES_INITIAL + ['neighbourhood_rentability_mean']

    X = df_clean[X_cols].copy()
    y = df_clean[TARGET_VARIABLE]

    numerical_features = NUMERICAL_FEATURES_INITIAL + ['neighbourhood_rentability_mean']

    preprocessing = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES_INITIAL)
        ]
    )

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ('preprocessor', preprocessing),
        ('model', model)
    ])

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    return pipeline, model


# -------------------------------------------------------------------
# --- 3. PREDICCIÓN ---
def predict_rentability(pipeline, input_dict):
    df_input = pd.DataFrame([input_dict])
    ordered_cols = NUMERICAL_FEATURES_INITIAL + CATEGORICAL_FEATURES_INITIAL + ['neighbourhood_rentability_mean']
    df_input = df_input[ordered_cols]

    log_pred = pipeline.predict(df_input)[0]
    proxy_pred = np.expm1(log_pred)

    return log_pred, proxy_pred


# -------------------------------------------------------------------
# --- 4. UI STREAMLIT ---
df_clean, neighbourhood_map = load_data()

if df_clean.empty:
    st.stop()

pipeline, model = train_model(df_clean)

st.title("💰 Análisis Predictivo de Rentabilidad Airbnb – CDMX")

# --- Media Global del Dataset ---
media_global = df_clean[TARGET_VARIABLE].mean()
st.markdown(f"""
### 📘 Media Histórica Global del Dataset  
**{media_global:.4f}**
""")

st.header("🎯 1. Métricas del Modelo")
colA, colB, colC = st.columns(3)
colA.metric("R²", f"{FINAL_R2*100:.2f}%")
colB.metric("RMSE (log)", f"{FINAL_RMSE:.4f}")
colC.metric("Filas Analizadas", f"{df_clean.shape[0]:,}")

st.markdown("---")


# -------------------------------------------------------------------
# VISUALIZACIONES EXPLORATORIAS
# -------------------------------------------------------------------

st.header("📊 2. Visualizaciones Exploratorias del Dataset")

# --- Visualización 1: Rentabilidad por Barrio ---
st.subheader("🏙 Top 10 Barrios con Mayor Rentabilidad Log")

top_barrios = (
    df_clean.groupby('neighbourhood_cleansed')[TARGET_VARIABLE]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots()
top_barrios.plot(kind='bar', color='lightgreen', ax=ax)
ax.set_ylabel("Rentabilidad Log Media")
ax.set_xlabel("Barrio")
ax.set_title("Top 10 Barrios (Media de Rentabilidad Log)")
st.pyplot(fig)

# --- Visualización 2: Heatmap ---
st.subheader("🔥 Mapa de Correlaciones (Variables Numéricas)")

corr = df_clean.select_dtypes(include=['float64', 'int64']).corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, cmap='coolwarm', annot=False)
st.pyplot(fig)

# --- Visualización 3: Distribución del Target ---
st.subheader("📈 Distribución del Target (Rentabilidad Log)")

fig, ax = plt.subplots()
ax.hist(df_clean[TARGET_VARIABLE], bins=30, color='skyblue', edgecolor='black')
ax.set_title("Distribución de Rentabilidad Log")
ax.set_xlabel("Log Rentabilidad")
ax.set_ylabel("Frecuencia")
st.pyplot(fig)

st.markdown("---")


# -------------------------------------------------------------------
# --- 5. SIMULADOR ---
# -------------------------------------------------------------------

st.header("🔮 3. Simulación de Rentabilidad")

col1, col2, col3 = st.columns(3)

with col1:
    accommodates = st.slider("Capacidad de Huéspedes", 1, 16, 4)
    minimum_nights = st.slider("Mínimo de Noches", 1, 30, 3)
    number_of_reviews = st.slider("Número de Reseñas", 0, 500, 50)

with col2:
    calculated_host_listings_count = st.number_input("Listings del Host", 1, 100, 1)
    review_scores_rating = st.slider("Puntuación de Reseñas", 3.0, 5.0, 4.84, step=0.01)
    host_is_superhost = st.selectbox("Superhost", ['t', 'f'])

with col3:
    neighbourhood = st.selectbox("Barrio", sorted(neighbourhood_map.keys()))
    room_type = st.selectbox("Tipo de Habitación", df_clean['room_type'].unique())
    property_type = st.selectbox("Tipo de Propiedad", df_clean['property_type'].unique())


if st.button("Calcular Rentabilidad", type="primary"):

    neighbourhood_val = neighbourhood_map[neighbourhood]

    st.info(f"📍 **Rentabilidad Media Histórica del Barrio {neighbourhood}: {neighbourhood_val:.4f}**")

    input_data = {
        'accommodates': accommodates,
        'review_scores_rating': review_scores_rating,
        'calculated_host_listings_count': calculated_host_listings_count,
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'host_is_superhost': host_is_superhost,
        'room_type': room_type,
        'property_type': property_type,
        'neighbourhood_rentability_mean': neighbourhood_val
    }

    log_pred, proxy_pred = predict_rentability(pipeline, input_data)

    st.subheader("🚀 Resultado")

    colR1, colR2 = st.columns(2)

    colR1.metric("Rentabilidad Proxy (Original)", f"{proxy_pred:,.2f}")
    colR2.metric("Log Rentabilidad (Predicho)", f"{log_pred:.4f}")
