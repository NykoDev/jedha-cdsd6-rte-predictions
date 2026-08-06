import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import timedelta
from dotenv import load_dotenv
import utils


load_dotenv()

st.set_page_config(layout="wide")

# Historique (référence pour les métriques "consommation réelle") + prédictions : les deux
# viennent de S3 (data/ est gitignoré, donc absent du dépôt cloné par Streamlit Community Cloud
# au déploiement). Les prédictions sont produites par l'API /predict, cf. api/main.py.
DATA_S3_KEY = "data/historic_dataset_2022_2025.csv"
PREDICTIONS_S3_KEY = "data/previsions.csv"


@st.cache_data(ttl=3600)
def load_recent_dataset():
    return utils.load_dataset_s3(DATA_S3_KEY)

@st.cache_data(ttl=3600)
def load_prediction_dataset():
    return utils.load_dataset_s3(PREDICTIONS_S3_KEY)


# Fichiers absents de S3 (historique pas encore chargé / prédiction pas encore lancée) : on
# arrête le rendu de la page ici, avec un message distinct par fichier, plutôt que de planter sur
# les traitements pandas qui suivent (colonnes attendues sur un df vide/inexistant).
try:
    df = load_recent_dataset()
except FileNotFoundError:
    st.warning("Pas d'historique disponible")
    st.stop()
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
df["datetime_fr"] = df["datetime_utc"].dt.tz_convert("Europe/Paris")

try:
    df_pred = load_prediction_dataset()
except FileNotFoundError:
    st.warning("Pas de prévision disponible")
    st.stop()
df_pred["target_datetime_utc"] = pd.to_datetime(df_pred["target_datetime_utc"], utc=True)
df_pred["target_datetime_fr"] = df_pred["target_datetime_utc"].dt.tz_convert("Europe/Paris")


# "Aujourd'hui" pour le dashboard : CURRENT_DAY (dashboard/.env, format DD/MM/YYYY) si définie,
# pour figer la démo sur une date connue plutôt que dépendre de données réellement à jour ;
# sinon la date réelle du jour (cf. utils.get_current_day)
data_end_date = utils.get_current_day()


# Session State
default_start_date = data_end_date
default_end_date = data_end_date

# select_period
if "select_period" not in st.session_state:
    st.session_state.select_period = 'J-1'

# start_datetime of data
if "pred_start_date" not in st.session_state:
    st.session_state.pred_start_date = default_start_date

# end_datetime of data
if "pred_end_date" not in st.session_state:
    st.session_state.pred_end_date = default_end_date

mask_start_date = (df["datetime_fr"].dt.date >= st.session_state.pred_start_date)
mask_end_date = (df["datetime_fr"].dt.date <= st.session_state.pred_end_date)
df2 = df[mask_start_date & mask_end_date]

#
def set_j_value():
    select_period = st.session_state.select_period
    if select_period == 'J-1':
        n_days = 0
    elif select_period == 'J-3':
        n_days = 2
    elif select_period == 'J-7':
        n_days = 6

    st.session_state.pred_start_date = default_start_date - timedelta(days=n_days)


# Calculate metrics for the selected period
mean_power = df2["consumption"].mean()
min_power = df2["consumption"].min()
max_power = df2["consumption"].max()

# calculate metrics for predictions if all in the period
mask_pred = df_pred["target_datetime_fr"].dt.date >= st.session_state.pred_start_date
df_pred2 = df_pred[mask_pred]
mean_power_pred = df_pred2["y_pred_ensemble"].mean()
min_power_pred = df_pred2["y_pred_ensemble"].min()
max_power_pred = df_pred2["y_pred_ensemble"].max()

delta_mean_power = mean_power_pred - mean_power
delta_min_power = min_power_pred - min_power
delta_max_power = max_power_pred - max_power


st.html("""
<style>
    div.st-key-mean_metrics, div.st-key-min_metrics, div.st-key-max_metrics {
        min-height: 135px !important;
    }
    div.st-key-min_metrics [data-testid="stMetricValue"],
    div.st-key-max_metrics [data-testid="stMetricValue"] {
    font-size: 24px !important;
    }
</style>
""")


###############################################
# Subtitle
if st.session_state.pred_start_date == st.session_state.pred_end_date:
    st.subheader(f"Puissance électrique consommée le {st.session_state.pred_start_date.strftime('%d/%m/%Y')}")
else:
    st.subheader(f"Puissance électrique consommée entre le {st.session_state.pred_start_date.strftime('%d/%m/%Y')} et le {st.session_state.pred_end_date.strftime('%d/%m/%Y')}")


###############################################
# Metrics power

col1_metrics, col2_metrics, col3_metrics = st.columns((4,2,2))

with col1_metrics:
    with st.container(border=True, key="mean_metrics"):
        col11_prod, col12_pred = st.columns(2)
        with col11_prod:
            st.metric("Puissance moyenne", f"{mean_power:.2f} MW")
        with col12_pred:
            st.metric("Prédiction moyenne", f"{mean_power_pred:.2f} MW", f"{delta_mean_power:.2f} MW")

with col2_metrics:
    with st.container(border=True, key="min_metrics"):
        col21_prod, col22_pred = st.columns(2)
        with col21_prod:
            st.metric("Puissance minimale", f"{min_power:.2f} MW")
        with col22_pred:
            st.metric("Prédiction minimale", f"{min_power_pred:.2f} MW", f"{delta_min_power:.2f} MW")

with col3_metrics:
    with st.container(border=True, key="max_metrics"):
        col31_prod, col32_pred = st.columns(2)
        with col31_prod:
            st.metric("Puissance maximale", f"{max_power:.2f} MW")
        with col32_pred:
            st.metric("Prédiction maximale", f"{max_power_pred:.2f} MW", f"{delta_max_power:.2f} MW")

##############################################


###############################################
# Filtre période
option_days_selection = [ 'J-1', 'J-3', 'J-7']

col1_btn, col2_btn = st.columns(2, vertical_alignment="bottom")
with col1_btn:
    st.pills(
        "",
        key="select_period",
        options=option_days_selection,
        selection_mode="single",  # un seul sélectionnable à la fois
        on_change=set_j_value
    )
with col2_btn:
    # Clear cache
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("", icon=":material/refresh:", help="Actualiser les données"):
            st.cache_data.clear()



################################################
# Chart
fig = px.line(df2, x="datetime_fr", y="consumption", range_y=[20000, None],
    title=f"Puissance électrique consommée et prédictions en MW",
    labels={"datetime_fr": "", "consumption": "Consommation (MW)"},)
fig.add_trace( px.line(df_pred2, x="target_datetime_fr", y="y_pred_ensemble",).data[0] )

fig.data[0].update({'name':'Consommation'})
fig.data[1].update({'line':dict(dash='dash', color="green"), 'name':'Prédiction'})
fig.update_traces(showlegend = True)

with st.container(border=1):
    st.plotly_chart(fig)
