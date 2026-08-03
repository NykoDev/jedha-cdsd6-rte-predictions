import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import datetime
from dotenv import load_dotenv
import utils


load_dotenv()

st.set_page_config(layout="wide")

# Chargé depuis S3 (et non plus un CSV local) : data/ est gitignoré, donc absent du dépôt cloné
# par Streamlit Community Cloud au déploiement.
DATA_KEY = "data/historic_dataset_2022_2025.csv"


@st.cache_data
def load_historical_dataset():
    return utils.load_dataset_s3(DATA_KEY)


df = load_historical_dataset()

df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
df["datetime_fr"] = df["datetime_utc"].dt.tz_convert("Europe/Paris")


# Session State
default_start_date = datetime.date(2022, 1, 1)
default_end_date = datetime.date(2025, 12, 31)

# start_datetime of data
if "start_date" not in st.session_state:
    st.session_state.start_date = default_start_date

# end_datetime of data
if "end_date" not in st.session_state:
    st.session_state.end_date = default_end_date

# initial values for date inputs
if "inp_start_date" not in st.session_state:
    st.session_state.inp_start_date = default_start_date

if "inp_end_date" not in st.session_state:
    st.session_state.inp_end_date = default_end_date

def update_range_year():
    st.session_state.start_date = datetime.date(st.session_state.range_year[0], 1, 1)
    st.session_state.end_date = datetime.date(st.session_state.range_year[1], 12, 31)
    st.session_state.inp_start_date = st.session_state.start_date
    st.session_state.inp_end_date = st.session_state.end_date

def update_start_datetime():
    st.session_state.start_date = st.session_state.inp_start_date
    st.session_state.range_year = (st.session_state.start_date.year, st.session_state.end_date.year)

def update_end_datetime():
    st.session_state.end_date = st.session_state.inp_end_date
    st.session_state.range_year = (st.session_state.start_date.year, st.session_state.end_date.year)


mask_start_date = (df["datetime_fr"].dt.date >= st.session_state.start_date)
mask_end_date = (df["datetime_fr"].dt.date <= st.session_state.end_date)
df2 = df[mask_start_date & mask_end_date]

mean_power = df2["consumption"].mean()
min_power = df2["consumption"].min()
max_power = df2["consumption"].max()

###############################################
# Subtitle

st.subheader(f"Puissance électrique consommée entre le {st.session_state.start_date.strftime('%d/%m/%Y')} et le {st.session_state.end_date.strftime('%d/%m/%Y')}")


###############################################
# Metrics power

col1_stats, col2_stats, col3_stats = st.columns(3)

with col1_stats:
    st.metric("Puissance moyenne", f"{mean_power:.2f} MW", border=True)
with col2_stats:
    st.metric("Puissance minimale", f"{min_power:.2f} MW", border=True)
with col3_stats:
    st.metric("Puissance maximale", f"{max_power:.2f} MW", border=True)

###############################################
# Filters

# Year slider
start_year, end_year = st.slider( label="Année", key="range_year",
    min_value=2022,
    max_value=2025,
    value=(2022, 2025),
    on_change=update_range_year
)


col1_date, col2_date = st.columns(2)
# Date calendar selector
with col1_date:
    start_time = st.date_input( label="Date de début", key="inp_start_date",
        min_value=datetime.date(2022, 1, 1),
        max_value=datetime.date(2025, 12, 31),
        format="DD/MM/YYYY",
        on_change=update_start_datetime
    )
with col2_date:
    end_time = st.date_input( label="Date de fin", key="inp_end_date",
        min_value=datetime.date(2022, 1, 1),
        max_value=datetime.date(2025, 12, 31),
        format="DD/MM/YYYY",
        on_change=update_end_datetime
    )

################################################
# Chart

fig2 = px.line(df2, x="datetime_utc", y="consumption", render_mode="svg",
    labels={"datetime_utc": "", "consumption": "Consommation (MW)"},
    title=f"Consommation électrique entre {st.session_state.start_date} et {st.session_state.end_date} en MW")
fig2.update_xaxes(rangeslider_visible=True)

with st.container(border=1):
    st.plotly_chart(fig2)
