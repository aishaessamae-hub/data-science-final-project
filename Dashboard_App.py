import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

df = pd.read_csv("fordgobike_cleaned.csv")

df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

df = df.dropna(subset=["start_time"])

df["trip_duration_min"] = df["duration_sec"] / 60

df["day_of_week"] = df["start_time"].dt.day_name()
df["month"] = df["start_time"].dt.strftime("%b")

current_year = df["start_time"].dt.year.max()
df["age"] = current_year - df["member_birth_year"]

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 29, 50, 100],
    labels=["Young", "Adult", "Senior"]
)

df["member_gender"] = df["member_gender"].fillna("Unknown")

# ================= DASH APP =================
app = Dash(__name__)
app.title = "Bike Sharing Dashboard"

# ================= LAYOUT =================
app.layout = html.Div([
    html.H1("Bike Sharing Interactive Dashboard", style={"textAlign": "center"}),

    html.Div([
        dcc.Dropdown(
            id="gender_filter",
            options=[{"label": g, "value": g} for g in df["member_gender"].unique()],
            multi=True,
            placeholder="Select Gender"
        ),

        dcc.Dropdown(
            id="user_filter",
            options=[{"label": u, "value": u} for u in df["user_type"].unique()],
            multi=True,
            placeholder="Select User Type"
        ),
    ], style={"width": "50%", "margin": "auto"}),

    html.Br(),

    html.Div([
        dcc.Graph(id="trips_by_day"),
        dcc.Graph(id="gender_pie"),
    ], style={"display": "flex"}),

    html.Div([
        dcc.Graph(id="age_hist"),
        dcc.Graph(id="station_bar"),
    ], style={"display": "flex"}),
])

# ================= CALLBACK =================
@app.callback(
    Output("trips_by_day", "figure"),
    Output("gender_pie", "figure"),
    Output("age_hist", "figure"),
    Output("station_bar", "figure"),
    Input("gender_filter", "value"),
    Input("user_filter", "value"),
)
def update_dashboard(gender, user):
    dff = df.copy()

    if gender:
        dff = dff[dff["member_gender"].isin(gender)]

    if user:
        dff = dff[dff["user_type"].isin(user)]

    fig_day = px.bar(
        dff.groupby("day_of_week").size().reset_index(name="Trips"),
        x="day_of_week",
        y="Trips",
        title="Trips by Day of Week"
    )

    fig_gender = px.pie(
        dff,
        names="member_gender",
        title="Gender Distribution"
    )

    fig_age = px.histogram(
        dff,
        x="age",
        nbins=20,
        title="Age Distribution"
    )

    top_stations = (
        dff["start_station_name"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    fig_station = px.bar(
        top_stations,
        x="count",
        y="start_station_name",
        orientation="h",
        title="Top 10 Start Stations"
    )

    return fig_day, fig_gender, fig_age, fig_station


if __name__ == "__main__":
    app.run(debug=True)
