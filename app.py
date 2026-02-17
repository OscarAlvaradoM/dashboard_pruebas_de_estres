import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os

# =========================
# CONFIGURACIÓN GENERAL
# =========================

DATA_PATH = "Data/"

main_file = "temperatura_celular_10-02-2026_17:27:15_natural.csv"

COLORS = {
    "bg": "#0F172A",
    "card": "#1E293B",
    "heating": "#F97316",
    "cooling": "#06B6D4",
    "moving_avg": "#22C55E",
    "text_main": "#E2E8F0",
    "text_secondary": "#94A3B8",
    "grid": "rgba(148,163,184,0.1)"
}

comparison_files = [
    "temperatura_celular_10-02-2026_17:27:15_natural.csv",
    "temperatura_celular_10-02-2026_18:14:52_ventilador.csv",
    "temperatura_celular_10-02-2026_18:49:08_aluminio.csv",
    "temperatura_celular_10-02-2026_19:29:46_hielos.csv",
    "temperatura_celular_10-02-2026_20:15:17_sin_apps.csv"
]

INPUT_STYLE = {
    "backgroundColor": COLORS["bg"],
    "color": COLORS["text_main"],
    "border": f"1px solid {COLORS['grid']}",
    "borderRadius": "6px",
    "padding": "6px"
}

# =========================
# CARGA DE DATOS
# =========================

df_main = pd.read_csv(os.path.join(DATA_PATH, main_file))

dfs_comparison = {
    file: pd.read_csv(os.path.join(DATA_PATH, file))
    for file in comparison_files
}

max_temp = df_main["temperatura_C"].max()
max_temp_time = df_main.loc[df_main["temperatura_C"] == max_temp, "tiempo_s"].values[0]

# =========================
# APP
# =========================

app = dash.Dash(__name__)
app.title = "Termal Dashboard"

app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "padding": "40px"},
    children=[

        # HEADER
        html.Div([
            html.H1("Termal Analysis Dashboard",
                    style={"color": COLORS["text_main"], "marginBottom": "5px"}),

            html.P("Experimental Thermal Run: Natural Cooling vs Other Methods",
                   style={"color": COLORS["text_secondary"]})
        ]),

        html.Br(),

        dcc.Tabs(
            id="tabs",
            value="tab1",
            children=[

                dcc.Tab(label="Natural cooling", value="tab1",
                        style={"backgroundColor": COLORS["card"], "color": COLORS["text_secondary"]},
                        selected_style={"backgroundColor": COLORS["card"],
                                        "color": COLORS["heating"],
                                        "borderTop": f"3px solid {COLORS['heating']}"}),

                dcc.Tab(label="Cooling comparison", value="tab2",
                        style={"backgroundColor": COLORS["card"], "color": COLORS["text_secondary"]},
                        selected_style={"backgroundColor": COLORS["card"],
                                        "color": COLORS["cooling"],
                                        "borderTop": f"3px solid {COLORS['cooling']}"})
            ]
        ),

        html.Div(id="tab-content")
    ]
)


# =========================
# TAB CONTENT
# =========================

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab(tab):

    if tab == "tab1":

        return html.Div([

            # KPI CARDS
            html.Div(id="kpi-cards",
                     style={"display": "flex", "gap": "20px", "marginTop": "30px"}),

            html.Br(),

            html.Div([

                # PANEL IZQUIERDO
                html.Div([

                    html.H3("Parameters",
                            style={"color": COLORS["text_main"]}),

                    html.Label("Window [s]:", style={"color": COLORS["text_secondary"]}),
                    dcc.Dropdown(
                        id="window",
                        options=[{"label": str(x), "value": x} for x in [20,30,45,60, 120, 300]],
                        value=20,
                        style={"backgroundColor": COLORS["bg"],
                            "color": COLORS["text_main"]},
                        className="dark-dropdown"
                    ),

                    html.Br(),

                    html.Label("Start heating [s]:", style={"color": COLORS["text_secondary"]}),
                    dcc.Input(id="start_heat", type="number", value=max_temp_time - 360, 
                              style=INPUT_STYLE),

                    html.Br(),

                    html.Label("Switch point [s]:", style={"color": COLORS["text_secondary"]}),
                    dcc.Input(id="switch_point", type="number", value=max_temp_time,
                              style=INPUT_STYLE),

                    html.Br(),

                    html.Label("End cooling [s]:", style={"color": COLORS["text_secondary"]}),
                    dcc.Input(id="end_cool", type="number", value=max_temp_time + 360,
                              style=INPUT_STYLE),

                ],
                style={
                    "width": "22%",
                    "backgroundColor": COLORS["card"],
                    "padding": "20px",
                    "borderRadius": "12px"
                }),

                # GRAPH
                html.Div([
                    dcc.Graph(id="main-graph")
                ],
                style={"width": "75%"})

            ],
            style={"display": "flex", "gap": "30px"})

        ])

    if tab == "tab2":

        return html.Div([

            html.Br(),

            html.Div([
                html.Label("Window (s)", style={"color": COLORS["text_secondary"]}),
                dcc.Dropdown(
                    id="ma-window-compare",
                    options=[{"label": str(x), "value": x} for x in [20,30,45,60, 120, 300]],
                    value=20,
                    style={"backgroundColor": COLORS["bg"],
                            "color": COLORS["text_main"]},
                    className="dark-dropdown"
                )
            ],
            style={
                "width": "20%",
                "backgroundColor": COLORS["card"],
                "padding": "20px",
                "borderRadius": "12px"
            }),

            html.Br(),

            dcc.Graph(id="comparison-graph")

        ])

# =========================
# TAB 1 CALLBACK
# =========================

@app.callback(
    Output("main-graph", "figure"),
    Input("window", "value"),
    Input("start_heat", "value"),
    Input("switch_point", "value"),
    Input("end_cool", "value")
)
def update_main_graph(window, start_heat, switch_point, end_cool):
    df = df_main.copy()
    df["MA"] = df["temperatura_C"].rolling(window=window).mean()

    Tmax = df["temperatura_C"].max()
    idx_max = df["temperatura_C"].idxmax()
    t_peak = df["tiempo_s"].iloc[idx_max]

    fig = go.Figure()

    # 🔥 Temperatura
    fig.add_trace(go.Scatter(
        x=df["tiempo_s"],
        y=df["temperatura_C"],
        mode="lines",
        line=dict(color=COLORS["heating"], width=2.5),
        name="Temperature"
    ))

    # 🟢 Media móvil
    fig.add_trace(go.Scatter(
        x=df["tiempo_s"],
        y=df["MA"],
        mode="lines",
        line=dict(color=COLORS["moving_avg"], width=3),
        name=f"Moving Avg ({window}s)"
    ))

    # 🔶 Heating zone
    fig.add_vrect(
        x0=start_heat,
        x1=switch_point,
        fillcolor="rgba(249,115,22,0.08)",
        line_width=0,
        layer="below"
    )

    # 🔷 Cooling zone
    fig.add_vrect(
        x0=switch_point,
        x1=end_cool,
        fillcolor="rgba(6,182,212,0.08)",
        line_width=0,
        layer="below"
    )

    # Líneas verticales de referencia
    fig.add_vline(
        x=start_heat,
        line_dash="dot",
        line_color=COLORS["heating"],
        annotation_text="Start Heating",
        annotation_position="top left",
        line_width=2
    )

    fig.add_vline(
        x=switch_point,
        line_dash="dot",
        line_color="orange",
        line_width=2
    )

    fig.add_vline(
        x=end_cool,
        line_dash="dot",
        line_color=COLORS["cooling"],
        annotation_text="End Cooling",
        annotation_position="top left",
        line_width=2
    )

    # 🎯 Peak annotation minimalista
    fig.add_annotation(
        x=t_peak,
        y=Tmax,
        text="• Peak Temperature",
        showarrow=True,
        arrowcolor=COLORS["text_secondary"],
        font=dict(color=COLORS["text_secondary"])
    )

    fig.update_layout(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text_main"]),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            title="Time (s)",
            gridcolor=COLORS["grid"]
        ),
        yaxis=dict(
            title="Temperature (°C)",
            gridcolor=COLORS["grid"]
        ),
        legend=dict(
            orientation="h",
            y=-0.2
        )
    )

    return fig

@app.callback(
    Output("kpi-cards", "children"),
    Input("window", "value"),
    Input("start_heat", "value"),
    Input("switch_point", "value"),
    Input("end_cool", "value")
)
def update_kpis(window, start_heat, switch_point, end_cool):

    heating_time = (switch_point - start_heat)/60
    cooling_time = (end_cool - switch_point)/60
    Tmax = df_main["temperatura_C"].max()

    # ejemplo simple de k
    # k = 0.0031

    def card(title, value):
        return html.Div([
            html.P(title, style={"color": COLORS["text_secondary"]}),
            html.H2(value, style={"color": COLORS["text_main"]})
        ],
        style={
            "backgroundColor": COLORS["card"],
            "padding": "20px",
            "borderRadius": "12px",
            "width": "22%"
        })

    return [
        card("🔥 Heating Time", f"{heating_time:.1f} min"),
        card("❄ Cooling Time", f"{cooling_time:.1f} min"),
        card("🌡 Max Temp", f"{Tmax:.1f} °C"),
        #card("k Cooling", f"{k:.4f} s⁻¹")
    ]

# =========================
# TAB 2 CALLBACK
# =========================
@app.callback(
    Output("comparison-graph", "figure"),
    Input("ma-window-compare", "value")
)
def update_comparison_graph(window):

    resultados = []

    for nombre, df in dfs_comparison.items():

        df = df.copy()
        df["MA"] = df["temperatura_C"].rolling(window=window).mean()

        idx_max = df["temperatura_C"].idxmax()
        df = df.loc[idx_max:].copy()
        df["tiempo_s"] -= df["tiempo_s"].iloc[0]

        tiempo_total = df["tiempo_s"].max()

        resultados.append((nombre, df, tiempo_total))

    resultados.sort(key=lambda x: x[2], reverse=True)

    fig = go.Figure()

    for nombre, df, tiempo_total in resultados:

        label = nombre.split("_")[-1].replace(".csv", "")

        fig.add_trace(go.Scatter(
            x=df["tiempo_s"],
            y=df["MA"],
            mode="lines",
            name=f"{label} ({tiempo_total/60:.1f} min)",
            line=dict(width=3)
        ))

    fig.update_layout(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text_main"]),
        xaxis=dict(
            title="Time since Tmax (s)",
            gridcolor=COLORS["grid"]
        ),
        yaxis=dict(
            title="Temperature (°C)",
            gridcolor=COLORS["grid"]
        ),
        legend=dict(
            bgcolor=COLORS["card"]
        )
    )

    return fig

# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run_server(debug=True)