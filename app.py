import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
import io
import base64
import re

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
DEFAULT_T_AMB = 24.5

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
time_min = float(df_main["tiempo_s"].min())
time_max = float(df_main["tiempo_s"].max())


def clamp(value, low, high):
    return max(low, min(value, high))


def normalize_ambient_temp(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_T_AMB


def extract_horizontal_line_y(relayout_data):
    if not isinstance(relayout_data, dict):
        return None

    y0_by_idx = {}
    y1_by_idx = {}

    for key, value in relayout_data.items():
        m0 = re.match(r"shapes\[(\d+)\]\.y0$", key)
        if m0:
            y0_by_idx[int(m0.group(1))] = float(value)
            continue
        m1 = re.match(r"shapes\[(\d+)\]\.y1$", key)
        if m1:
            y1_by_idx[int(m1.group(1))] = float(value)

    candidate_indices = sorted(set(y0_by_idx.keys()) | set(y1_by_idx.keys()), reverse=True)
    for idx in candidate_indices:
        y0 = y0_by_idx.get(idx)
        y1 = y1_by_idx.get(idx)
        if y0 is not None and (y1 is None or abs(y0 - y1) < 1e-9):
            return y0
        if y1 is not None and y0 is None:
            return y1

    shapes = relayout_data.get("shapes")
    if isinstance(shapes, list):
        for shape in reversed(shapes):
            if not isinstance(shape, dict):
                continue
            y0 = shape.get("y0")
            y1 = shape.get("y1")
            if y0 is None or y1 is None:
                continue
            try:
                y0 = float(y0)
                y1 = float(y1)
            except (TypeError, ValueError):
                continue
            if abs(y0 - y1) < 1e-9:
                return y0

    return None


default_switch = clamp(float(max_temp_time), time_min, time_max)
default_start = clamp(default_switch - 360, time_min, default_switch)
default_end = clamp(default_switch + 360, default_switch, time_max)


def normalize_phase_points(phase_points):
    if not isinstance(phase_points, (list, tuple)) or len(phase_points) < 3:
        return default_start, default_switch, default_end
    points = sorted(float(x) for x in phase_points[:3])
    start_heat = clamp(points[0], time_min, time_max)
    switch_point = clamp(points[1], start_heat, time_max)
    end_cool = clamp(points[2], switch_point, time_max)
    return start_heat, switch_point, end_cool


def extract_phase_points_from_relayout(relayout_data, current_points):
    start_heat, switch_point, end_cool = normalize_phase_points(current_points)
    points = [start_heat, switch_point, end_cool]
    moved = False

    shape_to_point = {2: 0, 3: 1, 4: 2}
    for shape_idx, point_idx in shape_to_point.items():
        x0_key = f"shapes[{shape_idx}].x0"
        x1_key = f"shapes[{shape_idx}].x1"
        if x0_key in relayout_data:
            points[point_idx] = float(relayout_data[x0_key])
            moved = True
        elif x1_key in relayout_data:
            points[point_idx] = float(relayout_data[x1_key])
            moved = True

    if not moved and isinstance(relayout_data.get("shapes"), list):
        shapes = relayout_data["shapes"]
        if len(shapes) >= 5:
            points = [
                float(shapes[2].get("x0", shapes[2].get("x1", start_heat))),
                float(shapes[3].get("x0", shapes[3].get("x1", switch_point))),
                float(shapes[4].get("x0", shapes[4].get("x1", end_cool))),
            ]
            moved = True

    if not moved:
        return None

    return list(normalize_phase_points(points))


def parse_uploaded_csv(contents):
    try:
        _, encoded = contents.split(",", 1)
        decoded = base64.b64decode(encoded).decode("utf-8")
        df = pd.read_csv(io.StringIO(decoded))
        if not {"tiempo_s", "temperatura_C"}.issubset(df.columns):
            return None
        df = df[["tiempo_s", "temperatura_C"]].copy()
        df["tiempo_s"] = pd.to_numeric(df["tiempo_s"], errors="coerce")
        df["temperatura_C"] = pd.to_numeric(df["temperatura_C"], errors="coerce")
        df = df.dropna().sort_values("tiempo_s").reset_index(drop=True)
        if len(df) < 2:
            return None
        return df
    except Exception:
        return None


def get_main_df(uploaded_main_store):
    if isinstance(uploaded_main_store, dict) and {"tiempo_s", "temperatura_C"}.issubset(uploaded_main_store.keys()):
        df = pd.DataFrame(uploaded_main_store)
        df["tiempo_s"] = pd.to_numeric(df["tiempo_s"], errors="coerce")
        df["temperatura_C"] = pd.to_numeric(df["temperatura_C"], errors="coerce")
        df = df.dropna().sort_values("tiempo_s").reset_index(drop=True)
        if len(df) >= 2:
            return df
    return df_main.copy()


def normalize_phase_points_for_df(phase_points, df):
    local_min = float(df["tiempo_s"].min())
    local_max = float(df["tiempo_s"].max())
    peak_time = float(df.loc[df["temperatura_C"].idxmax(), "tiempo_s"])
    default_local_switch = clamp(peak_time, local_min, local_max)
    default_local_start = clamp(default_local_switch - 360, local_min, default_local_switch)
    default_local_end = clamp(default_local_switch + 360, default_local_switch, local_max)

    if not isinstance(phase_points, (list, tuple)) or len(phase_points) < 3:
        return default_local_start, default_local_switch, default_local_end

    points = sorted(float(x) for x in phase_points[:3])
    start_heat = clamp(points[0], local_min, local_max)
    switch_point = clamp(points[1], start_heat, local_max)
    end_cool = clamp(points[2], switch_point, local_max)
    return start_heat, switch_point, end_cool

# =========================
# APP
# =========================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
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
        html.Div(
            [
                html.Label("Temperatura ambiente (°C)", style={"color": COLORS["text_secondary"]}),
                dcc.Input(
                    id="ambient-temp",
                    type="number",
                    value=DEFAULT_T_AMB,
                    step=0.1,
                    style={
                        "backgroundColor": COLORS["bg"],
                        "color": COLORS["text_main"],
                        "border": f"1px solid {COLORS['grid']}",
                        "borderRadius": "8px",
                        "padding": "8px 10px",
                        "width": "150px"
                    }
                ),
            ],
            style={
                "display": "flex",
                "gap": "10px",
                "alignItems": "center",
                "marginTop": "10px"
            }
        ),

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

        html.Div(id="tab-content"),
        dcc.Store(id="phase-points-store", data=[default_start, default_switch, default_end]),
        dcc.Store(id="comparison-anchor-store", data={}),
        dcc.Store(id="uploaded-main-store", data=None),
        dcc.Store(id="ambient-temp-store", data=DEFAULT_T_AMB),
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
                    dcc.Upload(
                        id="upload-main-data",
                        children=html.Button(
                            "Upload CSV",
                            style={
                                "backgroundColor": COLORS["bg"],
                                "color": COLORS["text_main"],
                                "border": f"1px solid {COLORS['grid']}",
                                "borderRadius": "8px",
                                "padding": "10px 14px",
                                "cursor": "pointer",
                                "height": "40px"
                            }
                        ),
                        multiple=False
                    ),
                    html.P(
                        "Formato: columnas tiempo_s y temperatura_C",
                        style={"color": COLORS["text_secondary"], "marginTop": "8px"}
                    ),

                ],
                style={
                    "width": "22%",
                    "backgroundColor": COLORS["card"],
                    "padding": "20px",
                    "borderRadius": "12px"
                }),

                # GRAPH
                html.Div([
                    dcc.Graph(
                        id="main-graph",
                        config={"edits": {"shapePosition": True}}
                    ),
                    dcc.Graph(id="comparison-graph", style={"display": "none"}),
                    dcc.Dropdown(
                        id="ma-window-compare",
                        options=[{"label": str(x), "value": x} for x in [20,30,45,60, 120, 300]],
                        value=20,
                        style={"display": "none"}
                    ),
                    html.Button(
                        id="reset-comparison-anchors",
                        n_clicks=0,
                        style={"display": "none"}
                    ),
                    html.P(
                        "Drag vertical lines directly on the chart: Start heating | Switch | End cooling.",
                        style={"color": COLORS["text_secondary"], "marginTop": "6px"}
                    )
                ],
                style={"width": "75%"})

            ],
            style={"display": "flex", "gap": "30px"})

        ])

    if tab == "tab2":

        return html.Div([

            html.Br(),

            html.Div(
                [
                    html.Div([
                        html.Label("Window (s)", style={"color": COLORS["text_secondary"]}),
                        dcc.Dropdown(
                            id="ma-window-compare",
                            options=[{"label": str(x), "value": x} for x in [2,10,20,30,45,60, 120, 300]],
                            value=20,
                            style={"backgroundColor": COLORS["bg"], "color": COLORS["text_main"]},
                            className="dark-dropdown"
                        )
                    ], style={"width": "260px"}),
                    html.Button(
                        "Reset anchors",
                        id="reset-comparison-anchors",
                        n_clicks=0,
                        style={
                            "backgroundColor": COLORS["bg"],
                            "color": COLORS["text_main"],
                            "border": f"1px solid {COLORS['grid']}",
                            "borderRadius": "8px",
                            "padding": "10px 14px",
                            "cursor": "pointer",
                            "height": "40px"
                        }
                    ),
                    html.P(
                        "Click any curve to set its anchor (start at x=0).",
                        style={
                            "color": COLORS["text_secondary"],
                            "margin": "0 0 6px 0",
                            "height": "34px",
                            "display": "flex",
                            "alignItems": "center"
                        }
                    )
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "alignItems": "flex-end",
                    "backgroundColor": COLORS["card"],
                    "padding": "20px",
                    "borderRadius": "12px"
                }
            ),

            html.Br(),

            dcc.Graph(id="comparison-graph", config={"edits": {"shapePosition": True}}),
            dcc.Graph(id="main-graph", style={"display": "none"}),
            dcc.Upload(id="upload-main-data", style={"display": "none"})

        ])


@app.callback(
    Output("ambient-temp-store", "data"),
    Input("ambient-temp", "value"),
    Input("main-graph", "relayoutData"),
    Input("comparison-graph", "relayoutData"),
    State("ambient-temp-store", "data"),
    prevent_initial_call=True
)
def sync_ambient_temp(ambient_temp_input, main_relayout, comparison_relayout, current_ambient):
    trigger = dash.ctx.triggered_id
    current_ambient = normalize_ambient_temp(current_ambient)

    if trigger == "ambient-temp":
        new_ambient = normalize_ambient_temp(ambient_temp_input)
        if abs(new_ambient - current_ambient) < 1e-9:
            raise PreventUpdate
        return new_ambient

    if trigger == "main-graph":
        y = extract_horizontal_line_y(main_relayout)
    elif trigger == "comparison-graph":
        y = extract_horizontal_line_y(comparison_relayout)
    else:
        raise PreventUpdate

    if y is None:
        raise PreventUpdate

    new_ambient = normalize_ambient_temp(y)
    if abs(new_ambient - current_ambient) < 1e-9:
        raise PreventUpdate
    return new_ambient


@app.callback(
    Output("ambient-temp", "value"),
    Input("ambient-temp-store", "data")
)
def reflect_ambient_temp(ambient_temp_store):
    return normalize_ambient_temp(ambient_temp_store)

# =========================
# TAB 1 CALLBACK
# =========================

@app.callback(
    Output("uploaded-main-store", "data"),
    Output("phase-points-store", "data", allow_duplicate=True),
    Input("upload-main-data", "contents"),
    prevent_initial_call=True
)
def update_uploaded_main_data(contents):
    if not contents:
        raise PreventUpdate

    df = parse_uploaded_csv(contents)
    if df is None:
        raise PreventUpdate

    start_heat, switch_point, end_cool = normalize_phase_points_for_df(None, df)
    return (
        {
            "tiempo_s": df["tiempo_s"].tolist(),
            "temperatura_C": df["temperatura_C"].tolist(),
        },
        [start_heat, switch_point, end_cool],
    )

@app.callback(
    Output("phase-points-store", "data"),
    Input("main-graph", "relayoutData"),
    Input("tabs", "value"),
    State("phase-points-store", "data"),
    State("uploaded-main-store", "data"),
    prevent_initial_call=True
)
def update_phase_points_from_graph(relayout_data, tab, phase_points_store, uploaded_main_store):
    if tab != "tab1" or not isinstance(relayout_data, dict):
        raise PreventUpdate

    df = get_main_df(uploaded_main_store)
    current_points = normalize_phase_points_for_df(phase_points_store, df)
    updated = extract_phase_points_from_relayout(
        relayout_data,
        current_points,
    )
    if updated is None:
        raise PreventUpdate
    return list(normalize_phase_points_for_df(updated, df))


@app.callback(
    Output("main-graph", "figure"),
    Input("window", "value"),
    Input("phase-points-store", "data"),
    Input("uploaded-main-store", "data"),
    Input("ambient-temp-store", "data")
)
def update_main_graph(window, phase_points_store, uploaded_main_store, ambient_temp_store):
    T_AMB = normalize_ambient_temp(ambient_temp_store)
    df = get_main_df(uploaded_main_store)
    start_heat, switch_point, end_cool = normalize_phase_points_for_df(phase_points_store, df)

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
        layer="below",
        editable=False
    )

    # 🔷 Cooling zone
    fig.add_vrect(
        x0=switch_point,
        x1=end_cool,
        fillcolor="rgba(6,182,212,0.08)",
        line_width=0,
        layer="below",
        editable=False
    )

    # Líneas verticales de referencia
    fig.add_vline(
        x=start_heat,
        line_dash="dot",
        line_color=COLORS["heating"],
        annotation_text="Start Heating",
        annotation_position="top left",
        line_width=2,
        editable=True
    )

    fig.add_vline(
        x=switch_point,
        line_dash="dot",
        line_color="orange",
        line_width=2,
        editable=True
    )

    fig.add_vline(
        x=end_cool,
        line_dash="dot",
        line_color=COLORS["cooling"],
        annotation_text="End Cooling",
        annotation_position="top left",
        line_width=2,
        editable=True
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
    fig.add_hline(
        y=T_AMB,
        line_dash="dash",
        line_width=2,
        line_color=COLORS["text_secondary"],
        annotation_text=f"Temperatura ambiente ({T_AMB:.1f}°C)",
        annotation_position="top right",
        editable=True,
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
        ),
        uirevision="main-graph-static"
    )

    return fig

@app.callback(
    Output("kpi-cards", "children"),
    Input("window", "value"),
    Input("phase-points-store", "data"),
    Input("uploaded-main-store", "data"),
    Input("ambient-temp-store", "data")
)
def update_kpis(window, phase_points_store, uploaded_main_store, ambient_temp_store):
    T_AMB_KPI = normalize_ambient_temp(ambient_temp_store)
    df = get_main_df(uploaded_main_store)
    start_heat, switch_point, end_cool = normalize_phase_points_for_df(phase_points_store, df)

    heating_time = (switch_point - start_heat)/60
    cooling_time = (end_cool - switch_point)/60
    Tmax = df["temperatura_C"].max()

    df_cooling = df[
        (df["tiempo_s"] >= switch_point) &
        (df["tiempo_s"] <= end_cool)
    ].copy()

    df_newton = df_cooling[df_cooling["temperatura_C"] > T_AMB_KPI].copy()
    if len(df_newton) >= 2:
        df_newton["tiempo_rel"] = df_newton["tiempo_s"] - df_newton["tiempo_s"].iloc[0]
        df_newton["log_term"] = np.log(df_newton["temperatura_C"] - T_AMB_KPI)
        coef = np.polyfit(df_newton["tiempo_rel"], df_newton["log_term"], 1)
        k = -coef[0]
        k_label = f"{k:.4f} s⁻¹"
    else:
        k_label = "N/A"

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
        card("📉 k Cooling", k_label),
    ]

# =========================
# TAB 2 CALLBACK
# =========================
@app.callback(
    Output("comparison-anchor-store", "data"),
    Input("comparison-graph", "clickData"),
    Input("reset-comparison-anchors", "n_clicks"),
    State("comparison-anchor-store", "data"),
    prevent_initial_call=True
)
def update_comparison_anchors(click_data, reset_clicks, anchor_store):
    trigger = dash.ctx.triggered_id
    if trigger == "reset-comparison-anchors":
        return {}

    if trigger != "comparison-graph" or not isinstance(click_data, dict):
        raise PreventUpdate

    points = click_data.get("points", [])
    if not points:
        raise PreventUpdate

    custom = points[0].get("customdata")
    if not isinstance(custom, (list, tuple)) or len(custom) < 2:
        raise PreventUpdate

    nombre = custom[0]
    anchor_time = float(custom[1])

    if nombre not in dfs_comparison:
        raise PreventUpdate

    df = dfs_comparison[nombre]
    tmax_time = float(df.loc[df["temperatura_C"].idxmax(), "tiempo_s"])
    anchor_time = max(anchor_time, tmax_time)

    new_store = anchor_store.copy() if isinstance(anchor_store, dict) else {}
    new_store[nombre] = anchor_time
    return new_store


@app.callback(
    Output("comparison-graph", "figure"),
    Input("ma-window-compare", "value"),
    Input("comparison-anchor-store", "data"),
    Input("ambient-temp-store", "data")
)
def update_comparison_graph(window, anchor_store, ambient_temp_store):

    T_AMB = normalize_ambient_temp(ambient_temp_store)
    anchor_store = anchor_store if isinstance(anchor_store, dict) else {}
    resultados = []

    for nombre, df in dfs_comparison.items():

        df = df.copy()
        df["MA"] = df["temperatura_C"].rolling(window=window, min_periods=1).mean()

        idx_max = df["temperatura_C"].idxmax()
        tmax_time = float(df.loc[idx_max, "tiempo_s"])
        anchor_time = max(float(anchor_store.get(nombre, tmax_time)), tmax_time)

        df_plot = df[df["tiempo_s"] >= anchor_time].copy()
        if df_plot.empty:
            df_plot = df.loc[[df.index[-1]]].copy()
            anchor_time = float(df_plot["tiempo_s"].iloc[0])

        df_plot["tiempo_rel_s"] = df_plot["tiempo_s"] - anchor_time

        reaches_amb = df_plot[df_plot["temperatura_C"] <= T_AMB]
        if not reaches_amb.empty:
            tiempo_total = float(reaches_amb["tiempo_rel_s"].iloc[0])
        else:
            tiempo_total = float(df_plot["tiempo_rel_s"].iloc[-1])

        df_newton = df_plot[df_plot["temperatura_C"] > T_AMB].copy()
        if len(df_newton) >= 2:
            df_newton["log_term"] = np.log(df_newton["temperatura_C"] - T_AMB)
            coef = np.polyfit(df_newton["tiempo_rel_s"], df_newton["log_term"], 1)
            k = -coef[0]
        else:
            k = float("nan")

        resultados.append((nombre, df_plot, tiempo_total, k, anchor_time))

    resultados.sort(key=lambda x: x[2])

    fig = go.Figure()

    for nombre, df, tiempo_total, k, anchor_time in resultados:

        label = nombre.split("_")[-1].replace(".csv", "")
        k_label = f"{k:.4f} s⁻¹" if np.isfinite(k) else "N/A"
        customdata = np.column_stack([
            np.full(len(df), nombre, dtype=object),
            df["tiempo_s"].to_numpy()
        ])

        fig.add_trace(go.Scatter(
            x=df["tiempo_rel_s"],
            y=df["MA"],
            mode="lines",
            name=f"{label} | ({tiempo_total/60:.1f} min) | k={k_label}",
            line=dict(width=3),
            customdata=customdata,
            hovertemplate="t_rel=%{x:.1f}s<br>T=%{y:.2f}°C<extra></extra>"
        ))

    fig.add_hline(
        y=T_AMB,
        line_dash="dash",
        line_width=2,
        line_color=COLORS["text_secondary"],
        annotation_text=f"Temperatura ambiente ({T_AMB:.1f}°C)",
        annotation_position="top right",
        editable=True,
    )

    fig.update_layout(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text_main"]),
        xaxis=dict(
            title="Time since anchor (s)",
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
    app.run_server(debug=False)
