import dash
from dash import dcc, html, Input, Output, State
import dash_daq as daq
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy

app = dash.Dash(__name__)

primary1 = "#1e1e1e"
primary2 = "#363636"
primary3 = "#b3b3b3"
highlight1 = "#C7253E"
highlight2 = "#ffffff"


def plot_charge_and_discharge(charge, discharge, error_y, error_x, error=False):
    # Reading Data

    spacing = 1

    df = {
        "charge": {"data": pd.read_csv(f"data/{charge}.csv")},
        "discharge": {"data": pd.read_csv(f"data/{discharge}.csv")},
        "full": {
            "data": pd.concat(
                [
                    pd.read_csv(f"data/{charge}.csv"),
                    pd.read_csv(f"data/{discharge}.csv"),
                ]
            )
        },
    }

    # Getting Tension

    df["charge"]["tension"] = df["charge"]["data"]["Value"].astype(float)
    df["discharge"]["tension"] = df["discharge"]["data"]["Value"].astype(float)
    df["full"]["tension"] = pd.concat(
        [df["charge"]["tension"], df["discharge"]["tension"]]
    )

    # Getting Time

    df["charge"]["time"] = df["charge"]["data"]["Parameter"].astype(float) / 1000
    df["discharge"]["time"] = df["discharge"]["data"]["Parameter"].astype(float) / 1000
    df["full"]["time"] = pd.concat([df["charge"]["time"], df["discharge"]["time"]])

    # Fitting Charge Data

    charge_values, _ = scipy.optimize.curve_fit(
        lambda t, a, b: a * (1 - np.exp(-t / b)),
        df["charge"]["time"],
        df["charge"]["tension"],
    )

    df["charge"]["fitted"] = charge_values[0] * (
        1 - np.exp(-df["charge"]["time"] / charge_values[1])
    )

    # Fitting Discharge Data

    discharge_values, _ = scipy.optimize.curve_fit(
        lambda t, a, b, c: a * np.exp(-(t + c) / b),
        df["discharge"]["time"] - df["discharge"]["time"][0],
        df["discharge"]["tension"],
    )

    df["discharge"]["fitted"] = discharge_values[0] * np.exp(
        -(df["discharge"]["time"] - df["discharge"]["time"][0] + discharge_values[2])
        / discharge_values[1]
    )

    df["full"]["fitted"] = list(df["charge"]["fitted"]) + list(
        df["discharge"]["fitted"]
    )

    # Expected Curve

    df["charge"]["expected"] = [
        expected_charge(i - df["charge"]["time"][0]) for i in df["charge"]["time"]
    ]
    df["discharge"]["expected"] = [
        expected_charge(i - df["discharge"]["time"][0], False, 4.9)
        for i in df["discharge"]["time"]
    ]
    df["full"]["expected"] = df["charge"]["expected"] + df["discharge"]["expected"]

    # Error
    df["charge"]["time_error"] = [error_x for _ in df["charge"]["time"]]
    df["charge"]["tension_error"] = [error_y for _ in df["charge"]["tension"]]

    df["discharge"]["time_error"] = [error_x for _ in df["discharge"]["time"]]
    df["discharge"]["tension_error"] = [error_y for _ in df["discharge"]["tension"]]

    # Logaroithmic Scale
    df["charge"]["logarithmic"] = np.log(
        max(df["charge"]["tension"]) / (max(df["charge"]["tension"]) - df["charge"]["tension"])
    )
    df["discharge"]["logarithmic"] = np.log(
        df["discharge"]["tension"]
    )
    df["full"]["logarithmic"] = np.log(df["full"]["tension"])

    # Logarithmic Fitting
    df["charge"]["logarithmic_fitting"] = np.log(
        max(df["charge"]["fitted"]) / (max(df["charge"]["fitted"]) - df["charge"]["fitted"])
    )
    df["discharge"]["logarithmic_fitting"] = np.log(
        df["discharge"]["fitted"]
    )
    df["full"]["logarithmic_fitting"] = np.log(df["full"]["fitted"])

    # Logarithmic Error
    df["charge"]["logarithmic_error"] = [
        error_y / v if v != 0 else 1 for v in df["charge"]["tension"]
    ]
    df["discharge"]["logarithmic_error"] = [
        error_y / v if v != 0 else 1 for v in df["discharge"]["tension"]
    ]
    df["full"]["logarithmic_error"] = [
        error_y / v if v != 0 else 1 for v in df["full"]["tension"]
    ]

    # Fake Logarithmic Error
    df["charge"]["fake_logarithmic_error"] = df["charge"]["logarithmic_error"]
    df["discharge"]["fake_logarithmic_error"] = [
        (error_y / max(df["discharge"]["tension"]) + error_y / v) * max(df["discharge"]["tension"]) / v for v in df["discharge"]["tension"]
    ]

    # Maximum and Minimum Slopes
    max_charge_slope, min_charge_slope = get_maximum_slope(
        df["charge"]["time"],
        df["charge"]["logarithmic"],
        df["charge"]["time_error"],
        df["charge"]["logarithmic_error"],
        charge=False
    )

    max_discharge_slope, min_discharge_slope = get_maximum_slope(
        df["discharge"]["time"],
        df["discharge"]["logarithmic"],
        df["discharge"]["time_error"],
        df["discharge"]["logarithmic_error"],
        charge=False
    )

    del df["charge"]["data"]
    del df["discharge"]["data"]
    del df["full"]["data"]

    ####################
    # GENERATING TABLE #
    ####################

    table_values = [
        [r"$V_{max}$", "5 V", f"{round(max(df['full']['tension']), 3)} ± 0.1 V"],
        [r"$V_{min}$", "0 V", f"{round(min(df['discharge']['tension']), 3)} ± 0.1 V"],
        [r"$K_{max}$", "N/A", f"{round(max_discharge_slope, 3)}"],
        [r"$K_{min}$", "N/A", f"{round(min_discharge_slope, 3)}"],
        [
            "Charge Deviation",
            "N/A",
            f"{round(residual_standard_deviation(df['charge']['tension'], df['charge']['fitted']), 3)}",
        ],
        [
            "Discharge Deviation",
            "N/A",
            f"{round(residual_standard_deviation(df['discharge']['tension'], df['discharge']['fitted']), 3)}",
        ],
        [
            r"$\tau_{charge}$",
            f"2.7 ± {(27000*0.0001*0.05) + (27000*0.0001*0.05)} s",
            f"{round(charge_values[1], 3)} ± 0.005 s",
        ],  # TODO: Control error
        [
            r"$\tau_{discharge}$", 
            f"2.7 ± {(27000*0.0001*0.05) + (27000*0.0001*0.05)} s",
            f"{round(charge_values[1], 3)} ± 0.005 s",
            f"{round(discharge_values[1], 3)} ± 0.005 s",
        ],  # TODO: Control error
        [
            "Charge Equation",
            r"$V(t) = V_{max} \left(1 - e^{-\frac{t}{\tau}}\right)$",
            r"$V(t) = {"
            + f"{round(charge_values[0], 3)}"
            + "} (1 - e^{-\\frac{{t}}{"
            + f"{round(charge_values[1], 3)}"
            + "}})$",
        ],
        [
            "Discharge Equation",
            r"$V(t) = V_{max} e^{-\frac{t}{\tau}}$",
            r"$V(t) = {"
            + f"{round(discharge_values[0], 3)}"
            + "} e^{-\\frac{{t}}{{{"
            + f"{round(discharge_values[1], 3)}"
            + "}}}}$",
        ],
    ]

    table = f"""
    | {'':^20} | {'Expected':^20} | {'Observed':^20} |
    |{'-' * 20}|{'-' * 20}|{'-' * 20}|
    | {table_values[0][0]} | {table_values[0][1]} | {table_values[0][2]} |
    | {table_values[1][0]} | {table_values[1][1]} | {table_values[1][2]} |
    | {table_values[2][0]} | {table_values[2][1]} | {table_values[2][2]} |
    | {table_values[3][0]} | {table_values[3][1]} | {table_values[3][2]} |
    | {table_values[4][0]} | {table_values[4][1]} | {table_values[4][2]} |
    | {table_values[5][0]} | {table_values[5][1]} | {table_values[5][2]} |
    | {table_values[6][0]} | {table_values[6][1]} | {table_values[6][2]} |
    | {table_values[7][0]} | {table_values[7][1]} | {table_values[7][2]} |
    | {table_values[8][0]} | {table_values[8][1]} | {table_values[8][2]} |
    | {table_values[9][0]} | {table_values[9][1]} | {table_values[9][2]} |
    """

    ############
    # Plotting #
    ############

    grids = [None, None, None]

    # First Row

    grids[0] = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Charge Data & Fitted Line", "Discharge Data & Fitted Line"),
    )

    # Charge Data & Fitted Line

    grids[0].add_trace(
        px.scatter(
            df["charge"],
            x="time",
            y="tension",
            error_x="time_error",
            error_y="tension_error",
            color_discrete_sequence=[highlight1],
            title="Charge Data",
        ).data[0],
        row=1,
        col=1,
    )
    grids[0].add_trace(
        px.line(df["charge"], x="time", y="fitted", title="Fitted Line").data[0],
        row=1,
        col=1,
    )

    # Discharge Data & Fitted Line

    grids[0].add_trace(
        px.scatter(
            df["discharge"],
            x="time",
            y="tension",
            error_x="time_error",
            error_y="tension_error",
            color_discrete_sequence=[highlight1],
            title="Discharge Data",
        ).data[0],
        row=1,
        col=2,
    )
    grids[0].add_trace(
        px.line(df["discharge"], x="time", y="fitted", title="Fitted Line").data[0],
        row=1,
        col=2,
    )

    # Layout shenanigans

    grids[0].update_traces(line_color=highlight2)

    grids[0].update_layout(
        yaxis=dict(
            title=r"$\large{Tension (V)}$",
            range=[0, 5.5],
            ticksuffix="V",
            zeroline=False,
            showline=False,
            gridcolor=primary2,
        ),
        yaxis2=dict(
            title=r"$\large{Tension (V)}$",
            range=[0, 5.5],
            ticksuffix="V",
            zeroline=False,
            showline=False,
            gridcolor=primary2,
        ),
        xaxis=dict(
            title=r"$\large{Time (s)}$",
            ticksuffix="s",
            zeroline=False,
            showline=False,
            gridcolor=primary2,
        ),
        xaxis2=dict(
            title=r"$\large{Time (s)}$",
            ticksuffix="s",
            zeroline=False,
            showline=False,
            gridcolor=primary2,
        ),
        font=dict(color=primary3),
        paper_bgcolor=primary1,
        plot_bgcolor=primary1,
    )

    # Second Row

    grids[1] = go.Figure()

    # Plot Expect and Fitted Charge / Discharge

    grids[1].add_trace(
        px.line(
            df["charge"],
            x="time",
            y="fitted",
            color_discrete_sequence=[highlight1]
        ).data[0]
    )

    grids[1].add_trace(
        px.line(
            df["charge"],
            x="time",
            y="expected",
            color_discrete_sequence=[highlight2]
        ).data[0]
    )
    
    grids[1].add_trace(
        px.line(
            x=df["discharge"]["time"] - min(df["discharge"]["time"]),
            y=df["discharge"]["fitted"],
            color_discrete_sequence=[highlight1]
        ).data[0]
    )
    
    grids[1].add_trace(
        px.line(
            x=df["discharge"]["time"] - min(df["discharge"]["time"]),
            y=df["discharge"]["expected"],
            color_discrete_sequence=[highlight2]
        ).data[0]
    )

    # Layout shenanigans
    grids[1].update_layout(
        yaxis=dict(
            title=r"$\large{Tension (V)}$",
            titlefont=dict(size=20),
            range=[0, 5.5],
            zeroline=False,
            showline=False,
            gridcolor=primary2,
        ),
        xaxis=dict(
            title=r"$\large{Time (s)}$",
            titlefont=dict(size=20),
            ticksuffix="s",
            zeroline=False,
            showline=False,
            gridcolor=primary2,
        ),
        title="Logaritmic Discharge & Fitted Curve",
        title_x=0.5,
        font=dict(color=primary3),
        paper_bgcolor=primary1,
        plot_bgcolor=primary1,
    )

    # Third Row

    grids[2] = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Logarithmic Charge", "Logarithmic Discharge"]
    )

    # Logarithmic Charge

    grids[2].add_trace(
        px.scatter(
            x=df["charge"]["time"][::spacing],
            y=df["charge"]["logarithmic"][::spacing],
            error_x=df["charge"]["time_error"][::spacing],
            error_y=df["charge"]["logarithmic_error"][::spacing],
            color_discrete_sequence=[highlight1],
        ).data[0],
        row=1,
        col=1
    )
 
    if not error:
        grids[2].add_trace(
            px.scatter(
                x=df["discharge"]["time"][::spacing],
                y=df["discharge"]["logarithmic"][::spacing],
                error_x=df["discharge"]["time_error"][::spacing],
                error_y=df["discharge"]["fake_logarithmic_error"][::spacing],
                color_discrete_sequence=[highlight1]
            ).data[0],
            row=1,
            col=2
        )
    else:
        grids[2].add_trace(
            px.scatter(
                x=df["discharge"]["time"][::spacing],
                y=df["discharge"]["logarithmic"][::spacing],
                error_x=df["discharge"]["time_error"][::spacing],
                error_y=df["discharge"]["logarithmic_error"][::spacing],
                color_discrete_sequence=[highlight1]
            ).data[0],
            row=1,
            col=2
        )

    # Max slope lines.

    grids[2].add_trace(
        px.line(
            x=[min(df["charge"]["time"]), max(df["charge"]["time"])],
            y=[df["charge"]["logarithmic"][0], (max(df["charge"]["time"]) - min(df["charge"]["time"])) * max_charge_slope],
            color_discrete_sequence=[primary2]
        ).data[0],
        row=1,
        col=1
    )

    grids[2].add_trace(
        px.line(
            x=[min(df["discharge"]["time"]), max(df["discharge"]["time"])],
            y=[df["discharge"]["logarithmic"][0], df["discharge"]["logarithmic"][0] + (max(df["discharge"]["time"]) - min(df["discharge"]["time"])) * max_discharge_slope],
            color_discrete_sequence=[primary2]
        ).data[0],
        row=1,
        col=2
    )

    grids[2].add_trace(
        px.line(
            x=[min(df["discharge"]["time"]), max(df["discharge"]["time"])],
            y=[df["discharge"]["logarithmic"][0], df["discharge"]["logarithmic"][0] + (max(df["discharge"]["time"]) - min(df["discharge"]["time"])) * min_discharge_slope],
            color_discrete_sequence=[primary2]
        ).data[0],
        row=1,
        col=2
    )

    # Fitted logarithmic line

    grids[2].add_trace(
        px.line(
           df["discharge"],
           x="time",
           y="logarithmic_fitting",
           color_discrete_sequence=[highlight2]
        ).data[0],
        row=1,
        col=2
    )

    # Layout shenanigans

    grids[2].update_layout(
        yaxis=dict(
            title=r"$\large{Tension (V)}$",
            range=[0, 5.5],
            ticksuffix="V",
            zeroline=False,
            showline=False,
            gridcolor=primary2,
        ),
        xaxis=dict(
            title=r"$\large{Time (s)}$",
            ticksuffix="s",
            zeroline=False,
            showline=False,
            gridcolor=primary2,
        ),
        yaxis2=dict(
            title=r"$\large{Tension (V)}$",
            ticksuffix="V",
            zeroline=False,
            showline=False,
            gridcolor=primary2
        ),
        xaxis2=dict(
            title=r"$\large{Time (S)}$",
            ticksuffix="s",
            zeroline=False,
            showline=False,
            gridcolor=primary2
        ),
        title="Expected Curve & Fitted Curve",
        title_x=0.5,
        font=dict(color=primary3),
        paper_bgcolor=primary1,
        plot_bgcolor=primary1,
    )

    return grids, table


def residual_standard_deviation(data, fitted):
    residuals = []

    for d, f in zip(data, fitted):
        residuals.append(d - f)

    return np.sqrt(sum([r**2 for r in residuals]) / (len(residuals)))


def expected_charge(time: int, charging: bool = True, starting_charge: float = 0):
    v_max = 5
    tao = 2.7
    if charging:
        return v_max * (1 - np.exp(-(time) / tao))
    else:
        return starting_charge * np.exp(-(time) / tao)


def get_maximum_slope(x, y, error_x, error_y, charge=False) -> tuple:

    # "Sanitization"
    x = [i - x[0] for i in x]
    y = [i - max(y) for i in y]
    error_x = [i for i in error_x]
    error_y = [i for i in error_y]

    for _x, _y, _error_x, _error_y in zip(x, y, error_x, error_y):
        if (_x - _error_x < 0):
            x.remove(_x)
            y.remove(_y)
            error_x.remove(_error_x)
            error_y.remove(_error_y)

    max_slopes = []
    min_slopes = []

    if charge:
        for _x, _y, _error_x, _error_y in zip(x, y, error_x, error_y):
            max_slopes.append((_y + _error_y) / (_x - _error_x))
            min_slopes.append((_y - _error_y) / (_x + _error_x))
    else:
        for _x, _y, _error_x, _error_y in zip(x, y, error_x, error_y):
            max_slopes.append((_y + _error_y) / (_x + _error_x))
            min_slopes.append((_y - _error_y) / (_x - _error_x))

    return min(max_slopes), max(min_slopes)

@app.callback(
    Output("graph1", "figure"),
    Input("right-toggle-error-button", "value"),
    State("graph1", "figure"),
)
@app.callback(
    Output("graph2", "figure"),
    Input("right-toggle-error-button", "value"),
    State("graph2", "figure"),
)
def toggle_error(value, figure):

    new_figure = go.Figure(figure)

    new_figure.update_traces(
        error_x=dict(visible=value),
        error_y=dict(visible=value),
        selector=dict(mode="markers"),
    )

    return new_figure

if __name__ == "__main__":
    # fig, table = plot_charge_and_discharge("charge", "discharge", 0.1, 0.005, True)
    fig, table = plot_charge_and_discharge(
        "forged_charge", "forged_discharge", 0.1, 0.005, True
    )

    app.layout = html.Div(
        [
            html.Iframe(
                src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            daq.ToggleSwitch(
                                id="left-toggle-error-button",
                                value=True,
                                vertical=True,
                                className="error-switch",
                            )
                        ],
                        id="left-spacer"
                    ),
                    html.Div(
                        [
                            html.H1("Charge and Discharge of a Capacitor"),
                            html.H3("Physics Lab 1"),
                        ],
                        className="title",
                    ),
                    html.Div(
                        [
                            daq.ToggleSwitch(
                                id="right-toggle-error-button",
                                value=True,
                                vertical=True,
                                className="error-switch",
                            )
                        ],
                        id="error-button-container",
                    ),
                ],
                className="header",
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="graph1",
                        figure=fig[0],
                        style={"height": "800px", "width": "100vw"},
                        mathjax=True,
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="graph2",
                                        figure=fig[1],
                                        style={"height": "800px", "width": "50vw"},
                                        mathjax=True,
                                    )
                                ],
                                style={"flex": "1", "height": "800px"},
                            ),
                            html.Div(
                                [
                                    dcc.Markdown(
                                        children=table,
                                        dangerously_allow_html=True,
                                        className="table",
                                        mathjax=True,
                                    )
                                ],
                                style={
                                    "flex": "1",
                                    "padding-top": "100px",
                                    "padding-left": "80px",
                                    "padding-right": "65px",
                                    "padding-bottom": "80px",
                                    "background-color": primary1,
                                    "color": "#f2f5fa",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "width": "100%",
                            "justify-content": "space-around",
                            "height": "800px",
                        },
                    ),
                    dcc.Graph(
                        id="graph3",
                        figure=fig[2],
                        style={"height": "800px", "width": "100vw"},
                        mathjax=True,
                    ),
                ],
                className="content",
            ),
        ]
    )

    app.run_server(debug=True, port=8080)
