import dash
from dash import dcc, html, dash_table
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy

app = dash.Dash(__name__)

def plot_charge_and_discharge():
    # Reading Data

    df = {
        'charge': {
            'data': pd.read_csv('data/forged_charge.csv')
        },
        'discharge': {
            'data': pd.read_csv('data/forged_discharge.csv')
        },
        'full': {
            'data': pd.concat([pd.read_csv('data/charge.csv'), pd.read_csv('data/discharge.csv')])
        }
    }

    # Getting Tension

    df['charge']['tension'] = df['charge']['data']['Value'].astype(float)
    df['discharge']['tension'] = df['discharge']['data']['Value'].astype(float)
    df['full']['tension'] = pd.concat([df['charge']['tension'], df['discharge']['tension']])

    # Getting Time

    df['charge']['time'] = df['charge']['data']['Parameter'].astype(float) / 1000
    df['discharge']['time'] = df['discharge']['data']['Parameter'].astype(float) / 1000
    df['full']['time'] = pd.concat([df['charge']['time'], df['discharge']['time']])

    # Fitting Charge Data

    charge_values, _ = scipy.optimize.curve_fit(
        lambda t, a, b: a * (1 - np.exp(-t / b)),
        df['charge']['time'],
        df['charge']['tension']
    )

    df['charge']['fitted'] = charge_values[0] * (1 - np.exp(-df['charge']['time'] / charge_values[1]))
    
    # Fitting Discharge Data

    discharge_values, _ = scipy.optimize.curve_fit(
        lambda t, a, b, c: a * np.exp(-(t + c) / b),
        df['discharge']['time'] - df['discharge']['time'][0],
        df['discharge']['tension']
    )

    df['discharge']['fitted'] = discharge_values[0] * np.exp(-(df['discharge']['time'] - df['discharge']['time'][0] + discharge_values[2]) / discharge_values[1])

    df['full']['fitted'] = list(df['charge']['fitted']) + list(df['discharge']['fitted'])

    # Expected Curve

    df['charge']['expected'] = [expected_charge(i - df['charge']['time'][0]) for i in df['charge']['time']]
    df['discharge']['expected'] = [expected_charge(i - df['discharge']['time'][0], False, 4.9) for i in df['discharge']['time']]
    df['full']['expected'] = df['charge']['expected'] + df['discharge']['expected']

    # Error
    df['charge']['time_error'] = [0.005 for _ in df['charge']['time']]
    df['charge']['tension_error'] = [0.1 for _ in df['charge']['tension']]

    df['discharge']['time_error'] = [0.005 for _ in df['discharge']['time']]
    df['discharge']['tension_error'] = [0.1 for _ in df['discharge']['tension']]

    # Logaroithmic Scale
    df['charge']['logarithmic'] = np.log(df['charge']['tension'])
    df['discharge']['logarithmic'] = np.log(max(df['discharge']['tension'])/df['discharge']['tension'])
    df['full']['logarithmic'] = np.log(df['full']['tension'])

    # Logarithmic Fitting
    df['charge']['logarithmic_fitting'] = np.log(df['charge']['fitted'])
    df['discharge']['logarithmic_fitting'] = np.log(max(df['discharge']['fitted'])/df['discharge']['fitted'])
    df['full']['logarithmic_fitting'] = np.log(df['full']['fitted'])

    # Logarithmic Error
    df['charge']['logarithmic_error'] = [0.1/v if v != 0 else 1 for v in df['charge']['tension']]
    df['discharge']['logarithmic_error'] = [0.1/v if v != 0 else 1 for v in df['discharge']['tension']]
    df['full']['logarithmic_error'] = [0.1/v if v != 0 else 1 for v in df['full']['tension']]

    # Maximum and Minimum Slopes
    max_slope, min_slope = get_maximum_slope(df['discharge']['time'], df['discharge']['logarithmic'], df['discharge']['time_error'], df['discharge']['logarithmic_error'])

    del df['charge']['data']
    del df['discharge']['data']
    del df['full']['data']

    ####################
    # GENERATING TABLE #
    ####################

    table_values = [
        [r'$V_{max}$', "5 V", f"{round(max(df['full']['tension']), 3)} ± 0.1 V"],
        [r'$V_{min}$', "0 V", f"{round(min(df['discharge']['tension']), 3)} ± 0.1 V"],
        [r'$K_{max}$', "N/A", f"{round(max_slope, 3)}"],
        [r'$K_{min}$', "N/A", f"{round(min_slope, 3)}"],
        ['Charge Deviation', "N/A", f"{round(residual_standard_deviation(df['charge']['tension'], df['charge']['fitted']), 3)}"],
        ['Discharge Deviation', "N/A", f"{round(residual_standard_deviation(df['discharge']['tension'], df['discharge']['fitted']), 3)}"],
        [r'$\tau_{charge}$', "2.7 s", f"{round(charge_values[1], 3)} ± 0.005 s"], # TODO: Control error
        [r'$\tau_{discharge}$', "2.7 s", f"{round(discharge_values[1], 3)} ± 0.005 s"], # TODO: Control error
        ['Charge Equation',
         r'$V(t) = V_{max} \left(1 - e^{-\frac{t}{\tau}}\right)$',
         r"$V(t) = {"+f"{round(charge_values[0], 3)}"+"} (1 - e^{-\\frac{{t}}{" + f"{round(charge_values[1], 3)}" + "}})$"],
        ['Discharge Equation',
         r'$V(t) = V_{max} e^{-\frac{t}{\tau}}$',
         r"$V(t) = {"+f"{round(discharge_values[0], 3)}"+"} e^{-\\frac{{t}}{{{"+f"{round(discharge_values[1], 3)}"+"}}}}$"]
    ]

    table = f"""
    | {'':^20} | {'Expected':^20} | {'Observed':^20} |
    |{'-'*20}|{'-'*20}|{'-'*20}|
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

    ## First Row

    grids[0] = make_subplots(
        rows=1,
        cols=2, 
        subplot_titles=(
            'Charge Data & Fitted Line',
            'Discharge Data & Fitted Line'
        )
    )

    # Charge Data & Fitted Line

    grids[0].add_trace(px.scatter(df['charge'], x='time', y='tension', error_x='time_error', error_y='tension_error', title="Charge Data").data[0], row=1, col=1)
    grids[0].add_trace(px.line(df['charge'], x="time", y="fitted", title='Fitted Line').data[0], row=1, col=1)
    
    # Discharge Data & Fitted Line

    grids[0].add_trace(px.scatter(df['discharge'], x="time", y="tension", error_x="time_error", error_y="tension_error", title='Discharge Data').data[0], row=1, col=2)
    grids[0].add_trace(px.line(df['discharge'], x="time", y="fitted", title='Fitted Line').data[0], row=1, col=2)

    # Layout shenanigans

    grids[0].update_traces(
        line_color='red'
    )

    grids[0].update_layout(
        yaxis = dict(
                title = 'Tension (V)',
                range = [0, 5.5],
                ticksuffix = 'V'
            ),
        yaxis2 = dict(
                title = 'Tension (V)',
                range = [0, 5.5],
                ticksuffix = 'V'
            ),
        xaxis = dict(
                title = 'Time (s)',
                ticksuffix = 's'
            ),
        xaxis2 = dict(
                title = 'Time (s)',
                ticksuffix = 's'
            ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        label="Toggle Error",
                        method="update",
                        args2=[{
                            "error_y.visible": [False, False, False, False],
                            "error_x.visible": [False, False, False, False]
                            }],
                        args=[{"error_y.visible": [True, False, True, False],
                               "error_x.visible": [True, False, True, False]
                            }],
                    ),
                ],
                showactive=True,
                x=0.5,
                y=1.1,
                xanchor="center",
                yanchor="top"
            )
        ]
    )
    
    ## Second Row

    grids[1] = make_subplots( 
        rows=1,
        cols=1,
        subplot_titles=(
            'Logarithmic Charge Fitted Curve',
            'Logarithmic Discharge Fitted Curve'
        )
    )

    # Data Sample in Logarithmic Scale (Plotting limited number of samples)

    grids[1].add_trace(px.scatter(
        df['discharge'],
        x='time',
        y='logarithmic',
        error_x='time_error',
        error_y='logarithmic_error'
    ).data[0], row=1, col=1)
    
    # Expected curve given tao
    grids[1].add_trace(px.line(
        x=[min(df['discharge']['time']), max(df['discharge']['time'])],
        y=[0,(max(df['discharge']['time'])-min(df['discharge']['time']))/discharge_values[1]],
        color_discrete_sequence=['red', 'red']
    ).data[0], row=1, col=1)

    # Maximum and Minimum Slopes
    grids[1].add_trace(px.line(
        x=[min(df['discharge']['time'])       , max(df['discharge']['time'])],
        y=[min(df['discharge']['logarithmic']), (max(df['discharge']['time']) - min(df['discharge']['time'])) * max_slope],
        color_discrete_sequence=['purple', 'purple']
    ).data[0], row=1, col=1)

    grids[1].add_trace(px.line(
        x=[min(df['discharge']['time'])       , max(df['discharge']['time'])],
        y=[min(df['discharge']['logarithmic']), (max(df['discharge']['time']) - min(df['discharge']['time'])) * min_slope],
        color_discrete_sequence=['purple', 'purple']
    ).data[0], row=1, col=1)
    
    # Layout shenanigans
    grids[1].update_layout(
        yaxis = dict(
                title = r'$ln{\frac{V_{max}}{V(t)}}$',
        ),
        xaxis = dict(
                title = 'Time (s)',
                ticksuffix = 's'
        ),
        yaxis2 = dict(
                title = 'Tension (Logarithmic)'
        ),
        xaxis2 = dict(
                title = 'Time (s)',
                ticksuffix = 's'
        )
    )

    ## Third Row

    grids[2] = go.Figure()

    # Expected Curve & Fitted Curve

    grids[2].add_trace(px.line(df['full'], x="time", y="expected", color_discrete_sequence=['green', 'green'], title='Expected Curve').data[0])
    grids[2].add_trace(px.line(df['full'], x="time", y="fitted", color_discrete_sequence=['red', 'red'], title='Fitted Curve').data[0])

    # Layout shenanigans

    grids[2].update_layout(
        yaxis = dict(
                title = 'Tension (V)',
                range = [0, 5.5],
                ticksuffix = 'V'
            ),
        xaxis = dict(
                title = 'Time (s)',
                ticksuffix = 's'
            ),
        title = 'Expected Curve & Fitted Curve',
        showlegend = True
    )


    return grids, table

def residual_standard_deviation(data, fitted):
    residuals = []
    
    for d, f in zip(data, fitted):
        residuals.append(d - f)
    
    return np.sqrt(sum([r**2 for r in residuals]) / (len(residuals)))


def expected_charge(time: int, charging: bool = True, starting_charge: int = 0):
    v_max = 5
    tao = 2.7
    if charging:
        return v_max * (1 - np.exp(-(time) / tao))
    else:
        return starting_charge * np.exp(-(time) / tao)

def get_maximum_slope(x, y, error_x, error_y) -> list[float]:

    # "Sanitization"
    x = [i - x[0] for i in x]
    y = [i for i in y]
    error_x = [i for i in error_x]
    error_y = [i for i in error_y]

    for _x, _y, _error_x, _error_y in zip(x, y, error_x, error_y):
        if (_x - _error_x < 0) or (_y - _error_y < 0):
            x.remove(_x)
            y.remove(_y)
            error_x.remove(_error_x)
            error_y.remove(_error_y)

    max_slopes = []
    min_slopes = []

    for _x, _y, _error_x, _error_y in zip(x, y, error_x, error_y):
        max_slopes.append((_y + _error_y) / (_x - _error_x))
        min_slopes.append((_y - _error_y) / (_x + _error_x))

    return min(max_slopes), max(min_slopes)

if __name__ == "__main__":
    fig, table = plot_charge_and_discharge()

    app.layout = html.Div([
        html.Iframe(
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML",
            style={"display": "none"}
        ),

        dcc.Graph(
            id='graph1',
            figure=fig[0],
            style={'height': '800px', 'width': '100vw'}
        ),
        
        html.Div([
            html.Div([
                dcc.Graph(id='graph2',
                    figure=fig[1],
                    style={'height': '800px', 'width': '50vw'},
                    mathjax=True
                )
            ], style={'flex': '1', 'height': '800px'}),

            html.Div([
                dcc.Markdown(
                    children=table,
                    dangerously_allow_html=True,
                    style={"font-size": "20px", "white-space": "pre-line"},
                    mathjax=True
                )
            ], style={'flex': '1', 'padding-top': '100px', 'padding-left': '80px', 'padding-right': '65px', 'padding-bottom': '80px'})

        ], style={'display': 'flex', 'width': '100%', 'justify-content': 'space-around', 'height': '800px'}),

        dcc.Graph(id='graph3',  
                    figure=fig[2],
                    style={'height': '800px', 'width': '100vw'},
                    mathjax=True
                )
    ], className='scrollable-container'
    )
    
    app.run_server(debug=True)