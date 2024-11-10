import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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
            'data': pd.read_csv('data/charge.csv')
        },
        'discharge': {
            'data': pd.read_csv('data/discharge.csv')
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

    # Getting Maximum Slopes
    """
    slopes = get_maximum_slope()

    charge_slope_x = np.linspace(0, 2500, 100)
    charge_slope_y = slopes[0] * charge_slope_x

    discharge_slope_x = np.linspace(discharge_time[0], discharge_time[0] + 2500, 100)
    discharge_slope_y = slopes[1] * discharge_slope_x
    discharge_slope_y -= min(discharge_slope_y)
    """
    
    # Expected Curve

    df['charge']['expected'] = [expected_charge(i - df['charge']['time'][0]) for i in df['charge']['time']]
    df['discharge']['expected'] = [expected_charge(i - df['discharge']['time'][0], False, 4.9) for i in df['discharge']['time']]
    df['full']['expected'] = df['charge']['expected'] + df['discharge']['expected']

    # Error
    df['charge']['time_error'] = [0.005 for _ in df['charge']['time']]
    df['charge']['tension_error'] = [t/20 for t in df['charge']['tension']]

    df['discharge']['time_error'] = [0.005 for _ in df['discharge']['time']]
    df['discharge']['tension_error'] = [t/20 for t in df['discharge']['tension']]

    del df['charge']['data']
    del df['discharge']['data']
    del df['full']['data']

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

    grids[1] = go.Figure()

    # Expected Curve & Fitted Curve

    grids[1].add_trace(px.line(df['full'], x="time", y="expected", title='Expected Curve').data[0])
    grids[1].add_trace(px.line(df['full'], x="time", y="fitted", title='Fitted Curve').data[0])
    grids[1].layout.title = 'Expected Curve & Fitted Curve'

    ## Third Row

    grids[2] = make_subplots( 
        rows=1,
        cols=2,
        subplot_titles=(
            'Logarithmic Charge Fitted Curve',
            'Logarithmic Discharge Fitted Curve'
        )
    )

    # Logarithmic Charge Fitted Curve
    grids[2].add_trace(px.line(df['charge'],    x="time", y="fitted").data[0], row=1, col=1)

    # Logarithmic Disharge Fitted Curve

    grids[2].add_trace(px.line(df['discharge'], x="time", y="fitted").data[0], row=1, col=2)
    
    # Layout shenanigans
    grids[2].update_layout(
        yaxis = dict(
                title = 'Tension (V)',
                range = [0, 5.5],
                ticksuffix = 'V'
        ),
        xaxis = dict(
                title = 'Time (s)',
                ticksuffix = 's',
                type = 'log'
        ),
        yaxis2 = dict(
                title = 'Tension (V)',
                range = [0, 5.5],
                ticksuffix = 'V'
        ),
        xaxis2 = dict(
                title = 'Time (s)',
                ticksuffix = 's',
                type = 'log'
        )
    )

    return grids

def expected_charge(time: int, charging: bool = True, starting_charge: int = 0):
    v_max = 5
    tao = 2.7
    if charging:
        return v_max * (1 - np.exp(-(time) / tao))
    else:
        return starting_charge * np.exp(-(time) / tao)

def get_maximum_slope() -> list[float]:
    charge_df = pd.read_csv('data/charge.csv')
    charge_df['Value'] = charge_df['Value'].astype(float)
    charge_slopes = [(charge_df['Value'][i] - charge_df['Value'][i - 1]) / (charge_df['Parameter'][i] - charge_df['Parameter'][i - 1]) for i in range(1, len(charge_df['Value']))]
    max_slope = max(charge_slopes)

    discharge_df = pd.read_csv('data/discharge.csv')
    discharge_df['Value'] = discharge_df['Value'].astype(float)
    discharge_slopes = [(discharge_df['Value'][i] - discharge_df['Value'][i - 1]) / (discharge_df['Parameter'][i] - discharge_df['Parameter'][i - 1]) for i in range(1, len(discharge_df['Value']))]
    min_slope = min(discharge_slopes)

    return max_slope, min_slope

if __name__ == "__main__":
    fig = plot_charge_and_discharge()

    app.layout = html.Div([
        dcc.Graph(id='graph1',
                  figure=fig[0],
                  style={'height': '800px', 'width': '100vw'}
                ),
        
        dcc.Graph(id='graph2',
                    figure=fig[1],
                    style={'height': '800px', 'width': '100vw'},
                ),

        dcc.Graph(id='graph3',  
                    figure=fig[2],
                    style={'height': '800px', 'width': '100vw'}
                )
    ], className='scrollable-container'
    )
    
    app.run_server(debug=True)