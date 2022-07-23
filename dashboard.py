import pandas as pd
import numpy as np

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output,State

import plotly.graph_objects as go
import os

from scipy.integrate import odeint
from scipy.optimize import curve_fit

df_input_large=pd.read_csv('COVID_final_set.csv',sep=';')


infection = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
infection = infection.drop(["Province/State", "Lat", "Long"], axis =1)
infection = infection.groupby(["Country/Region"]).sum()

recovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
recovered = recovered.drop(["Province/State","Lat", "Long"], axis=1)
recovered = recovered.groupby(["Country/Region"]).sum()

deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
deaths = deaths.drop(["Province/State","Lat", "Long"], axis=1)
deaths = deaths.groupby(["Country/Region"]).sum()

data_frame = pd.DataFrame([infection[infection.columns[-1]], recovered[recovered.columns[-1]], deaths[deaths.columns[-1]]])
data_frame = data_frame.T

data_frame.columns = ['infections','recover','dead']
data_frame['active'] = data_frame['infections'] - data_frame['recover'] - data_frame['dead']
world_data = data_frame

dates = np.array('2020-01-22', dtype=np.datetime64) + np.arange(len(infection.columns))



fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    ## Select Countries
    '''),


    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['Germany'], # which are pre-selected
        multi=True
    ),

    dcc.Markdown('''
        ## Select Timeline 
        '''),


    dcc.Dropdown(
    id='doubling_time',
    options=[
        {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
        {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
        {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
    ],
    value='confirmed',
    multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope'),

    html.H4('SIR Model', style ={'textAlign' : 'center'}),
    html.Div(
        dcc.Dropdown(id = 'SIR_country',
        options=[{'label': each,'value':each} for each in df_input_large['country'].unique()],
        value="Germany")),

        html.Div( dcc.Graph(id = 'SIR_figure')),
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')])
def update_figure(country_list,show_doubling):


    if 'doubling_rate' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
              }
    else:
        my_yaxis={'type':"log",
                  'title':'Infected people'
              }


    traces = []
    for each in country_list:

        df_plot=df_input_large[df_input_large['country']==each]

        if show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
       #print(show_doubling)


        traces.append(dict(x=df_plot.date,
                                y=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                name=each
                        )
                )

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis
        )
    }


@app.callback([Output('SIR_figure', 'figure')],
[Input('SIR_country', 'value')])

def sir_figure(country):

    length = 30 # duration for simulations

    confirmed_data = infection.loc[f'{country}'].values
    recovered_data = recovered.loc[f'{country}'].values
    confirmed_data = confirmed_data[-100 : ]
    recovered_data = recovered_data[-100 : ]

    pre_dates = dates[-30: ]

    N = 1000000
    I_0 = confirmed_data[0]
    R_0 = recovered_data[0]
    S_0 = N - R_0 - I_0

    def SIR(y, t, beta, gamma):
        S = y[0]
        I = y[1]
        R = y[2]
        return -beta*S*I/N, (beta*S*I)/N-(gamma*I), gamma*I

    def fit_odeint(t,beta, gamma):
        return odeint(SIR,(S_0,I_0,R_0), t, args = (beta,gamma))[:,1]

    t = np.arange(len(confirmed_data))
    params, cerr = curve_fit(fit_odeint,t, confirmed_data)
    beta,gamma = params
    prediction = list(fit_odeint(t,beta,gamma))


    fig = go.Figure()
    fig.add_trace(go.Scatter(x= pre_dates, y= prediction, mode='lines+markers', name='Simulated'))
    fig.add_bar(x = pre_dates, y= confirmed_data, name = "Actual")
    fig.update_layout(height = 700)


    return [fig]


if __name__ == '__main__':

    app.run_server(debug=True, use_reloader=False)