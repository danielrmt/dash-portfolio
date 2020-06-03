import numpy as np
import pandas as pd
from pandas_datareader import data as wb

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table as dt
from dash_table.Format import Format, Scheme, Sign

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff

from data_funcs import assets_df, get_quotes


#
pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h'),
        colorway=["#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                  "#0072B2", "#D55E00", "#CC79A7", "#999999"]
    )
)
pio.templates.default = 'custom'

plot_style = {'height': '80vh'}

#
assets = assets_df('IBRA')
is_main_ticker = assets.groupby('base_ticker')['part'].max()
selected_assets = assets[assets['part'].isin(is_main_ticker)].index[:20]

#
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server
app.title = 'Portfolio'

#
navbar = dbc.NavbarSimple(
    children=[
        dbc.Button('Selecionar ativos', id='assets_open', className='ml-auto'),
    ],
    brand=app.title,
    brand_href="#",
    color='dark',
    dark=True
)




# SEARCH MODAL
assets_modal = dbc.Modal([
    dbc.ModalHeader([
        'Selecionar ativos',
    ]),
    dbc.ModalBody([
        dt.DataTable(
            id='assets_table',
            data=assets.to_dict('records'),
            columns=[{"name": i, "id": i} for i in assets.columns],
            selected_rows=[i for i in selected_assets],
            style_as_list_view=True, style_header={'fontWeight': 'bold'},
            row_selectable='multi')
    ]),
    dbc.ModalFooter([
        dbc.Button('Fechar', id='assets_close', className='ml-auto')
    ])
], id='assets_modal', size="xl", scrollable=True)


#
tabs = dbc.Tabs([
    dbc.Tab([
        html.H4('Log-Retornos'),
        dcc.Graph('logreturns_plot', style=plot_style)
    ], label='Retornos'),
    #
    dbc.Tab([
        html.H4('Distribuição dos retornos'),
        dcc.Graph('logreturns_ridge_plot', style=plot_style)
    ], label='Distribuição'),
    #
    dbc.Tab([
        html.H4('Matriz de Covariância'),
        dcc.Graph('covmatrix_plot', style=plot_style)
    ], label='Covariância'),
    #
])

#
stores = html.Div([
    dcc.Store(id=f"{s}_data")
    for s in['assets', 'prices', 'logreturns', 'covmatrix']
])


# LAYOUT
app.layout = html.Div([
    navbar,
    html.Div([
        tabs,
    ], className='container-fluid'),
    assets_modal,
    stores
])



#
@app.callback(
    Output("assets_modal", "is_open"),
    [Input("assets_open", "n_clicks"),
     Input("assets_close", "n_clicks")],
    [State("assets_modal", "is_open")],
)
def toggle_search_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [Output('prices_data', 'data'),
     Output('logreturns_data', 'data'),
     Output('covmatrix_data', 'data')],
    [Input('assets_table', 'data'),
     Input('assets_table', 'selected_rows')]
)
def update_data(assets_table, selected_rows):
    df = pd.DataFrame(assets_table)
    assets = df[df.index.isin(selected_rows)][['ticker', 'part']]
    tickers = assets['ticker'].values
    prices = get_quotes(tickers)
    logreturns = (
        np.log(prices.set_index('Date'))
        .resample('MS')
        .last()
        .diff()
        .reset_index()
    )
    
    covmatrix = logreturns.cov().reset_index()
    return prices.to_dict('records'), \
        logreturns.to_dict('records'), \
        covmatrix.to_dict('records')


@app.callback(
    Output('logreturns_plot', 'figure'),
    [Input('logreturns_data', 'data')]
)
def update_logreturns_plot(logreturns):
    df = pd.DataFrame(logreturns).melt('Date').sort_values('Date')
    fig = px.line(df, x='Date', y='value', line_group='variable')
    return fig


@app.callback(
    Output('logreturns_ridge_plot', 'figure'),
    [Input('logreturns_data', 'data')]
)
def update_logreturns_ridge_plot(logreturns):
    df = pd.DataFrame(logreturns).set_index('Date')
    fig = px.violin(df.melt(), x='value', y='variable')
    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    return fig


@app.callback(
    Output('covmatrix_plot', 'figure'),
    [Input('covmatrix_data', 'data')]
)
def update_covmatrix_plot(covmatrix):
    df = pd.DataFrame(covmatrix).set_index('index')
    cols = df.columns.values.tolist()
    z = df.values.tolist()
    ztext = np.round(z, 5)
    fig = ff.create_annotated_heatmap(
        z, x=cols, y=cols, annotation_text=ztext, colorscale='Viridis'
    )
    fig.update_yaxes(autorange="reversed")
    return fig


#
if __name__ == "__main__":
    app.run_server(debug=True)
