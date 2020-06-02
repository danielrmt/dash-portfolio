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


#
assets = assets_df('IBRA')
is_main_ticker = assets.groupby('base_ticker')['part'].max()
selected_assets = assets[assets['part'].isin(is_main_ticker)].index[:5]

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
        html.H4('Matriz de Covariância'),
        dt.DataTable(
            id='cov_matrix',
            data=[], columns=[],
            style_as_list_view=True, style_header={'fontWeight': 'bold'},
        )
    ], label='Covariância'),
    #
    dbc.Tab([
        html.H4('Cotações'),
        dt.DataTable(
            id='prices_table',
            data=[], columns=[],
            style_as_list_view=True, style_header={'fontWeight': 'bold'},
        )
    ], label='Cotações')
])


# LAYOUT
app.layout = html.Div([
    navbar,
    html.Div([
        tabs,
    ], className='container-fluid'),
    assets_modal
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

#
@app.callback(
    [Output('prices_table', 'data'),
     Output('prices_table', 'columns')],
    [Input('assets_table', 'data'),
     Input('assets_table', 'selected_rows')]
)
def update_prices(assets_table, selected_rows):
    df = pd.DataFrame(assets_table)
    tickers = df['ticker'][df.index.isin(selected_rows)].values
    df = get_quotes(tickers)
    cols = [{
        'name': s,
        'id': s
    } for s in df.columns]
    return df.to_dict('records'), cols

@app.callback(
    [Output('cov_matrix', 'data'),
     Output('cov_matrix', 'columns')],
    [Input('prices_table', 'data')]
)
def update_covmatrix(prices):
    logreturns = np.log(pd.DataFrame(prices).set_index('index')).diff() * 252
    covmatrix = logreturns.cov().reset_index()
    return covmatrix.to_dict('records'), \
        [{'name':s,'id':s} for s in covmatrix.columns]


#
if __name__ == "__main__":
    app.run_server(debug=True)
