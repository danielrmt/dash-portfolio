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

from sklearn.covariance import LedoitWolf, oas

from data_funcs import assets_df, get_quotes, bcb_sgs
from fin_funcs import MeanVariancePortfolio


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
selected_assets = assets[assets['part'].isin(is_main_ticker)].index[:12]

cdi = bcb_sgs('01/01/2010', '31/12/2020', CDI=12)
cdi['CDI'] = np.log(1 + cdi['CDI']/100)

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
        html.H4('Log-Retornos excedentes mensais'),
        dcc.Graph('logreturns_plot', style=plot_style)
    ], label='Retornos'),
    #
    dbc.Tab([
        html.H4('Distribuição dos retornos excedentes mensais'),
        dcc.Graph('logreturns_ridge_plot', style=plot_style)
    ], label='Distribuição'),
    #
    dbc.Tab([
        html.H4('Matriz de Covariância dos retornos excedentes mensais'),
        dbc.RadioItems(id='cov_method', value='cov', inline=True,
            options=[
                {'label': 'Padrão', 'value': 'cov'},
                {'label': 'Ledoit-Wolf Shrinkage', 'value': 'ledoit-wolf'},
                {'label': 'Oracle Approximating Shrinkage', 'value': 'oas'}
            ]),
        dcc.Graph('covmatrix_plot', style=plot_style)
    ], label='Covariância'),
    #
    dbc.Tab([
        html.H4('Fronteira eficiente de portfolios'),
        dbc.Row([
            dbc.Col([
                dcc.Graph('frontier_plot', style=plot_style)
            ], width=8),
            dbc.Col([
                dcc.Graph('weights_plot', style=plot_style)
            ], width=4)
        ])
        
    ], label='Fronteira')
])

#
stores = html.Div([
    dcc.Store(id=f"{s}_data")
    for s in['assets', 'prices', 'logreturns', 'covmatrix', 'frontier']
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
     Output('logreturns_data', 'data')],
    [Input('assets_table', 'data'),
     Input('assets_table', 'selected_rows')]
)
def update_data(assets_table, selected_rows):
    df = pd.DataFrame(assets_table)
    assets = df[df.index.isin(selected_rows)][['ticker', 'part']]
    assets['part'] = assets['part'] / assets['part'].sum()
    tickers = assets['ticker'].values
    prices = get_quotes(tickers)
    logreturns = (
        np.log(prices.set_index('Date'))
        .resample('MS')
        .last()
        .diff()
    )
    m_cdi = cdi.resample('MS').sum()
    m_cdi = logreturns.merge(m_cdi, left_index=True, right_index=True)['CDI']

    for t in tickers:
        logreturns[t] = logreturns[t] - m_cdi

    logreturns = logreturns.reset_index()

    return prices.to_dict('records'), \
        logreturns.to_dict('records')


@app.callback(
    [Output('covmatrix_data', 'data'),
     Output('assets_data', 'data')],
    [Input('logreturns_data', 'data'),
     Input('assets_table', 'data'),
     Input('assets_table', 'selected_rows'),
     Input('cov_method', 'value')]
)
def update_covmatrix(logreturns, assets_table, selected_rows, method):
    logreturns = pd.DataFrame(logreturns).set_index('Date')
    df = pd.DataFrame(assets_table)
    assets = df[df.index.isin(selected_rows)][['ticker', 'part']]
    assets['part'] = assets['part'] / assets['part'].sum()

    tickers = assets['ticker'].values
    if method == 'ledoit-wolf':
        covmatrix = LedoitWolf().fit(logreturns.dropna()).covariance_
        covmatrix = pd.DataFrame(covmatrix, index=tickers,
            columns=tickers)
    elif method == 'oas':
        covmatrix, x = oas(logreturns.dropna())
        covmatrix = pd.DataFrame(covmatrix, index=tickers,
            columns=tickers)
    else:
        covmatrix = logreturns.cov()

    assets['implied'] = covmatrix.values @ assets['part'].values

    return covmatrix.reset_index().to_dict('records'), \
        assets.to_dict('records')


@app.callback(
    Output('logreturns_plot', 'figure'),
    [Input('logreturns_data', 'data')]
)
def update_logreturns_plot(logreturns):
    df = pd.DataFrame(logreturns).melt('Date').sort_values('Date')
    fig = px.line(df, x='Date', y='value',
        facet_col='variable', facet_col_wrap=3,
        labels={'Date': '', 'value': '', 'variable': ''})
    fig.update_yaxes(matches=None, showticklabels=False)
    return fig


@app.callback(
    Output('logreturns_ridge_plot', 'figure'),
    [Input('logreturns_data', 'data')]
)
def update_logreturns_ridge_plot(logreturns):
    df = pd.DataFrame(logreturns).set_index('Date')
    fig = px.violin(df.melt(), x='value', y='variable',
        labels={'value': '', 'variable': ''})
    fig.update_traces(orientation='h', side='negative', width=3, points=False,
        meanline_visible=True)
    fig.update_traces()
    fig.update_yaxes(autorange="reversed")
    return fig


@app.callback(
    Output('covmatrix_plot', 'figure'),
    [Input('covmatrix_data', 'data')]
)
def update_covmatrix_plot(covmatrix):
    df = pd.DataFrame(covmatrix).set_index('index')
    cols = df.columns.values.tolist()
    z = df.values.tolist()
    ztext = np.round(z, 3)
    fig = ff.create_annotated_heatmap(
        z, x=cols, y=cols, annotation_text=ztext, colorscale='Viridis'
    )
    fig.update_yaxes(autorange="reversed")
    return fig

@app.callback(
    Output('frontier_data', 'data'),
    [Input('covmatrix_data', 'data'),
     Input('logreturns_data', 'data'),
     Input('assets_data', 'data')]
)
def update_frontier_data(covmatrix, logreturns, assets):
    assets = pd.DataFrame(assets)
    logreturns = pd.DataFrame(logreturns).set_index('Date')
    covmatrix = pd.DataFrame(covmatrix).set_index('index')

    ativos = assets.merge(
        logreturns.agg(['mean', 'std']).T,
        left_on='ticker', right_index=True
    )

    ativos = ativos.rename(columns={'mean': 'historical'})

    fronteira = (
        MeanVariancePortfolio(ativos['implied'], covmatrix, ativos['ticker'])
        .frontier(max=ativos['implied'].max() * 1.5)
        .sort_values(['mu'])
    )
    return fronteira.to_dict('records')


@app.callback(
    Output('frontier_plot', 'figure'),
    [Input('logreturns_data', 'data'),
     Input('assets_data', 'data'),
     Input('frontier_data', 'data')]
)
def update_frontier_plot(logreturns, assets, fronteira):
    assets = pd.DataFrame(assets)
    logreturns = pd.DataFrame(logreturns).set_index('Date')
    fronteira = pd.DataFrame(fronteira)


    ativos = assets.merge(
        logreturns.agg(['mean', 'std']).T,
        left_on='ticker', right_index=True
    )
    ativos['sharpe'] = ativos['implied'] / ativos['std']
    ativos['error'] = ativos['implied'] - ativos['mean']
    ativos['historical+'] = np.where(ativos['error'] > 0, ativos['error'], 0)
    ativos['historical-'] = np.where(ativos['error'] < 0, -ativos['error'], 0)

    fig = px.line(
        fronteira, x='sigma', y='mu',
        labels={'sigma': 'volatilidade', 'mu': 'retorno'}
    )

    scatter = px.scatter(
        ativos, x='std', y='implied', text='ticker', size='sharpe',
        error_y='historical-', error_y_minus='historical+'
        )
    for s in scatter.data:
        fig.add_trace(s)
    fig.update_traces(textposition='top center')

    return fig


@app.callback(
    Output('weights_plot', 'figure'),
    [Input('frontier_data', 'data'),
     Input('assets_data', 'data'),
     Input('frontier_plot', 'hoverData')]
)
def update_weights_plot(fronteira, assets, plot_click):
    assets = pd.DataFrame(assets)
    tickers = assets['ticker']
    fronteira = pd.DataFrame(fronteira)

    df = pd.merge(
        fronteira[fronteira['minimal_var']][tickers].melt(
            value_name='min var', var_name='ticker'),
        fronteira[fronteira['tangent']][tickers].melt(
            value_name='max sharpe', var_name='ticker')
    )

    if plot_click is not None:
        idx = plot_click['points'][0]['pointNumber']
        mu = fronteira['mu'][idx]
        if np.sum(fronteira['mu'] == mu) >= 0:
            df = pd.merge(
                df, fronteira[fronteira['mu'] == mu][tickers].melt(
                    value_name='seleção', var_name='ticker')
            )

    fig = px.scatter(
        df.melt('ticker'), x='value', y='ticker', color='variable',
        labels={'value': '', 'ticker': ''}
    )

    return fig

#
if __name__ == "__main__":
    app.run_server(debug=True)
