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
ibov = get_quotes(['IBOV']).set_index('Date')
r_ibov = (
    np.log(ibov)
    .diff().dropna()
    .merge(cdi, left_index=True, right_index=True)
    .assign(IBOV = lambda x: x['IBOV'] - x['CDI'])
    .drop(columns=['CDI'])
)



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
        html.H4('Capital Asset Pricing Model'),
        dcc.Graph('capm_plot', style=plot_style)
    ], label='CAPM'),
    #
    dbc.Tab([
        html.H4('Matriz da Raiz da Covariância dos retornos excedentes mensais'),
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
        dbc.RadioItems(id='expected_method', value='implied', inline=True,
            options=[
                {'label': 'Implícito no índice', 'value': 'implied'},
                {'label': 'CAPM', 'value': 'capm'},
                {'label': 'Média histórica', 'value': 'mean'},
                {'label': 'Mediana histórica', 'value': 'median'}
            ]
        ),
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
    for s in['assets', 'prices', 'logreturns', 'covmatrix', 'frontier', 'capm']
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

    m_ibov = r_ibov.resample('MS').sum()
    L = (.06 / 12) / (2. * m_ibov.std()[0]**2)
    assets['implied'] = 2 * L * covmatrix.values @ assets['part'].values

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
        labels={'Date': '', 'value': ''})
    fig.update_yaxes(matches=None, showticklabels=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


@app.callback(
    [Output('capm_plot', 'figure'),
     Output('capm_data', 'data')],
    [Input('logreturns_data', 'data')]
)
def update_capm_plot(logreturns):
    df = pd.merge(
        pd.DataFrame(logreturns)
        .assign(Date=lambda x: pd.to_datetime(x['Date'])),
        r_ibov.resample('MS').sum().reset_index()
    ).set_index('Date').melt('IBOV')
    fig = px.scatter(df, x='IBOV', y='value', trendline="ols",
        facet_col='variable', facet_col_wrap=4, opacity=.5,
        labels={'value': 'Retorno excedente',
                'IBOV': 'Retorno excedente IBOV'})
    fig.update_yaxes(matches=None, showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_traces(
        line=dict(dash="dot"),
        selector=dict(type="scatter", mode="lines")
    )

    results = px.get_trendline_results(fig)
    results['beta'] = results['px_fit_results'].apply(lambda x: x.params[1])
    results['alpha'] = results['px_fit_results'].apply(lambda x: x.params[0])
    results = results.reset_index().rename(columns={'variable': 'ticker'})
    results = results[['ticker', 'beta', 'alpha']]

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig, results.to_dict('records')


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
    fig.update_layout(xaxis_tickformat='.1%')
    fig.update_yaxes(autorange="reversed",
        showgrid=True, gridwidth=1, gridcolor='#b0b0b0')
    return fig


@app.callback(
    Output('covmatrix_plot', 'figure'),
    [Input('covmatrix_data', 'data')]
)
def update_covmatrix_plot(covmatrix):
    df = pd.DataFrame(covmatrix).set_index('index')
    cols = df.columns.values.tolist()
    z = np.sqrt(df).values.tolist()
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
     Input('assets_data', 'data'),
     Input('capm_data', 'data'),
     Input('expected_method', 'value')]
)
def update_frontier_data(covmatrix, logreturns, assets, capm, method):
    logreturns = pd.DataFrame(logreturns).set_index('Date')
    covmatrix = pd.DataFrame(covmatrix).set_index('index')
    capm = pd.DataFrame(capm)
    capm['capm'] = capm['beta'] * .06/12
    assets = pd.DataFrame(assets).merge(capm[['ticker', 'capm']])

    ativos = assets.merge(
        logreturns.agg(['mean', 'median']).T,
        left_on='ticker', right_index=True
    )

    fronteira = (
        MeanVariancePortfolio(ativos[method], covmatrix, ativos['ticker'])
        .frontier(max=ativos[method].max() * 1.5)
        .sort_values(['mu'])
    )
    return fronteira.to_dict('records')


@app.callback(
    Output('frontier_plot', 'figure'),
    [Input('logreturns_data', 'data'),
     Input('assets_data', 'data'),
     Input('capm_data', 'data'),
     Input('frontier_data', 'data'),
     Input('covmatrix_data', 'data'),
     Input('expected_method', 'value')]
)
def update_frontier_plot(logreturns, assets, capm, fronteira, covmatrix, method):
    logreturns = pd.DataFrame(logreturns).set_index('Date')
    covmatrix = pd.DataFrame(covmatrix).set_index('index')
    capm = pd.DataFrame(capm)
    capm['capm'] = capm['beta'] * .06/12
    assets = pd.DataFrame(assets).merge(capm[['ticker', 'capm']])
    fronteira = pd.DataFrame(fronteira)

    ativos = assets.merge(
        logreturns.agg(['mean', 'median']).T,
        left_on='ticker', right_index=True
    )
    ativos['std'] = np.sqrt(np.diag(covmatrix))
    ativos['sharpe'] = ativos[method] / ativos['std']
    
    fig = px.line(
        fronteira, x='sigma', y='mu',
        labels={'sigma': 'volatilidade', 'mu': 'retorno'}
    )
    fig.update_layout(yaxis_tickformat='.1%', xaxis_tickformat='.1%')

    tan_port = fronteira[fronteira['tangent']]
    fig.add_shape(
        type='line', xref="x", yref="y",
        x0=0, y0=0,
        x1=tan_port['sigma'].values[0], y1=tan_port['mu'].values[0],
        line={'color': 'black', 'width': 1}
    )

    ativos['sharpe_rescaled'] = ativos['sharpe'] - ativos['sharpe'].min()+.01

    scatter = px.scatter(
        ativos, x='std', y=method, text='ticker', size='sharpe_rescaled'
    )
    for s in scatter.data:
        fig.add_trace(s)
    fig.update_traces(textposition='top center')

    return fig


@app.callback(
    Output('weights_plot', 'figure'),
    [Input('frontier_data', 'data'),
     Input('assets_data', 'data')]
)
def update_weights_plot(fronteira, assets):
    assets = pd.DataFrame(assets)
    tickers = assets['ticker']
    fronteira = pd.DataFrame(fronteira)

    df = fronteira.melt(['mu', 'sigma'], tickers,
        value_name='peso', var_name='ticker')
    df['text'] = round(df['peso'] * 100, 1)
    df['E(r)'] = round(df['mu'] * 100, 2).astype(str)+ \
        ', dp(r)=' + round(df['sigma'] * 100, 2).astype(str)
    fig = px.scatter(
        df, x='peso', y='ticker', text='text', #, color='variable'
        animation_frame='E(r)',
        labels={'peso': '', 'ticker': ''}
    )
    fig["layout"].pop("updatemenus")
    fig.update_layout(xaxis_tickformat='%')
    fig.update_traces(marker={'size': 12, 'opacity': .9}, textposition='top center')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#b0b0b0')

    return fig

#
if __name__ == "__main__":
    app.run_server(debug=True)
