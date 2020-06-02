
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
from dash_table.Format import Format, Scheme, Sign

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from data_funcs import assets_df


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
selected_assets = assets[assets['part'].isin(is_main_ticker)].index[:10]

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
        dash_table.DataTable(
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


# LAYOUT
app.layout = html.Div([
    navbar,
    dbc.Button('Selecionar ativos', id='assets_open', className='ml-auto'),
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

if __name__ == "__main__":
    app.run_server(debug=True)
