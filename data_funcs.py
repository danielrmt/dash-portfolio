import pandas as pd


def index_composition(index_name='IBRA'):
    url = 'http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraTeorica.aspx?' + \
        f'Indice={index_name}'
    acoes = pd.read_html(url)[0]
    acoes.columns = ['ticker_acao', 'empresa', 'tipo', 'qtde', 'part']
    acoes['part'] = acoes['part'] / 1000
    acoes = acoes[acoes['part'] != 100]
    return acoes
