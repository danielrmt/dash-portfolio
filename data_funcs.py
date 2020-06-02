import io
import numpy as np
import pandas as pd
import requests
import urllib.request as ur
from bs4 import BeautifulSoup
from zipfile import ZipFile


def index_composition(index_name='IBRA'):
    url = 'http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraTeorica.aspx?' + \
        f'Indice={index_name}'
    acoes = pd.read_html(url)[0]
    acoes.columns = ['ticker', 'empresa', 'tipo', 'qtde', 'part']
    acoes['part'] = acoes['part'] / 1000
    acoes = acoes[acoes['part'] != 100]
    return acoes


def assets_sectors():
    url = 'http://bvmf.bmfbovespa.com.br/cias-listadas/empresas-listadas/' + \
        'BuscaEmpresaListada.aspx?opcao=1&indiceAba=1&Idioma=pt-br'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    url = soup.find("a", string="Download").get('href')

    # Unzip
    filehandle, _ = ur.urlretrieve(url)
    with ZipFile(filehandle, 'r') as zf:
        fn = zf.namelist()[0]
        df = pd.read_excel(
            io.BytesIO(zf.read(fn)),
            skiprows=7, skipfooter=18,
            names=['setor', 'subsetor', 'empresa', 'base_ticker', 'governanca']
        )
    df['segmento'] = np.where(
        df['base_ticker'].isnull(), df['empresa'], np.NaN)
    for col in ['setor', 'subsetor', 'segmento']:
        df[col] = df[col].fillna(method='ffill')
    df['governanca'] = df['governanca'].fillna('')
    df = df.dropna(subset=['base_ticker'])
    df = df[df['base_ticker'] != 'CÓDIGO']
    df = df[df['subsetor'] != 'SUBSETOR']
    df = df.reset_index(drop=True)
    return df[['setor', 'subsetor', 'segmento', 'governanca', 'base_ticker']]


def assets_df(index_name='IBRA'):
    sectors = assets_sectors()
    assets = index_composition(index_name)
    assets['base_ticker'] = assets['ticker'].str[:4]
    return assets.merge(sectors, on='base_ticker')
