import os
import io
import numpy as np
import pandas as pd
import requests
import urllib.request as ur
from bs4 import BeautifulSoup
from zipfile import ZipFile
from pandas_datareader import data as wb


def index_composition(index_name='IBRA'):
    url = 'http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraTeorica.aspx?' + \
        f'Indice={index_name}'
    acoes = pd.read_html(url)[0]
    acoes.columns = ['ticker', 'empresa', 'tipo', 'qtde', 'part']
    acoes['part'] = acoes['part'] / 1000
    acoes = acoes[acoes['part'] != 100]
    acoes = acoes.sort_values('part', ascending=False)
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


def cache_data(fn, fun, *args, **kwargs):
    if not os.path.isdir('data'):
        os.mkdir('data')
    fn = os.path.join('data', fn)
    if os.path.exists(fn):
        print(f'{fn} exists, using cached version')
        return pd.read_csv(fn)
    else:
        print(f'{fn} does not exist, creating file')
        df = fun(*args, **kwargs)
        df.to_csv(fn, index=False)
        return df


def assets_df(index_name='IBRA'):
    mktcap = get_mktcap()
    sectors = cache_data('setores.csv', assets_sectors)
    assets = cache_data('ibra.csv', index_composition, index_name)
    assets['base_ticker'] = assets['ticker'].str[:4]
    assets = assets.merge(mktcap, on="empresa")
    assets = assets.merge(sectors, on='base_ticker')
    return assets


def get_quote(ticker):
    if ticker in ['BVSP', 'IBOV']:
        ticker2 = '^BVSP'
    else:
        ticker2 = f'{ticker}.SA'
    df = (
        wb.DataReader(f'{ticker2}', start='2010-1-1', data_source='yahoo')
        .rename(columns={'Adj Close': ticker})
        .reset_index()
    )
    return df[['Date', ticker]]


def get_quotes(tickers):
    return pd.concat([
        cache_data(f'{t}.csv', get_quote, t)
        .assign(Date=lambda x: pd.to_datetime(x['Date']))
        .set_index('Date')
        for t in tickers
    ], axis=1).reset_index()


def get_mktcap():
    url = "http://www.b3.com.br/pt_br/market-data-e-indices/" + \
        "servicos-de-dados/market-data/consultas/mercado-a-vista/" + \
            "valor-de-mercado-das-empresas-listadas/bolsa-de-valores/"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    url = soup.find("a", string="Histórico diário").get('href')
    url = "http://www.b3.com.br/" + url.replace('../', '')
    df = (
        pd.read_excel(url, skiprows=7, skipfooter=5)
        .dropna(axis=1, how="all")
        .rename(columns={"Empresa": "empresa", "R$ (Mil)": "mktcap"})
        .assign(
            mktcap=lambda x: x['mktcap'] / 1000,
            empresa=lambda x: x['empresa'].str.strip()
        )
        [["empresa", "mktcap"]]
    )
    return df


def bcb_sgs(beg_date, end_date, **kwargs):
    '''
    beg_date, end_date: string
    **kwargs: str=int
    '''
    return pd.concat([
        pd.read_json(f"http://api.bcb.gov.br/dados/serie/bcdata.sgs.{v}" +
                     f"/dados?formato=json&dataInicial={beg_date}&" +
                     f"dataFinal={end_date}",
                     convert_dates=False)
        .assign(data=lambda x: pd.to_datetime(x.data, dayfirst=True))
        .rename(columns={'valor': k, 'data': 'Date'})
        .set_index('Date')
        for k, v in kwargs.items()
    ], axis=1)
