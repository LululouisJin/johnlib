# Author: John
# Creation date: 2022-12-2
# Description: This script is used to obtain the search result for a query string through requesting specified search engine.
#              It is a information enchancement tool to get further information for the query object.


import re, os, json, requests
from lxml import etree
from urllib.parse import quote_plus

# Google official search API
from googleapiclient.discovery import build
# 100 per monty
my_api_key = os.getenv('GOOGLE_API_KEY')
my_cse_id = os.getenv('GOOGLE_SEARCH_ID')

# Request config
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

# Google search result content extraction regex pattern
RE_PATTERN = '(?<=\(function\(\){var m\=).+?(?=var a\=m)'
URL_RE = '^(https?|http?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'

# Proxy
PROXIES = {
    "http": "http://wusualis_purefda:5c698a-e8b9d3-8ac3c5-2167eb-aa394a@global.rotating.proxyrack.net:9000",
    "https": "http://wusualis_purefda:5c698a-e8b9d3-8ac3c5-2167eb-aa394a@global.rotating.proxyrack.net:9000",
}


class SearchEngineAPI:
    def get_proxy_new(proxy_type="abroad"):
        """
        proxy_type: available value is one of ["abroad","home","ssr"]
        If none is specified, the default value is "abroad"
        """
        proxy = requests.get(f"http://52.131.216.68:5555/random?proxy_type={proxy_type}").text
        return proxy

class GoogleSearchAPI(SearchEngineAPI):
    def search_free(self, query, item_num, language, proxies):
        res = requests.get(
            url="https://www.google.com/search",
            headers=HEADERS,
            params=dict(
                q=query,
                num=item_num + 2,  # Prevents multiple requests
                hl=language,
            ),
            proxies=proxies,
        )
        res.raise_for_status()
        return res

    def search_official_api(search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res['items']

    def anal_html(self, html_text, re_pattern):
        result = re.findall(re_pattern, html_text, re.S)
        result = result[0]
        result = result.strip(';')
        result = result.replace('null', '[]')
        d = eval(result)
        washed = []
        for i in range(len(d)):
            if not d[i]:
                washed.append([])
            else:
                if i % 2 == 0:
                    washed.append(d[i])
                else:
                    washed.append(eval(d[i]))
        return washed

    def extract_info(self, washed):
        info = []
        for rec in washed:
            dict_ = dict()
            if rec and isinstance(rec, list) and len(rec) > 2:
                url = rec[0]
                if isinstance(url, str) and re.findall(URL_RE, url):
                    dict_['url'] = rec[0]
                    dict_['title'] = rec[1]
                    dict_['snippet'] = rec[2]
                    info.append(dict_)
        return info

    def search(self, query, item_num=10, language='en', proxies=None):
        res = self.search_free(query, item_num, language, proxies)
        info = self.anal_html(res.text, re_pattern=RE_PATTERN)
        info = self.extract_info(info)
        return info


if __name__ == '__main__':
    query = '北京普恩光德生物科技开发有限公司'
    engine = GoogleSearchAPI()
    info = engine.search(query, proxies=None, item_num=10)
    pass
