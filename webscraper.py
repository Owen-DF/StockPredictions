#goal is to scrape from the web all historical stock prices for a given stock


import requests 
from bs4 import BeautifulSoup 
import pandas as pd 



headers = {r'user-agent': 'Mozilla/5.0 (Windows NT 10.0; \\ Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \\ Chrome/84.0.4147.105 Safari/537.36'}


url = 'https://www.investing.com/equities/nvidia-corp/historical-data'




