import json
import requests
import xml.etree.ElementTree as ET

url = "https://api.odcloud.kr/api/RealTradingPriceIndexSvc/v1/getAptRealTradingPriceIndex?page=1&perPage=10&cond%5BSIZE_GBN%3A%3AEQ%5D=0"

requestData = requests.get(url)

print(requestData)