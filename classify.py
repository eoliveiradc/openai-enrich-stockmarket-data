import pandas as pd
import openai

# initialize variables
openai.api_key = "key"
#openai.organization = "org"
n100_info = {}

# get symbols and names
for _, row in pd.read_csv("nasdaq100.csv").iterrows():
    n100_info[row['symbol']] = {'name': row['name']}

# get market cap
for _, row in pd.read_csv("nasdaq100_quote.csv").iterrows():
    n100_info[row['symbol']] = n100_info[row['symbol']] | {'marketCap': row['marketCap']}

# get YTD price change
for _, row in pd.read_csv("nasdaq100_price_change.csv").iterrows():
    n100_info[row['symbol']] = n100_info[row['symbol']] | {'ytd': row['ytd']}

# enrich with OpenAI information
prompt = '''Classify company {company} in one of the following sectors. Answer only with the sector name:
    Technology, Consumer Cyclical, Industrials, Utilities, Healthcare, Communication Services, Energy, Consumer Defensive, Real Estate, Financial Services.
'''

for key in n100_info:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt.format(company=n100_info[key]['name'])}],
        temperature=0,
    )
    sector = response['choices'][0]['message']['content']
    n100_info[key] = n100_info[key] | {'sector': sector}

# finally print final output
for key in n100_info:
    print(n100_info[key])