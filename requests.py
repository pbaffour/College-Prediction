import requests



url = 'http://localhost:5000/results'

r = requests.post(url,json={'response1':1, 'response2':1, 'response3':1})



print(r.json())