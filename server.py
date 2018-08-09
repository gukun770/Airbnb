# import the nessecary pieces from Flask
from flask import Flask,render_template, request,jsonify,Response
import pandas as pd
import pickle
import json
import requests
import datetime

keys = json.load('keys.json')['key']


#Create the app object that will route our calls
app = Flask(__name__)
# Route the user to the homepage
@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/price', methods = ['GET'])
def price():
    return render_template('price.html')

model = pickle.load(open('rf.pkl','rb'))
@app.route('/inference', methods = ['POST'])
def inference():
    req = request.get_json()
    print(req)

    address = req['address']

    search_query = "https://maps.googleapis.com/maps/api/geocode/json?address="
    search_query += address + '&key=' + keys

    response = requests.get(search_query)
    result = json.loads(response.text)


    latitude = result['results'][0]['geometry']['location']['lat']
    longitude = result['results'][0]['geometry']['location']['lng']
    neighbourhood_cleansed = result['results'][0]['address_components'][2]['short_name']

    new_data = {'bedrooms': [int(req['bedrooms'])],
 'cleaning_fee':  [str(req['cleaning_fee'])],
 'price': [str(req['price'])],
 'property_type': [str(req['property_type'])],
 'room_type': [str(req['room_type'])],
 'latitude':[float(latitude)],
 'longitude':[float(longitude)],
 'neighbourhood_cleansed':[neighbourhood_cleansed],
 'guests_included':[int(req['guests_included'])]
 }

    prediction = model.predict(pd.DataFrame(new_data))
    return jsonify({
        'bedrooms': req['bedrooms'],
        'guests_included':req['guests_included'],
        'cleaning_fee': req['cleaning_fee'],
        'price': req['price'],
        'property_type': req['property_type'],
        'room_type': req['room_type'],
        'latitude': latitude,
        'longitude': longitude,
        'neighbourhood_cleansed': neighbourhood_cleansed,
        'prediction':prediction[0]
})





#When run from command line, start the server
if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 3333, debug = True)
