# import the nessecary pieces from Flask
from flask import Flask,render_template, request,jsonify,Response
import pandas as pd
import numpy as np
import pickle
import json
import requests
import predict as P

with open('keys.json') as f:
    my_keys = json.load(f)
keys = my_keys['key']


#Create the app object that will route our calls
app = Flask(__name__)
# Route the user to the homepage
@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/about_model', methods = ['GET'])
def about_model():
    return render_template('about_model.html')

@app.route('/about_market_part1', methods = ['GET'])
def about_market_part1():
    return render_template('about_market_part1.html')

@app.route('/about_market_part2', methods = ['GET'])
def about_market_part2():
    return render_template('about_market_part2.html')

model_rf = pickle.load(open('rf.pkl','rb'))
model_ridge = pickle.load(open('ridge.pkl','rb'))
@app.route('/inference', methods = ['POST'])
def inference():
    req = request.get_json()

    address = req['address']

    search_query = "https://maps.googleapis.com/maps/api/geocode/json?address="
    search_query += address + '&key=' + keys

    response = requests.get(search_query)
    result = json.loads(response.text)


    latitude = result['results'][0]['geometry']['location']['lat']
    longitude = result['results'][0]['geometry']['location']['lng']
    neighbourhood_cleansed = result['results'][0]['address_components'][2]['short_name']
    zipcode = result['results'][0]['address_components'][7]['long_name']

    nearest = pd.read_csv('data/nearest/nearest.csv')
    price = P.get_price(nearest,
    pd.DataFrame({'latitude': [latitude],'longitude':[longitude]}),
    int(req['bedrooms']),
    str(req['property_type']),
    str(req['room_type']),
    int(zipcode))

    craigslist = P.price_craigslist(req['bedrooms'], zipcode)

    new_data = {'bedrooms': [int(req['bedrooms'])],
 'cleaning_fee':  [float(req['cleaning_fee'])],
 'price': [float(price)],
 'property_type': [str(req['property_type'])],
 'room_type': [str(req['room_type'])],
 'latitude':[float(latitude)],
 'longitude':[float(longitude)],
 'guests_included':[int(req['guests_included'])],
 'zipcode':[str(zipcode)],
 }

    prediction = np.clip(model_rf.predict(pd.DataFrame(new_data))*0.2 + 
        model_ridge.predict(pd.DataFrame(new_data) )*0.8, 0, 30)
    occupancy = np.round(30 - prediction,1)
    monthly_revenue = np.round(occupancy * float(price),1)
    annual_revenue = np.round(occupancy * float(price) * 12,1)

    return jsonify({
        'bedrooms': req['bedrooms'],
        'guests_included':req['guests_included'],
        'cleaning_fee': req['cleaning_fee'],
        'price': price[0],
        'property_type': req['property_type'],
        'room_type': req['room_type'],
        'latitude': latitude,
        'longitude': longitude,
        'neighbourhood_cleansed': neighbourhood_cleansed,
        'prediction':prediction[0],
        'zipcode':zipcode,
        'occupancy': occupancy[0],
        'monthly_revenue': monthly_revenue[0],
        'annual_revenue': annual_revenue[0],
        'price_craigslist': craigslist,
})





#When run from command line, start the server
if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 3333, debug = True)
