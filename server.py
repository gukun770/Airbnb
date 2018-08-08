# import the nessecary pieces from Flask
from flask import Flask,render_template, request,jsonify,Response
import pandas as pd
import pickle


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
    # price,room_type,property_type,bedrooms,cleaning_fee = req['price'],req['room_type'],req['property_type'],req['bedrooms'],req['cleaning_fee']

    new_data = {'bedrooms': [int(req['bedrooms'])],
 'cleaning_fee':  [str(req['cleaning_fee'])],
 'price': [str(req['price'])],
 'property_type': [str(req['property_type'])],
 'room_type': [str(req['room_type'])]}
    print(pd.DataFrame(new_data))

    prediction = model.predict(pd.DataFrame(new_data))
    return jsonify({
        'bedrooms': req['bedrooms'],
        'cleaning_fee': req['cleaning_fee'],
        'price': req['price'],
        'property_type': req['property_type'],
        'room_type': req['room_type'],
        'prediction':list(prediction)[0]
})





#When run from command line, start the server
if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 3333, debug = True)