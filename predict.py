import numpy as np
from sklearn.neighbors import KNeighborsRegressor 
import requests
from bs4 import BeautifulSoup


def get_price(df, lat_lon, bedrooms, property_type, room_type, zipcode):
    condition_bedrooms = df.bedrooms == bedrooms
    condition_type = df.room_type == room_type
    condition_zipcode = df.zipcode == zipcode
    condition_property = df.property_type == property_type
    selected = df[condition_bedrooms & condition_type & condition_zipcode & condition_property]


    if selected.shape[0] >= 5:
        print('0')
        knn = KNeighborsRegressor(n_neighbors=5)
        knn_model = knn.fit(selected[['latitude','longitude']], selected.price)
        price_pred = knn_model.predict(lat_lon)
    else: 
        selected = df[condition_bedrooms & condition_type & condition_zipcode]
        if selected.shape[0] >= 5:
            print('1')
            knn = KNeighborsRegressor(n_neighbors=5)
            knn_model = knn.fit(selected[['latitude','longitude']], selected.price)
            price_pred = knn_model.predict(lat_lon)
        else:
            selected = df[condition_bedrooms & condition_type]
            if selected.shape[0] >= 5:
                print('2')
                knn = KNeighborsRegressor(n_neighbors=5)
                knn_model = knn.fit(selected[['latitude','longitude']], selected.price)
                price_pred = knn_model.predict(lat_lon)
            else:
                print('3')
                selected = df[condition_bedrooms & condition_type]
                price_pred = np.mean(selected.price)

    return price_pred


def price_craigslist(bedrooms, zipcode):
    '''
    INPUT: string
    OUTPUT: int
    '''

    url = "https://sfbay.craigslist.org/search/apa?query={0}&min_bedrooms={1}&max_bedrooms={1}&availabilityMode=0&sale_date=all+dates".format(zipcode, bedrooms)

    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    search_count = soup.findAll('span', class_='result-price')
    price_list  = list(set([ float(x.text.strip('$')) for x in search_count[:20]]))
    price = np.median(price_list)
    print(price_list)
    
    return np.round(price,1)