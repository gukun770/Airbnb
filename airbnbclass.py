import numpy as np
import pandas as pd
import function as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import os



columns_to_drop1 = [
                'last_review','availability_60','last30','last60', 'last90',
                'last120','last180','last_scraped','host_since','number_of_reviews',
                'reviews_per_month','notes','access','index','index','level_0',
                'house_rules','description','transit','id','property_type_others']

columns_to_drop2 = ['beds',
 'host_identity_verified',
 'cancellation_policy_flexible',
 'neighbourhood_cleansed_Outer Richmond',
 'bedrooms',
 'neighbourhood_cleansed_Nob Hill',
 'holiday_No_holiday',
 'neighbourhood_cleansed_South of Market',
 'neighbourhood_cleansed_Western Addition',
 'property_type_others',
 'neighbourhood_cleansed_Inner Richmond',
 'neighbourhood_cleansed_Castro/Upper Market',
 'neighbourhood_cleansed_Excelsior',
 'property_type_Condominium',
 'room_type_Shared room',
 'neighbourhood_cleansed_Downtown/Civic Center',
 'neighbourhood_cleansed_Ocean View',
 'neighbourhood_cleansed_Haight Ashbury',
 'property_type_Loft',
 'host_response_time_a few days or more',
 'cancellation_policy_super_strict_60',
 'holiday_Christmas Day',
 'month_scraped_10',
 'month_scraped_9',
 'month_scraped_8',
 'month_scraped_7',
 'month_scraped_6',
 'holiday_Columbus Day',
 'month_scraped_11',
 'month_scraped_12',
 'holiday_Veterans Day (Observed)',
 'cancellation_policy_super_strict_30',
 "holiday_New Year's Day",
 'holiday_Labor Day',
 'holiday_Independence Day',
 'year_scraped_2017',
 'neighbourhood_cleansed_Bernal Heights',
 'host_response_time_unclear',
 'month_scraped_3',
 'holiday_César Chávez Day',
 'month_scraped_4']

# columns_to_drop = list(set(columns_to_drop1 + columns_to_drop2))

class Airbnb:

    def __init__(self):
        self.data = None
        self.filtered_data = None
        self.cleaned_data = None
        self.fe_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.price_neighbour_class = None
        self.model = None
        self.X_train_last = None
        self.X_test_last = None
        self.X_train_transpose = None
        self.metrics = {}
        self.feature_importance = None
        self.notes = None
        self.description = None
        self.house_rules = None
        self.access = None
        self.transit = None


    def load_data(self, columns_used):
        # read and load data
        columns = columns_used

        listings = pd.read_csv('data/listings20180509.csv',usecols=columns,parse_dates=['last_scraped'])
        reviews = pd.read_csv('data/reviews/reviews20180509.csv',parse_dates=['date'])
        listings = F.add_reviews_cnt(listings, reviews) # add number of reviews within last 30,60,90,120 and 180 days.

        path = os.getcwd()
        folder_path = path + '/data/'
        for file_name_listing in os.listdir(folder_path):
            if file_name_listing.endswith('.csv') and file_name_listing != 'listings20180509.csv':
                path=folder_path+file_name_listing
                date_part = file_name_listing[-12:-4]
                print('Reading: {}...'.format(path))
                temp = pd.read_csv(path,usecols=columns,parse_dates=['last_scraped'])
                for file_name_review in os.listdir(folder_path + 'reviews/'):
                    if file_name_review == 'reviews{}.csv'.format(date_part):
                        path_review = folder_path +  'reviews/' + file_name_review
                        reviews = pd.read_csv(path_review,parse_dates=['date'])
                        temp = F.add_reviews_cnt(temp, reviews)
                        print('reviews{}.csv'.format(date_part), temp.shape)
                listings = pd.concat([listings, temp])
                print(listings.shape)


        # Convert date to datetime object
        listings['last_review'] = pd.to_datetime(listings['last_review'])
        listings['last_scraped'] = pd.to_datetime(listings['last_scraped'])
        listings['host_since'] = pd.to_datetime(listings['host_since'])

        # Fill nan number of reviews with 0 
        listings[['last30','last60','last90','last120','last180']] = listings[
            ['last30','last60','last90','last120','last180']].fillna(0, axis=1)

        self.data = listings.copy()

        return listings

    def clean_data(self, df):
        # Clean price label: remove the dollar sign and comma.
        df.price=df.price.str.replace(r'[$,]','').astype(float)
        df.cleaning_fee=df.cleaning_fee.str.replace(r'[$,]','').astype(float)
        df.extra_people=df.extra_people.str.replace(r'[$,]','').astype(float)

        # Convert 0 bed and 0 bedrooms to 1.
        df.loc[df['bedrooms']==0,'bedrooms'] = 1
        df.loc[df['beds']==0,'beds'] = 1

        # Convert minor cases for property type to 'other property types'
        property_types = ['House','Apartment','Condominium','Loft']
        mask = ~df.property_type.isin(property_types)
        df.loc[mask, 'property_type'] = 'others'

        # Convert unimportant neigbourhood to 'Other Neighbourhoods'
        neighbourhood_cleansed = ['Castro/Upper Market','Inner Richmond','Downtown/Civic Center','Haight Ashbury',
        'Mission','Outer Richmond','South of Market','Nob Hill','Western Addition','Ocean View','Excelsior']
        mask = ~df.neighbourhood_cleansed.isin(neighbourhood_cleansed)
        df.loc[mask, 'neighbourhood_cleansed'] = 'Other Neighbourhoods'

        # transform t/f column to 1 and 0
        columns_t_f = ['host_is_superhost','instant_bookable','host_identity_verified']
        df[columns_t_f] = df[columns_t_f].replace({'f':0,'t':1})

        # deal with nan
        df.bedrooms = df.bedrooms.fillna(1)
        df.cleaning_fee = df.cleaning_fee.fillna(0)
        df.reviews_per_month = df.reviews_per_month.fillna(0)
        df.review_scores_rating = df.review_scores_rating.fillna(0)
        df.beds = df.beds.fillna(1)
        df.host_response_time = df.host_response_time.fillna('unclear')
        df.host_is_superhost = df.host_is_superhost.fillna(0)
        df.host_identity_verified = df.host_identity_verified.fillna(0)


        self.cleaned_data = df.copy()
        return df

    def filter_data(self, df):
        df_new = df.copy()
        df_new = df_new[df_new.minimum_nights<=7]
        nb_lost1 = df.shape[0] - df_new.shape[0]
        print('Filter 1: {}, i.e. {:.2f}% listings removed.\n'.format(nb_lost1, nb_lost1/df.shape[0]*100))

        df_new = df_new[df_new.reviews_per_month>0]
        nb_lost2 = df.shape[0] - df_new.shape[0] - nb_lost1
        print('FIlter 2 : {}, i.e.{:.2f}% listings removed.\n'.format(nb_lost2, nb_lost2/df.shape[0]*100 ))

        pair_available_nb_reviews = []
        # for i in range(31):
        #     pair_available_nb_reviews.append((i, (30-i)/5*0.7,'last30'))
        pair_available_nb_reviews = [(0,9, 'last60'),(1,9,'last60'),(2,8,'last60'), (3,8,'last60'), (4,7, 'last60'),
                            (5,7, 'last60'),(6,6,'last60'),(7,6,'last60'), (8,5,'last60'), (9,4, 'last60'),
                            (10,3, 'last60'),(11,2, 'last60'),(12,2, 'last90'),(13,1, 'last90'),]
        for available,nb_reviews,my_condition in pair_available_nb_reviews:
            df_new = F.select_active_host(df_new, available, nb_reviews,my_condition )

        nb_lost3 = df.shape[0] - df_new.shape[0] - nb_lost1 - nb_lost2
        print('FIlter 3 : {}, i.e.{:.2f}% listings removed.\n'.format(nb_lost3, nb_lost3/df.shape[0]*100 ))

        
        return df_new

    def feature_engineer(self,df):
        df_new = df.copy()
        # Feature engineering - month, year
        df_new['month_scraped'] =  df_new['last_scraped'].dt.month.astype(str)
        df_new['year_scraped'] =  df_new['last_scraped'].dt.year.astype(str)

        # Feature engineering - total_hosting_days, price_per_bedroom, price_per_bed
        df_new['total_hosting_days']=df_new['last_scraped'] - df_new['host_since']
        df_new['total_hosting_days']=df_new['total_hosting_days'].apply(lambda row: row.days)
        df_new.total_hosting_days = df_new.total_hosting_days.fillna(0)
        df_new['price_per_bedroom']=df_new['price']/df_new['bedrooms']
        df_new['price_per_bed']=df_new['price']/df_new['beds']

        # Cleaning fee per person
        df_new['cleaning_fee_person'] = df_new['cleaning_fee'] / df_new['guests_included']

        # Feature engineering - Add the number of words of house rules, access, transit, description
        # len_house_rules, len_access, len_transit, len_description
        df_new['len_house_rules'] = df_new['house_rules'].astype(str).apply(lambda x: len(x.split(' ')))
        df_new['len_description'] = df_new['description'].astype(str).apply(lambda x: len(x.split(' ')))
        df_new['len_access'] = df_new['access'].astype(str).apply(lambda x: len(x.split(' ')))
        df_new['len_transit'] = df_new['transit'].astype(str).apply(lambda x: len(x.split(' ')))

        # Feature engineering - Check if the predicted month includes any national holidays, add the name of the holiday if so.
        holidays = pd.read_csv('holidays.csv', header=None)
        holidays.columns=['date','holidays']
        holidays.date = pd.to_datetime(holidays.date)

        # Feature engineering - Compute distance from middle
        middle = np.array([df_new.latitude.mean(), df_new.longitude.mean()])
        df_new['distance'] = df_new[['latitude', 'longitude']].apply(lambda row: np.linalg.norm(row - middle), axis=1)

        df_new['holiday'] = df_new['last_scraped'].apply(
            lambda x: F.add_holidays(x, holidays))
        self.fe_data = df_new.copy()
        return df_new


    def my_train_test_split(self, df, date):
        mask_test = df.last_scraped >= pd.to_datetime(date)
        X_train = df[~mask_test]
        X_test = df[mask_test]

        y_train = X_train.pop('availability_30')
        y_test = X_test.pop('availability_30')

        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()

        return X_train, X_test, y_train, y_test

    def train_text_topic_weight(self, X_train, text_col, n_components):
        text_obj = F.text()
        X_train_text = X_train[text_col]
        df_weight = text_obj.train_text_features(X_train_text, n_components)
        X_train = pd.concat([X_train, df_weight], axis=1)
        return X_train, text_obj

    def predict_text_topic_weight(self, X_test, text_col, obj):
        X_test_text = X_test[text_col]
        df_weight = obj.predict_text_features(X_test_text)
        X_test = pd.concat([X_test, df_weight], axis=1)
        return X_test

    def train_model(self,model,X_train,y_train):
        X_train = X_train.reset_index()
        price_neighbour = F.PricePerBedroom()
        price_neighbour.fit(X_train)
        self.price_neighbour_class = price_neighbour
        X_train = price_neighbour.transform(X_train)

        # X_train, self.notes = self.train_text_topic_weight(X_train, 'notes', 5)
        # X_train, self.description = self.train_text_topic_weight(X_train, 'description', 5)
        # X_train, self.transit = self.train_text_topic_weight(X_train, 'transit', 5)
        # X_train, self.house_rules = self.train_text_topic_weight(X_train, 'house_rules', 5)
        # X_train, self.access = self.train_text_topic_weight(X_train, 'access', 5)
        X_train = F.drop_columns(X_train,columns_to_drop1)
        X_train = pd.get_dummies(X_train)
        X_train = F.drop_columns(X_train,columns_to_drop2)

        self.X_train_last = X_train.copy()
        self.X_train_transpose = X_train.head(1).T
        self.X_train_transpose.columns =  [9999999]

        m = model.fit(X_train,y_train)
        self.model = m

    def evaluate_model(self, X_test, y_test, f_importance=True):
        X_test = X_test.reset_index()

        X_test = self.price_neighbour_class.transform(X_test)

        # X_test = self.predict_text_topic_weight(X_test, 'notes', self.notes)
        # X_test = self.predict_text_topic_weight(X_test, 'description', self.description)
        # X_test = self.predict_text_topic_weight(X_test, 'transit', self.transit)
        # X_test = self.predict_text_topic_weight(X_test, 'house_rules', self.house_rules)
        # X_test = self.predict_text_topic_weight(X_test, 'access', self.access)


        X_test = F.drop_columns(X_test,columns_to_drop1)
        X_test = pd.get_dummies(X_test)
        X_test = F.drop_columns(X_test,columns_to_drop2)

        X_test = self.X_train_transpose.join(X_test.T, how='left').T
        if X_test.isnull().sum().sum() > 0:
            nan_col = X_test.apply(lambda x:x.isnull().sum())
            print(nan_col[nan_col>0])

        X_test = X_test.fillna(0)
        X_test = X_test.iloc[1:,:]
        self.X_test_last = X_test.copy()

        y_pred = self.model.predict(X_test)
        print(mean_absolute_error(y_test, y_pred))
        print(mean_squared_log_error(y_test, y_pred))
        self.metrics['mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
        self.metrics['mean_squared_log_error'] = mean_squared_log_error(y_test, y_pred)

        if f_importance:
            neg_scorer = lambda y1, y2: mean_squared_log_error(y1, y2)*(-1)
            feature_importance = F.permutation_importance(self.model, X_test.values, y_test.values, scorer=neg_scorer)
            series_feature_importance = pd.Series(feature_importance,index=self.X_train_last.columns)
            self.feature_importance = series_feature_importance.copy()



    def model_predict(self):
        pass







