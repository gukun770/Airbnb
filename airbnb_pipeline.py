import function as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import PredefinedSplit, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import re
import os
list_to_drop1 = list(set([
            'last_review','availability_60',
            'last30','last60', 'last90',
            'last120','last180','last_scraped',
            'host_since',
            # 'number_of_reviews',
            'reviews_per_month','notes',
            'access','index','house_rules',
            'description','transit',
            'id'
            ]))
list_to_drop2 = list(set(['beds',
 'host_identity_verified',
 'cancellation_policy_flexible',
 'neighbourhood_cleansed_Outer Richmond',
 'neighbourhood_cleansed_Nob Hill',
 'holiday_No_holiday',
 'neighbourhood_cleansed_South of Market',
 'neighbourhood_cleansed_Western Addition',
 'neighbourhood_cleansed_Inner Richmond',
 'neighbourhood_cleansed_Castro/Upper Market',
 'neighbourhood_cleansed_Excelsior',
 'property_type_Condominium',
 'neighbourhood_cleansed_Downtown/Civic Center',
 'neighbourhood_cleansed_Ocean View',
 'neighbourhood_cleansed_Haight Ashbury',
 'property_type_Loft',
 'host_response_time_a few days or more',
 'holiday_Christmas Day',
 'holiday_Columbus Day',
 'holiday_Veterans Day (Observed)',
 "holiday_New Year's Day",
 'holiday_Labor Day',
 'holiday_Independence Day',
 'neighbourhood_cleansed_Bernal Heights',
 'host_response_time_unclear',
 'holiday_César Chávez Day',
 ]))
preselect_cols = list(set([
        # 'id',
        'price',
        # 'reviews_per_month',
        # 'number_of_reviews',
        # 'last_review',
        # 'host_since',
        # 'minimum_nights',
        'room_type',
        # 'host_response_time',
        # 'host_is_superhost',
        # 'review_scores_rating',
        'property_type',
        # 'neighbourhood_cleansed',
        'bedrooms',
        # 'calculated_host_listings_count',
        # 'host_identity_verified',
        'cleaning_fee',
        # 'last_scraped',
        # 'latitude',
        # 'longitude',
        # 'beds',
        # 'cancellation_policy',
        # 'access',
        # 'description',
        # 'notes','transit',
        # 'instant_bookable',
        # 'extra_people',
        # 'maximum_nights',
        # 'house_rules',
        ]))
class PreselectColumns(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.reset_index()
        X = X.loc[:,preselect_cols]
        return X

class DataType(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        df = X.copy()
        # Clean price label: remove the dollar sign and comma.
        df.price=df.price.str.replace(r'[$,]','').astype(float)

        if 'cleaning_fee' in df.columns:
            df.cleaning_fee=df.cleaning_fee.str.replace(r'[$,]','').astype(float)

        if 'extra_people' in df.columns:
            df.extra_people=df.extra_people.str.replace(r'[$,]','').astype(float)

        # Convert 0 bed and 0 bedrooms to 1.
        if 'bedrooms' in df.columns:
            df.loc[df['bedrooms']==0,'bedrooms'] = 1
            df.bedrooms = df.bedrooms.fillna(1)
        if 'beds' in df.columns:
            df.loc[df['beds']==0,'beds'] = 1
            df.beds = df.beds.fillna(1)

        # Convert minor cases for property type to 'other property types'
        property_types = ['House','Apartment','Condominium','Loft']
        mask = ~df.property_type.isin(property_types)
        df.loc[mask, 'property_type'] = 'others'

        # Convert unimportant neigbourhood to 'Other Neighbourhoods'
        if 'neighbourhood_cleansed' in df.columns:
            neighbourhood_cleansed = ['Castro/Upper Market','Inner Richmond','Downtown/Civic Center','Haight Ashbury',
            'Mission','Outer Richmond','South of Market','Nob Hill','Western Addition','Ocean View','Excelsior']
            mask = ~df.neighbourhood_cleansed.isin(neighbourhood_cleansed)
            df.loc[mask, 'neighbourhood_cleansed'] = 'Other Neighbourhoods'

        # transform t/f column to 1 and 0
        for columns_t_f in ['host_is_superhost','instant_bookable','host_identity_verified']:
            if columns_t_f in df.columns:
                df[columns_t_f] = df[columns_t_f].replace({'f':0,'t':1})

        # deal with nan
        if 'cleaning_fee' in df.columns:
            df.cleaning_fee = df.cleaning_fee.fillna(0)
        if 'reviews_per_month' in df.columns:
            df.reviews_per_month = df.reviews_per_month.fillna(0)
        if 'review_scores_rating' in df.columns:
            df.review_scores_rating = df.review_scores_rating.fillna(0)
        if 'host_response_time' in df.columns:
            df.host_response_time = df.host_response_time.fillna('unclear')
        if 'host_is_superhost' in df.columns:
            df.host_is_superhost = df.host_is_superhost.fillna(0)
        if 'host_identity_verified' in df.columns:
            df.host_identity_verified = df.host_identity_verified.fillna(0)
        return df

class FeatureEnginner(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        df_new = X.copy()
        # Feature engineering - month, year
        if 'last_scraped' in df_new.columns:
            df_new['month_scraped'] =  df_new['last_scraped'].dt.month.astype(str)
        if 'last_scraped' in df_new.columns:
            df_new['year_scraped'] =  df_new['last_scraped'].dt.year.astype(str)

        # Feature engineering - total_hosting_days, price_per_bedroom, price_per_bed
        if 'host_since' in df_new.columns and 'last_scraped' in df_new.columns:
            df_new['total_hosting_days']=df_new['last_scraped'] - df_new['host_since']
            df_new['total_hosting_days']=df_new['total_hosting_days'].apply(lambda row: row.days)
            df_new.total_hosting_days = df_new.total_hosting_days.fillna(0)
        df_new['price_per_bedroom']=df_new['price']/df_new['bedrooms']
        if 'beds' in df_new.columns:
            df_new['price_per_bed']=df_new['price']/df_new['beds']

        # Feature engineering - Add the number of words of house rules, access, transit, description
        # len_house_rules, len_access, len_transit, len_description
        if 'house_rules' in df_new.columns:
            df_new['len_house_rules'] = df_new['house_rules'].astype(str).apply(lambda x: len(x.split(' ')))
        if 'description' in df_new.columns:
            df_new['len_description'] = df_new['description'].astype(str).apply(lambda x: len(x.split(' ')))
        if 'access' in df_new.columns:
            df_new['len_access'] = df_new['access'].astype(str).apply(lambda x: len(x.split(' ')))
        if 'transit' in df_new.columns:
            df_new['len_transit'] = df_new['transit'].astype(str).apply(lambda x: len(x.split(' ')))

        # Feature engineering - Check if the predicted month includes any national holidays, add the name of the holiday if so.
        # holidays = pd.read_csv('holidays.csv', header=None)
        # holidays.columns=['date','holidays']
        # holidays.date = pd.to_datetime(holidays.date)

        # Feature engineering - Compute distance from middle
        center = np.array([37.762835, -122.434239])
        if 'latitiude' in df_new.columns and 'longitude' in df_new.columns:
            df_new['distance'] = df_new[['latitude', 'longitude']].apply(lambda row: np.linalg.norm(row - center), axis=1)

        # df_new['holiday'] = df_new['last_scraped'].apply(
        #     lambda x: F.add_holidays(x, holidays))
        return df_new

class PricePerBedroom(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mean_train_df_with_month = None
        self.mean_train_df_without_month = None

    def fit(self, X, y):
        df = X.copy()
        self.mean_train_df_with_month = df.groupby(['neighbourhood_cleansed','month_scraped'])[['price_per_bedroom']].mean()
        self.mean_train_df_without_month = df.groupby('neighbourhood_cleansed')[['price_per_bedroom']].mean()
        return self

    def transform(self, X):
        df = X.copy()
        df = df.join(self.mean_train_df_with_month, on=['neighbourhood_cleansed', 'month_scraped'],
                    how='left', rsuffix='_per_neig_month')

        df['diff_price_per_bedroom_hood_month'] = df['price_per_bedroom'] - df['price_per_bedroom_per_neig_month']
        df['diff_price_per_bedroom_hood_month'] = df['diff_price_per_bedroom_hood_month'].fillna(df['diff_price_per_bedroom_hood_month'].mean())

        df = df.join(self.mean_train_df_without_month, on='neighbourhood_cleansed',
                    how='left', rsuffix='_per_neighbourhood')
        df['diff_price_per_bedroom_hood'] = df['price_per_bedroom'] - df['price_per_bedroom_per_neighbourhood']
        df['diff_price_per_bedroom_hood'] = df['diff_price_per_bedroom_hood'].fillna(df['diff_price_per_bedroom_hood'].mean())

        df.drop('price_per_bedroom_per_neig_month', axis=1, inplace=True)
        df.drop('price_per_bedroom_per_neighbourhood', axis=1, inplace=True)
        return df

class DropColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        df_new = X.copy()
        for name in list_to_drop1:
            if name in df_new.columns:
                df_new.drop(name, axis=1,inplace=True)
        return df_new

class Getdummies(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X_train_transpose = None
        self.X_train = None
        self.X_test = None
        self.counter = 0

    def fit(self, X, y):
        X_train = X.copy()
        X_train = pd.get_dummies(X_train)
        for name in list_to_drop2:
            if name in X_train.columns:
                X_train.drop(name, axis=1,inplace=True)
        self.X_train_transpose = X_train.head(1).T
        self.X_train_transpose.columns =  [9999999]
        self.X_train = X_train.copy()
        return self

    def transform(self, X):
        df = X.copy()
        df = pd.get_dummies(df)
        df = self.X_train_transpose.join(df.T, how='left').T
        # print('{}'.format(self.counter))
        # self.counter += 1
        if df.isnull().sum().sum() > 0:
            nan_col = df.apply(lambda x:x.isnull().sum())
            print(nan_col[nan_col>0])
        df = df.fillna(0)
        df = df.iloc[1:,:]
        self.X_test = df.copy()
        return df