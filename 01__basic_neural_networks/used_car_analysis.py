import numpy as np, pandas as pd
from datetime import datetime   
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class Used_Car_Analysis:

    def __init__(self):
        self.path = r"D:\Datasets\Indian Car Sale Data\Car_Sell_Dataset.csv"
        self.car_type_dict = {
        'Sedan':1, 'Hatchback':2, 'SUV':3, 'MPV':4,'Luxury':5
        }
        self.transmission_dict = {
            'Manual':1, 'Automatic':2
        }
        self.fuel_type_dict = {
            'Petrol':1, 'Diesel':2, 'CNG':3, 'Electric':4, 'Hybrid':5
        }
        self.le_brand = LabelEncoder()
        self.le_model = LabelEncoder()

    def get_dataset(self):    

        df = pd.read_csv(self.path)

        dt = datetime.now()

        df['Car_Type'] = df['Car_Type'].map(self.car_type_dict)
        df['Transmission'] = df['Transmission'].map(self.transmission_dict)
        df['Fuel_Type'] = df['Fuel_Type'].map(self.fuel_type_dict)
        df['Accidental'] = df['Accidental'].map({'No': 0, 'Yes': 1})
        df['Age'] = dt.year - df['Year']

        df = df.drop('Model_Variant', axis=1)
        df = df.drop('State', axis=1)
        df = df.drop('Year', axis=1)

        
        df['Brand_enc'] = self.le_brand.fit_transform(df['Brand'])
        df['Model_enc'] = self.le_model.fit_transform(df['Model_Name'])
        df.drop(['Brand', 'Model_Name'], axis=1, inplace=True)

        
        X = df.drop('Price', axis=1)
        y = df['Price']

        X_train_np, X_test_np, y_train_ser, y_test_ser = train_test_split(X, y, test_size=0.25, random_state=42)

        scaler = MinMaxScaler()
        X_train_np = scaler.fit_transform(X_train_np)
        X_test_np = scaler.transform(X_test_np)

        return X_train_np, X_test_np, y_train_ser, y_test_ser

        