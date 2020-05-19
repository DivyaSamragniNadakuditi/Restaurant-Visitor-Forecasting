import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor

from keras.layers import Embedding, Input, Dense
from keras.models import Model
import keras
import keras.backend as K

import matplotlib.pyplot as plt
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

def plot_actual_predicted(actual, predicted):
    print('RMSE: ', RMSLE(actual, predicted))
    tmp = pd.DataFrame({'actual': actual, 'predicted': predicted}).sort_values(['actual'])
    plt.scatter(range(tmp.shape[0]), tmp['predicted'], color='green')
    plt.scatter(range(tmp.shape[0]), tmp['actual'], color='blue')
    plt.show()
    del tmp
data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

for df in ['ar','hr']:
    data[df]['reserve_visitors'] = np.log1p(data[df]['reserve_visitors'])
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_dow'] = data[df]['visit_datetime'].dt.dayofweek
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

value_col = ['holiday_flg','min_visitors','mean_visitors','median_visitors','max_visitors','count_observations',
'rs1_x','rv1_x','rs2_x','rv2_x','rs1_y','rv1_y','rs2_y','rv2_y','total_reserv_sum','total_reserv_mean',
'total_reserv_dt_diff_mean','date_int','var_max_lat','var_max_long','lon_plus_lat']

nn_col = value_col + ['dow', 'year', 'month', 'air_store_id2', 'air_area_name', 'air_genre_name',
'air_area_name0', 'air_area_name1', 'air_area_name2', 'air_area_name3', 'air_area_name4',
'air_area_name5', 'air_area_name6', 'air_genre_name0', 'air_genre_name1',
'air_genre_name2', 'air_genre_name3', 'air_genre_name4']

X = train.copy()
X_test = test[nn_col].copy()

value_scaler = preprocessing.MinMaxScaler()
for vcol in value_col:
    X[vcol] = value_scaler.fit_transform(X[vcol].values.astype(np.float64).reshape(-1, 1))
    X_test[vcol] = value_scaler.transform(X_test[vcol].values.astype(np.float64).reshape(-1, 1))

X_train = list(X[nn_col].T.as_matrix())
Y_train = X['visitors'].values
nn_train = [X_train, Y_train]
nn_test = [list(X_test[nn_col].T.as_matrix())]
print("Train and test data prepared for NN")

def get_nn_complete_model(train, hidden1_neurons=35, hidden2_neurons=15):
K.clear_session()

    air_store_id = Input(shape=(1,), dtype='int32', name='air_store_id')
    air_store_id_emb = Embedding(len(train['air_store_id2'].unique()) + 1, 15, input_shape=(1,),
                                 name='air_store_id_emb')(air_store_id)
    air_store_id_emb = keras.layers.Flatten(name='air_store_id_emb_flatten')(air_store_id_emb)

    dow = Input(shape=(1,), dtype='int32', name='dow')
    dow_emb = Embedding(8, 3, input_shape=(1,), name='dow_emb')(dow)
    dow_emb = keras.layers.Flatten(name='dow_emb_flatten')(dow_emb)

    month = Input(shape=(1,), dtype='int32', name='month')
    month_emb = Embedding(13, 3, input_shape=(1,), name='month_emb')(month)
    month_emb = keras.layers.Flatten(name='month_emb_flatten')(month_emb)

    air_area_name, air_genre_name = [], []
    air_area_name_emb, air_genre_name_emb = [], []
    for i in range(7):
        area_name_col = 'air_area_name' + str(i)
        air_area_name.append(Input(shape=(1,), dtype='int32', name=area_name_col))
        tmp = Embedding(len(train[area_name_col].unique()), 3, input_shape=(1,),
                        name=area_name_col + '_emb')(air_area_name[-1])
        tmp = keras.layers.Flatten(name=area_name_col + '_emb_flatten')(tmp)
        air_area_name_emb.append(tmp)

        if i > 4:
            continue
        area_genre_col = 'air_genre_name' + str(i)
        air_genre_name.append(Input(shape=(1,), dtype='int32', name=area_genre_col))
        tmp = Embedding(len(train[area_genre_col].unique()), 3, input_shape=(1,),
                        name=area_genre_col + '_emb')(air_genre_name[-1])
        tmp = keras.layers.Flatten(name=area_genre_col + '_emb_flatten')(tmp)
        air_genre_name_emb.append(tmp)

    air_genre_name_emb = keras.layers.concatenate(air_genre_name_emb)
    air_genre_name_emb = Dense(4, activation='sigmoid', name='final_air_genre_emb')(air_genre_name_emb)

    air_area_name_emb = keras.layers.concatenate(air_area_name_emb)
    air_area_name_emb = Dense(4, activation='sigmoid', name='final_air_area_emb')(air_area_name_emb)
    
    air_area_code = Input(shape=(1,), dtype='int32', name='air_area_code')
    air_area_code_emb = Embedding(len(train['air_area_name'].unique()), 8, input_shape=(1,), name='air_area_code_emb')(air_area_code)
    air_area_code_emb = keras.layers.Flatten(name='air_area_code_emb_flatten')(air_area_code_emb)
    
    air_genre_code = Input(shape=(1,), dtype='int32', name='air_genre_code')
    air_genre_code_emb = Embedding(len(train['air_genre_name'].unique()), 5, input_shape=(1,),
                                   name='air_genre_code_emb')(air_genre_code)
    air_genre_code_emb = keras.layers.Flatten(name='air_genre_code_emb_flatten')(air_genre_code_emb)

    
    holiday_flg = Input(shape=(1,), dtype='float32', name='holiday_flg')
    year = Input(shape=(1,), dtype='float32', name='year')
    min_visitors = Input(shape=(1,), dtype='float32', name='min_visitors')
    mean_visitors = Input(shape=(1,), dtype='float32', name='mean_visitors')
    median_visitors = Input(shape=(1,), dtype='float32', name='median_visitors')
    max_visitors = Input(shape=(1,), dtype='float32', name='max_visitors')
    count_observations = Input(shape=(1,), dtype='float32', name='count_observations')
    rs1_x = Input(shape=(1,), dtype='float32', name='rs1_x')
    rv1_x = Input(shape=(1,), dtype='float32', name='rv1_x')
    rs2_x = Input(shape=(1,), dtype='float32', name='rs2_x')
    rv2_x = Input(shape=(1,), dtype='float32', name='rv2_x')
    rs1_y = Input(shape=(1,), dtype='float32', name='rs1_y')
    rv1_y = Input(shape=(1,), dtype='float32', name='rv1_y')
    rs2_y = Input(shape=(1,), dtype='float32', name='rs2_y')
    rv2_y = Input(shape=(1,), dtype='float32', name='rv2_y')
    total_reserv_sum = Input(shape=(1,), dtype='float32', name='total_reserv_sum')
    total_reserv_mean = Input(shape=(1,), dtype='float32', name='total_reserv_mean')
    total_reserv_dt_diff_mean = Input(shape=(1,), dtype='float32', name='total_reserv_dt_diff_mean')
    date_int = Input(shape=(1,), dtype='float32', name='date_int')
    var_max_lat = Input(shape=(1,), dtype='float32', name='var_max_lat')
    var_max_long = Input(shape=(1,), dtype='float32', name='var_max_long')
    lon_plus_lat = Input(shape=(1,), dtype='float32', name='lon_plus_lat')

    date_emb = keras.layers.concatenate([dow_emb, month_emb, year, holiday_flg])
    date_emb = Dense(5, activation='sigmoid', name='date_merged_emb')(date_emb)

    cat_layer = keras.layers.concatenate([holiday_flg, min_visitors, mean_visitors,
                    median_visitors, max_visitors, count_observations, rs1_x, rv1_x,
                    rs2_x, rv2_x, rs1_y, rv1_y, rs2_y, rv2_y,
                    total_reserv_sum, total_reserv_mean, total_reserv_dt_diff_mean,
                    date_int, var_max_lat, var_max_long, lon_plus_lat,
                    date_emb, air_area_name_emb, air_genre_name_emb,
                    air_area_code_emb, air_genre_code_emb, air_store_id_emb])

    m = Dense(hidden1_neurons, name='hidden1',
             kernel_initializer=keras.initializers.RandomNormal(mean=0.0,
                            stddev=0.05, seed=None))(cat_layer)
    m = keras.layers.LeakyReLU(alpha=0.2)(m)
    m = keras.layers.BatchNormalization()(m)
    
    m1 = Dense(hidden2_neurons, name='hidden2')(m)
    m1 = keras.layers.LeakyReLU(alpha=0.2)(m1)
    m = Dense(1, activation='relu')(m1)

    inp_ten = [
        holiday_flg, min_visitors, mean_visitors, median_visitors, max_visitors, count_observations,
        rs1_x, rv1_x, rs2_x, rv2_x, rs1_y, rv1_y, rs2_y, rv2_y, total_reserv_sum, total_reserv_mean,
        total_reserv_dt_diff_mean, date_int, var_max_lat, var_max_long, lon_plus_lat,
        dow, year, month, air_store_id, air_area_code, air_genre_code
    ]
    inp_ten += air_area_name
    inp_ten += air_genre_name
    model = Model(inp_ten, m)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    return model
model1 = ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=3,
                    n_estimators=180, subsample=0.78, max_depth=10)
model3 = XGBRegressor(learning_rate=0.2, random_state=3, n_estimators=330, subsample=0.8, 
                      colsample_bytree=0.8, max_depth=10)
model4 = get_nn_complete_model(train, hidden1_neurons=45)

model1.fit(train[col], train['visitors'].values)
print("Model1 trained")

model3.fit(train[col], train['visitors'].values)
print("Model3 trained")

for i in range(5):
    model4.fit(nn_train[0], nn_train[1], epochs=8, verbose=0, batch_size=256, shuffle=True)
    model4.fit(nn_train[0], nn_train[1], epochs=3, verbose=0,
        batch_size=256, shuffle=True, validation_split=0.15)
model4.fit(nn_train[0], nn_train[1], epochs=4, verbose=0, batch_size=512, shuffle=True)
print("Model4 trained")

preds1 = model1.predict(train[col])
preds3 = model3.predict(train[col])
preds4 = pd.Series(model4.predict(nn_train[0]).reshape(-1)).clip(0, 6.8).values

actual_output = train['visitors'].values
print('GradientBoostingRegressor:')
plot_actual_predicted(actual_output, preds1)
print('XGBRegressor:')
plot_actual_predicted(actual_output, preds3)
print('NeuralNetwork:')
plot_actual_predicted(actual_output, preds4)

preds1 = model1.predict(test[col])
preds3 = model3.predict(test[col])
# .clip(0, 6.8) used to avoid random high values that might occur
preds4 = pd.Series(model4.predict(nn_test[0]).reshape(-1)).clip(0, 6.8).values

test['visitors'] = 0.2*preds1+0.4*preds3+0.4*preds4
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
sub1 = test[['id','visitors']].copy()
sub1['preds1'] = pd.Series(preds1)
sub1['preds3'] = pd.Series(preds3)
sub1['preds4'] = pd.Series(preds4)
print("Model predictions done.")
