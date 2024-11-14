# Emotion classes: Angry, Negative, Neutral and Positive dictionary for splitting the data
emotion_dict = {}

angry = ['Enraged','Panicked','Stressed','Fuming','Anxious','Frustrated','Irritated','Worried',
         'Angry','Shocked','Jittery','Apprehensive','Uneasy','Annoyed','Restless','Bitter']

for anger in angry:
    emotion_dict[anger] = 'angry'
    
negative = ['Hopeless','Desolate','Depressed','Despondent','Exhausted','Sullen','Lonely','Alienated',
            'Miserable','Dismayed','Glum','Disappointed','Apathetic','Bored','Discouraged','Fatigued','Dissapointed']

for neg in negative:
    emotion_dict[neg] = 'negative'
    
neutral = ['Serene','Peaceful','Carefree','Contemplative','Restful','Thoughtful','Secure','Connected',
           'Balanced','Sleepy','Relaxed','Calm','At ease','Content','Grateful','Fulfilled']

for nu in neutral:
    emotion_dict[nu] = 'neutral'
    
positive = ['Elated','Inspired','Exhillerated','Ecstatic','Exhilarated','Exhilerated','Excited',
            'Optimistic','Playful','Enthusiastic','Cheerful','Blissful','Proud','Pleased','Pleasant',
            'Energized','Hyper','Surprised']

for pos in positive:
    emotion_dict[pos] = 'positive'

# All imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from collections import Counter
import nltk
nltk.download('vader_lexicon')  # Download the VADER lexicon for sentiment analysis
# Create a sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import numpy as np
import keras as keras
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.optimizers import Adam
from tensorflow.keras.models import load_model


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

window_size = 7 # The size for the window which needs to be taken
file_check = window_size+1 # Minimum number of data points in the file

num_classes = 4 # Number of classes which need to be taken
null_check = 60 # If a single participant has more null entries than this, that participant will not be considered


# Final list containing the 3d data formate
final_train_X = []
final_train_y = []
final_val_X = []
final_val_y = []

#List to store the name of the ID Files using in the training and validation data
ids = []

# The function to window across the entire data; for example making 30 days window for a particpant to be used for LSTM model building
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[int_cols].iloc[i:i+window_size])#.drop(columns=labels)[i:i+window_size])
        y.append(data[labels].iloc[i+window_size])
    return np.array(X), np.array(y)

# Data processing folder
data_path_processedData1_11_07 = '/Users/ayahamer/Library/CloudStorage/OneDrive-UniversityofWaterloo/3B/URA/2024-04-01/combinedData1/'
countr = 0
count_sleep = 0
count_checkin = 0
df_activity = pd.DataFrame()

for id_file in os.listdir(data_path_processedData1_11_07): # Looping through all the participants
    # Reading checkIn file for the participants
    try:
        checkIn_data_path = data_path_processedData1_11_07 + id_file + '/checkindf.csv' # reading checkIn file
        check_in_file = pd.read_csv(checkIn_data_path,usecols=['startTime','Mood_text'])
        
    # Ignoring entire code if checkIn is not present
    except Exception as e: 
        #print(f"CheckIn file for {id_file} not present!!")
        count_checkin+=1
        continue
    # Reading the Sleep file for the entire data
    try:
        sleep_data_path = data_path_processedData1_11_07 + id_file + '/sleepdf.csv' # Sleep file path
        # Columns to be selected for sleep file. 
        sleep_file = pd.read_csv(sleep_data_path,usecols=['startTime','endDay', 
                                                          #'sleep_efficiency',
                                                          'awake.duration_awake_state',
                                                          #'asleep.duration_light_sleep_state',
                                                          # 'asleep.duration_REM_sleep_state',
                                                          #'asleep.duration_asleep_state'
                                                         ])
        # Dropping all duplicate entries for the file
        sleep_file.drop_duplicates(inplace=True)
    # Ignore the entire code if sleep file not present
    except Exception as e:
        #print(f"Sleep file for {id_file} not present!!")
        count_sleep+=1
        continue
    
    # Reading the Acitivty file for the entire data
    try:
        activity_data_path = data_path_processedData1_11_07 + id_file + '/activitydf.csv' # Activity file path
        # Columns to be selected for activity file. # Only using selected sleep columns for analysis
        activity_file = pd.read_csv(activity_data_path,usecols=['startTime','endDay','net_activity_calories','summary.avg_hr', 'activity_seconds'])
    except Exception as e:
        #print(f'Activity File for {id_file} not Present')
        countr+=1
        continue
        
       
    # Checking null values for all the files
    a = check_in_file.isnull().sum()/len(check_in_file)*100
    check_in_check_null_values = a[a>null_check]
    
    b = sleep_file.isnull().sum()/len(sleep_file)*100
    sleep_check_null_values = b[b>null_check]
    c = activity_file.isnull().sum()/len(activity_file)*100
    activity_check_null_values = c[c>null_check]
    # print(list(check_in_check_null_values.index))
    
    # Printing the file if it contains more null values than the null_check
    if list(check_in_check_null_values.index):
        #print(f"Null values in CheckIn file for {id_file}")
        count_checkin+=1
        continue
    if list(sleep_check_null_values.index):
        #print(f"Null values in Sleep file for {id_file}")
        count_sleep+=1
        continue  
    if list(activity_check_null_values.index):
        countr+=1
        #print(f"Null values in Activity file for {id_file}")
        continue
  
    # CheckIn date entry conversions to datetime objects
    check_in_file['startTime'] = pd.to_datetime(check_in_file['startTime'])
    check_in_file['endDay'] = check_in_file['startTime'].dt.strftime('%Y-%m-%d')
    check_in_file['endDay']=pd.to_datetime(check_in_file['endDay'])
    
    # Sleep date entry conversions to datetime objects
    sleep_file['endDay'] = pd.to_datetime(sleep_file['endDay']).dt.tz_localize(None) # Removing any localized time
    sleep_file['startTime'] = pd.to_datetime(sleep_file['startTime'])
    
    #Activity date entry conversions
    activity_file['endDay'] = pd.to_datetime(activity_file['endDay']).dt.tz_localize(None) # Removing any localized time
    activity_file['startTime'] = pd.to_datetime(activity_file['startTime'])
    
    # Filling all null values using the previous values
    check_in_file['Mood_text'].fillna(method='ffill',inplace=True)
    # sleep_file['sleep_efficiency'].fillna(method='ffill',inplace=True)
    sleep_file['awake.duration_awake_state'].fillna(method='ffill',inplace=True)
    
    # sleep_file['asleep.duration_light_sleep_state'].fillna(method='ffill',inplace=True)
    # sleep_file['asleep.duration_REM_sleep_state'].fillna(method='ffill',inplace=True)
    # sleep_file['asleep.duration_asleep_state'].fillna(method='ffill',inplace=True)

    activity_file['net_activity_calories'].fillna(method='ffill',inplace=True)
    activity_file['summary.avg_hr'].fillna(method='ffill',inplace=True)
    activity_file['activity_seconds'].fillna(method='ffill',inplace=True)
    
    # Dropping all Nan values in checkIn
    check_in_file.dropna(inplace=True)
    activity_file.dropna(inplace=True)
    
    # Mood text class conversion
    check_in_file['Mood_text'] = check_in_file['Mood_text'].apply(lambda x: emotion_dict[x])
    check_in_file.reset_index(drop=True,inplace=True)
    
    # Appending the previous day's sleep for the current day's mood
    drop_rows = []
    count_emotion = {}
    # Looping through the entire checkIn file
    
    for check in range(1,len(check_in_file)):
        # dropping rows if CheckIn end day for previous = checkIn end day for current
        if check_in_file['endDay'][check] == check_in_file['endDay'][check-1]:
            drop_rows.append(check)
            date = check_in_file['endDay'][check]
            # Setting a singular list to later count max occurances of mood text values for the endDay
            if date not in count_emotion.keys():
                count_emotion[date] = [check_in_file['Mood_text'][check],check_in_file['Mood_text'][check-1]]
            else:
                count_emotion[date].append(check_in_file['Mood_text'][check])
                
    # Create a dictionary to store the counts of each value for each key
    count_dict = {}
    
    # Iterate through the data and count the occurrences to take only the maximum occurances
    for timestamp, values in count_emotion.items():
        count = Counter(values)
        most_common_value = max(count, key=lambda k: (count[k], values.index(k)))
        count_dict[timestamp] = most_common_value
    
    # Drop all unwanted checkIn data for Nan
    check_in_file.drop(drop_rows,inplace=True)
    check_in_file.reset_index(drop=True,inplace=True)
    
    # Loop through the entire data and set the max mood count for the corresponding checkIn file
    for check in range(len(check_in_file)):
        if check_in_file['endDay'][check] in count_dict.keys():
            check_in_file['Mood_text'][check] = count_dict[check_in_file['endDay'][check]]
    
    # Merge the maximum mood values with the original DataFrame
    df = sleep_file.merge(check_in_file, on='endDay', how='inner')
    # df = df.merge(activity_file, on='endDay',how='inner')
    if df_activity.empty:
        df_activity = activity_file
        
    else:
        df_activity = pd.concat([df_activity, activity_file])

    df = df.merge(df_activity,on='endDay',how='inner')
    
    # Stopping the processing if the file contains less than 20 values
    if len(df)<file_check:
        print(f"Insufficient data points in {id_file} for training!!!")
        continue
        
    # # Filling Nan values with Mean
    # mean_sleep_efficiency = df['sleep_efficiency'].mean()
    # # print(mean_sleep_efficiency)
    # df['sleep_efficiency'].fillna(mean_sleep_efficiency,inplace=True)
    
    mean_awake_duration_awake_state = df['awake.duration_awake_state'].mean()
    df['awake.duration_awake_state'].fillna(mean_awake_duration_awake_state,inplace=True)
    
    #appending ID file name to list, as assuming at this point, that data from this point onward will be included in X and y. 
    ids.append(id_file)
    
    # mean_asleep_duration_light_sleep_state = df['asleep.duration_light_sleep_state'].mean()
    # df['asleep.duration_light_sleep_state'].fillna(mean_asleep_duration_light_sleep_state,inplace=True)
    
    # # mean_asleep_duration_REM_sleep_state = df['asleep.duration_REM_sleep_state'].mean()
    # # df['asleep.duration_REM_sleep_state'].fillna(mean_asleep_duration_REM_sleep_state,inplace=True)
    
    # mean_asleep_duration_asleep_state = df['asleep.duration_asleep_state'].mean()
    # df['asleep.duration_asleep_state'].fillna(mean_asleep_duration_asleep_state,inplace=True)
    
    # Adding additional information on date entries for the data
    df['year'] = df['endDay'].apply(lambda x: x.year)
    df['month'] = df['endDay'].apply(lambda x: x.month)
    df['day'] = df['endDay'].apply(lambda x: x.day)
    
    # Dropping columns which are not required
    drop_col = ['endDay','startTime_x','startTime_y','startTime']
    df.drop(columns=drop_col,inplace=True)
    #print('X')

    # Converting mood values to binary
    df['angry'] = df['Mood_text'].apply(lambda x:1 if x=='angry' else 0)
    df['positive'] = df['Mood_text'].apply(lambda x:1 if x=='positive' else 0)
    df['negative'] = df['Mood_text'].apply(lambda x:1 if x=='negative' else 0)
    df['neutral'] = df['Mood_text'].apply(lambda x:1 if x=='neutral' else 0)
    
    # Dropping mood text column since we already have the classes
    df.drop(columns=['Mood_text'],inplace=True)
    
    # Standard scaling all the columns in the data except labels
    labels = ['angry','negative','positive','neutral']
    int_cols = df.drop(columns=labels).columns
    df[int_cols] = scaler.fit_transform(df[int_cols])
    column_order = ['awake.duration_awake_state', 'activity_seconds','net_activity_calories', 'summary.avg_hr', 
             'year', 'month', 'day','angry', 'positive', 'negative', 'neutral']
    df = df[column_order]
    #Converting the data into windowed data
    X, y = create_sequences(df, window_size)
    
    # Splitting up the data into training and validation by row? 70/30 validation split by length. 
    final_train_X.append(X[:round(len(X)*0.70)]) #start at first row and end at round(len(X)*0.70) exluded row
    final_train_y.append(y[:round(len(X)*0.70)]) 
    
    final_val_X.append(X[round(len(X)*0.70):])
    final_val_y.append(y[round(len(X)*0.70):]) #start at round(len(X)*0.70) row included until the last row

df['angry'].value_counts()
df['positive'].value_counts()
df['negative'].value_counts()
df['neutral'].value_counts()
df['awake.duration_awake_state'].value_counts()

#printing the how many IDs will be using in the data set and printing the names of the IDs. 
print(len(ids))
print(ids)

# Converting the lists into training and testing data
y_train = [one_val for check in final_train_y for one_val in check]
y_train = np.array(y_train)

X_train = [one_val for check in final_train_X for one_val in check]
X_train = np.array(X_train)

# Converting the lists into training and testing data
y_val = [one_val for check in final_val_y for one_val in check]
y_val = np.array(y_val)

X_val = [one_val for check in final_val_X for one_val in check]
X_val = np.array(X_val)

#print(X_val)
print(len(X_val))
print(len(X_train))

# Dictionary representation of the mood classes for labeling

{
    'angry': 0,
    'positive':1,
    'negative':2,
    'neutral':3
}

# # Train-Test split of the data using 70/30 split
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

#X_train.shape
#X_val.shape

## Define the LSTM model
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(window_size,X.shape[-1])))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(units=num_classes, activation='softmax'))  # Change 'num_classes' to the actual number of classes
opitimizer = Adam(learning_rate= 1.1000e-03) # Setting the optimizer and learning rate
# model.compile(optimizer=opitimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Model compilation

# Maintaining the callbacks for the model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)#, min_lr=0.0001)
# Define the ModelCheckpoint callback to save the best model weights
# checkpoint_path = "best_model_weights.h5"
# checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=10)

# Fitting the model using training and validation data
history = model.fit(X_train, y_train,validation_data=(X_val,y_val), epochs=50, batch_size=64, verbose=1, callbacks=[reduce_lr])

## Learning rate curve for the model
import matplotlib.pyplot as plt

# Retrieve training and validation loss values from the history object
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Create a range of epochs
epochs = range(1, len(training_loss) + 1)

# Plotting the learning curve
plt.plot(epochs, training_loss, 'bo', label='Training loss')
plt.plot(epochs, validation_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load the best model saved during training
# model = load_model("best_model_weights.h5")

# Checking the model using a testing file 
other_id_file = '0a2466d5-3fdf-482a-979a-1ae37e2491cf'
# data_path = '../23-08-02/Archive/processedData1/'
other_data_path_processedData1_11_07 = '/Users/ayahamer/Library/CloudStorage/OneDrive-UniversityofWaterloo/3B/URA/2024-04-01/'
other_check_in_data_path = '/Users/ayahamer/Library/CloudStorage/OneDrive-UniversityofWaterloo/3B/URA/2024-04-01/' + other_id_file + '/checkindf.csv'
other_sleep_data_path = '/Users/ayahamer/Library/CloudStorage/OneDrive-UniversityofWaterloo/3B/URA/2024-04-01/' + other_id_file + '/sleepdf.csv'
other_activity_data_path = '/Users/ayahamer/Library/CloudStorage/OneDrive-UniversityofWaterloo/3B/URA/2024-04-01/' + other_id_file + '/activitydf.csv'

# int_cols = df.drop(columns=labels).columns
# df[int_cols] = scaler.transform(df[int_cols])
# test_X, test_y = create_sequences(df, window_size)

# Columns to consider
other_check_in_file = pd.read_csv(other_check_in_data_path,usecols=['startTime','Mood_text'])
other_sleep_file = pd.read_csv(other_sleep_data_path,usecols=['startTime','endDay',
                                                              #'sleep_efficiency',
                                                          'awake.duration_awake_state',
                                                          #'asleep.duration_light_sleep_state',
                                                          # 'asleep.duration_REM_sleep_state',
                                                          # 'asleep.duration_asleep_state'
                                                             ])

other_activity_file = pd.read_csv(other_activity_data_path,usecols=['startTime','endDay', # Only using selected activity columns for analysis
                                                                 'net_activity_calories','summary.avg_hr',  'activity_seconds',
                                                               ])

# Checking for null values
a = other_check_in_file.isnull().sum()/len(other_check_in_file)*100
check_in_check_null_values = a[a>50]
b = other_sleep_file.isnull().sum()/len(other_sleep_file)*100
sleep_check_null_values = b[b>50]
c = other_activity_file.isnull().sum()/len(other_activity_file)*100
activity_check_null_values = c[c>50]

if list(check_in_check_null_values.index):
    print(f"Null values in CheckIn file for {id_file}")
    
if list(sleep_check_null_values.index):
    print(f"Null values in Sleep file for {id_file}")

if list(activity_check_null_values.index):
    print(f"Null values in Sleep file for {id_file}")

# Datetime conversions for checkIn files
other_check_in_file['startTime'] = pd.to_datetime(other_check_in_file['startTime'])
other_check_in_file['endDay'] = other_check_in_file['startTime'].dt.strftime('%Y-%m-%d')
other_check_in_file['endDay']=pd.to_datetime(other_check_in_file['endDay'])

# Datetime conversions for sleep files
other_sleep_file['endDay'] = pd.to_datetime(other_sleep_file['endDay']).dt.tz_localize(None)
other_sleep_file['startTime'] = pd.to_datetime(other_sleep_file['startTime'])

# Datetime conversions for Activity files
other_activity_file['endDay'] = pd.to_datetime(other_activity_file['endDay']).dt.tz_localize(None)
other_activity_file['startTime'] = pd.to_datetime(other_activity_file['startTime'])

# filling Mood with previous values
other_check_in_file['Mood_text'].fillna(method='ffill',inplace=True)

# Dropping nan values in mood
other_check_in_file.dropna(inplace=True)

other_activity_file.dropna(inplace=True)

# Mood text conversion into 4 classes
other_check_in_file['Mood_text'] = other_check_in_file['Mood_text'].apply(lambda x: emotion_dict[x])
other_check_in_file.reset_index(drop=True,inplace=True)

# Counting max occurances for mood values
drop_rows = []
count_emotion = {}
# Looping through the checkIn file
for check in range(1,len(other_check_in_file)):
    # Checking the previous if the previous row for date matches the current rows endDay
    if other_check_in_file['endDay'][check] == other_check_in_file['endDay'][check-1]:
        # Appending it into a list later to be dropped
        drop_rows.append(check)
        date = other_check_in_file['endDay'][check]
        # Setting a singular list to later count max occurances of mood text values for the endDay
        if date not in count_emotion.keys():
            count_emotion[date] = [other_check_in_file['Mood_text'][check],other_check_in_file['Mood_text'][check-1]]
        else:
            count_emotion[date].append(other_check_in_file['Mood_text'][check])
            
# Create a dictionary to store the counts of each value for each key
count_dict = {}

# Iterate through the data and count the occurrences
for timestamp, values in count_emotion.items():
    count = Counter(values)
    most_common_value = max(count, key=lambda k: (count[k], values.index(k)))
    count_dict[timestamp] = most_common_value

# Dropping unwanted rows in the list
other_check_in_file.drop(drop_rows,inplace=True)
other_check_in_file.reset_index(drop=True,inplace=True)

# Setting the max occurance values for the checkIn file mood columns
for check in range(len(check_in_file)):
    if other_check_in_file['endDay'][check] in count_dict.keys():
        other_check_in_file['Mood_text'][check] = count_dict[other_check_in_file['endDay'][check]]

# Merge the maximum mood values with the original DataFrame
df = other_sleep_file.merge(other_check_in_file, on='endDay', how='inner')
df = df.merge(other_activity_file,on='endDay',how='inner')

# if len(df)<20:
#     continue

# # Filling Nan values with mean
# mean_sleep_efficiency = df['sleep_efficiency'].mean()
# # print(mean_sleep_efficiency)
# df['sleep_efficiency'].fillna(mean_sleep_efficiency,inplace=True)

mean_awake_duration_awake_state = df['awake.duration_awake_state'].mean()
df['awake.duration_awake_state'].fillna(mean_awake_duration_awake_state,inplace=True)

# mean_asleep_duration_light_sleep_state = df['asleep.duration_light_sleep_state'].mean()
# df['asleep.duration_light_sleep_state'].fillna(mean_asleep_duration_light_sleep_state,inplace=True)

# # mean_asleep_duration_REM_sleep_state = df['asleep.duration_REM_sleep_state'].mean()
# # df['asleep.duration_REM_sleep_state'].fillna(mean_asleep_duration_REM_sleep_state,inplace=True)

# mean_asleep_duration_asleep_state = df['asleep.duration_asleep_state'].mean()
# df['asleep.duration_asleep_state'].fillna(mean_asleep_duration_asleep_state,inplace=True)


# Setting year, month and day columns
df['year'] = df['endDay'].apply(lambda x: x.year)
df['month'] = df['endDay'].apply(lambda x: x.month)
df['day'] = df['endDay'].apply(lambda x: x.day)

# Dropping unwanted columns from the data
drop_col = ['endDay','startTime_x','startTime_y']
df.drop(columns=drop_col,inplace=True)

# # check_in_file["Mood_text"] = check_in_file["Mood_text"].apply(lambda x: mood_conversion[x])
# df['angry_high'] = df['Mood_text'].apply(lambda x:1 if x=='angry_high' else 0)
# df['angry_mid'] = df['Mood_text'].apply(lambda x:1 if x=='angry_mid' else 0)
# df['angry_low'] = df['Mood_text'].apply(lambda x:1 if x=='angry_low' else 0)

# df['positive_high'] = df['Mood_text'].apply(lambda x:1 if x=='positive_high' else 0)
# df['positive_mid'] = df['Mood_text'].apply(lambda x:1 if x=='positive_mid' else 0)
# df['positive_low'] = df['Mood_text'].apply(lambda x:1 if x=='positive_low' else 0)

# df['negative_high'] = df['Mood_text'].apply(lambda x:1 if x=='negative_high' else 0)
# df['negative_mid'] = df['Mood_text'].apply(lambda x:1 if x=='negative_mid' else 0)
# df['negative_low'] = df['Mood_text'].apply(lambda x:1 if x=='negative3' else 0)

# df['neutral_high'] = df['Mood_text'].apply(lambda x:1 if x=='neutral_high' else 0)
# df['neutral_mid'] = df['Mood_text'].apply(lambda x:1 if x=='neutral_mid' else 0)
# df['neutral_low'] = df['Mood_text'].apply(lambda x:1 if x=='neutral_low' else 0)

# check_in_file["Mood_text"] = check_in_file["Mood_text"].apply(lambda x: mood_conversion[x])

# Converting into the 4 class columns
df['angry'] = df['Mood_text'].apply(lambda x:1 if x=='angry' else 0)
df['positive'] = df['Mood_text'].apply(lambda x:1 if x=='positive' else 0)
df['negative'] = df['Mood_text'].apply(lambda x:1 if x=='negative' else 0)
df['neutral'] = df['Mood_text'].apply(lambda x:1 if x=='neutral' else 0)

# Dropping the Mood_text column from the data since we already have the classes
df.drop(columns=['Mood_text'],inplace=True)

# labels = ['angry_high','angry_mid','angry_low','positive_high','positive_mid','positive_low','negative_high'
#           ,'negative_mid','negative_low','neutral_high','neutral_mid','neutral_low']

labels = ['angry','negative','positive','neutral']

# Using standard scaler to transform the data
# int_cols = df.drop(columns=labels).columns
df.drop(columns=['startTime'],inplace=True)
df[int_cols] = scaler.transform(df[int_cols])
test_X, test_y = create_sequences(df, window_size)

test_X.shape
X_train.shape
y_pred = model.predict(test_X)

# Get the index of the class with the highest probability for each prediction
predicted_classes = np.argmax(y_pred, axis=1)
predicted_classes

# Get the index of the class with the highest probability for each real class
real_other_y = np.argmax(test_y, axis=1)
real_other_y

print(classification_report(real_other_y ,predicted_classes))
print(confusion_matrix(real_other_y, predicted_classes))
