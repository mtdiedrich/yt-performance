### Get Data
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import boto3

import pandas as pd

from tqdm import tqdm

import aiohttp
import asyncio
import nest_asyncio

import os
import sys

# Set up your API key and channel ID
# API_KEY = 'YOUR_YOUTUBE_API_KEY'
# API_KEY = 'YOUR_YOUTUBE_API_KEY' 
# API_KEY = 'YOUR_YOUTUBE_API_KEY'
API_KEY = 'YOUR_YOUTUBE_API_KEY'
# Perform the search and get the first 500 videos
queries = [
    "dead rising analysis critique review retrospective", 
    "dead island analysis critique review retrospective", 
    "left 4 dead analysis critique review retrospective",
    "resident evil analysis critique review retrospective",
    "silent hill analysis critique review retrospective",
    "dead space analysis critique review retrospective",
    "undead nightmare analysis critique review retrospective",
    ]
target_videos = 500
dir_key = "_".join(queries).replace(" ", "-")

import hashlib

# Generate a short hash of the original dir_key
dir_key = hashlib.md5(dir_key.encode()).hexdigest()[:10]  # first 10 characters of the hash

# Create a filename with the hash
filename = f"data/metadata/{dir_key}.csv"
#### Get YouTube Data
def youtube_search(query, max_results, page_token=None):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.search().list(
        part="snippet",
        q=query,
        maxResults=max_results,
        type="video",
        pageToken=page_token
    )
    response = request.execute()
    return response

def get_video_statistics(video_ids):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.videos().list(
        part="statistics",
        id=",".join(video_ids)
    )
    response = request.execute()
    return response

def get_all_videos(query, target_videos=100):
    max_results = 100  # YouTube API maxResults limit is 50 per request
    all_videos = []
    next_page_token = None
    while len(all_videos) < target_videos:
        search_results = youtube_search(query, max_results, next_page_token)
        videos = search_results.get("items", [])
        all_videos.extend(videos)

        if not (next_page_token := search_results.get("nextPageToken")):
            print('Out of videos')
            break
        sys.stdout.write('\r'+str(len(all_videos) / target_videos))
    return all_videos

def columnize_dict(frame, column_name):
    frame_sans_column = frame.drop([column_name], axis=1)
    frame_column = pd.json_normalize(frame[column_name])
    return pd.concat([frame_sans_column, frame_column], axis=1)

# Get statistics for all video IDs
def merge_statistics_data(df):
    video_ids = df['id.videoId'].tolist()
    statistics_data = []
    batch_size = 50  # API limit is 50 video IDs per request
    for i in tqdm(range(0, len(video_ids), batch_size)):
        batch_ids = video_ids[i:i + batch_size]
        stats_response = get_video_statistics(batch_ids)
        statistics_data.extend(stats_response.get("items", []))
    # Normalize and merge statistics data
    stats_df = pd.json_normalize(statistics_data)
    # Merge statistics with the original DataFrame
    merged_df = pd.merge(df, stats_df, left_on='id.videoId', right_on='id', how='inner')
    return merged_df

all_videos = [get_all_videos(query, target_videos) for query in tqdm(queries)]
video_df = pd.concat([pd.json_normalize(video) for video in all_videos])
video_df.drop_duplicates(subset=['id.videoId'], inplace=True)
merged_df = merge_statistics_data(video_df)
merged_df.drop_duplicates(subset=['id.videoId'], inplace=True)
merged_df.to_csv(f"data/metadata/{dir_key}.csv", index=False)
merged_df

#### Analyze Transcript
#### Download thumbnails
async def download_image(session, video_id, progress_bar, image_dir="images"):
    base_url = f"https://img.youtube.com/vi/{video_id}/"
    resolutions = ["maxresdefault.jpg", "hqdefault.jpg", "mqdefault.jpg", "default.jpg"]

    for res in resolutions:
        url = base_url + res
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(os.path.join(image_dir, f"{video_id}.jpg"), 'wb') as file:
                    file.write(content)
                progress_bar.update(1)
                return
    print(f"Failed to download image for video ID: {video_id}")
    progress_bar.update(1)

async def main(video_ids, image_dir):
    progress_bar = tqdm(total=len(video_ids), desc="Downloading images")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for video_id in video_ids:
            task = asyncio.ensure_future(download_image(session, video_id, progress_bar, image_dir))
            tasks.append(task)
        await asyncio.gather(*tasks)
    progress_bar.close()
    
merged_df = pd.read_csv(f"data/metadata/{dir_key}.csv")
video_ids = merged_df['id.videoId'].tolist()
# Apply the nest_asyncio patch
nest_asyncio.apply()
# Directory to save images
image_dir = f"thumbnails/{dir_key}"
os.makedirs(image_dir, exist_ok=True)
# Run the asyncio event loop
asyncio.get_event_loop().run_until_complete(main(video_ids, image_dir))
### Upload to S3
merged_df = pd.read_csv(f"data/metadata/{dir_key}.csv")

s3 = boto3.client('s3')
bucket_name = 'thumbnail-lake'

# Get a list of all the keys in the bucket
response = s3.list_objects_v2(Bucket=bucket_name)

try:
    existing_keys = [obj['Key'] for obj in response['Contents']]
    upload_keys = [f"{image_dir}/{key}.jpg" for key in merged_df['id.videoId'].tolist() if f"{key}.jpg" not in existing_keys]
except KeyError:
    upload_keys = [f"{image_dir}/{key}.jpg" for key in merged_df['id.videoId'].tolist()]
for key in tqdm(upload_keys):
    s3.upload_file(key, bucket_name, dir_key + '/' + key.split('/')[-1])
### Analyze
#### Analyze Thumbnails
##### Rekognize

import os
import json
from tqdm import tqdm
import boto3

def check_for_saved_response(file_key):
    split_key = file_key.split('/')
    thumbnail = split_key[1].replace('.jpg', '')
    return os.path.exists(f'responses/{thumbnail}.json')

def get_image_labels(bucket_name, key, client):
    label_response = client.detect_labels(
        Image={
            'S3Object': {
                'Bucket': bucket_name,
                'Name': key
            }
        },
        Features=['GENERAL_LABELS', 'IMAGE_PROPERTIES']
    )
    return label_response

def get_image_text(bucket_name, key, client):
    text_response = client.detect_text(
        Image={
            'S3Object': {
                'Bucket': bucket_name, 
                'Name': key
            }
        }
    )
    return text_response

def get_image_response(bucket_name, key, client):
    label_response = get_image_labels(bucket_name, key, client)
    text_response = get_image_text(bucket_name, key, client)
    combined_response = {
        'Label Response': label_response,
        'Text Response': text_response
    }
    return combined_response

s3 = boto3.client('s3')
bucket_name = 'thumbnail-lake'

# Get a list of all the keys in the bucket
response = s3.list_objects_v2(Bucket=bucket_name)
bucket_objects = [obj['Key'] for obj in tqdm(response['Contents'])]
local_objects = [dir_key + '/' + filename for filename in os.listdir(f'thumbnails/{dir_key}')]
file_keys = [key for key in local_objects if key not in bucket_objects]

print('Bucket Objects:', len(bucket_objects))
print('Local Objects:', len(local_objects))
print('File Keys:', len(file_keys))

# Create a new session and client
session = boto3.Session(profile_name='default')
client = session.client('rekognition')

for key in tqdm(file_keys):
    image_response = get_image_response(bucket_name, key, client)
    response_key = key.split('/')[1].replace('.jpg', '.json')
    dir_path = f'data/responses/{dir_key}'
    os.makedirs(dir_path, exist_ok=True)
    file_name = f'{dir_path}/{response_key}'
    # Save the JSON response
    with open(file_name, 'w') as f:
        json.dump(image_response, f, indent=4)    
##### Analyze Data
import os
from tqdm import tqdm
import json

def parse_labels(label_data):
    names = {
        label['Name']: 
            {
                'Confidence': label['Confidence'], 
                'Categories': label['Categories'][0]['Name']
            }
            for label in label_data
    }
    names_to_remove = []
    
    # Remove Parents
    for label in label_data:
        parents = label['Parents']
        for parent in parents:
            names_to_remove.append(parent['Name'])
    
    # # Rmeove Labels      
    # for label in label_data:
    #     parents = label['Parents']
    #     for parent in parents: 
    #         parent_name = parent['Name']
    #         if parent_name in names:
    #             names[parent_name]['Confidence'] = max(names[parent_name]['Confidence'], label['Confidence'])
    #             names_to_remove.append(label['Name'])
    
    unique_names_to_remove = list(set(names_to_remove))
    for name in unique_names_to_remove:
        names.pop(name)
    return names


def parse_text(text_data):
    thumb_text = []
    for text in text_data:
        if text['Type'] == 'LINE':
            row_thumb_text = {}
            row_thumb_text['DetectedText'] = text['DetectedText']
            row_thumb_text['Confidence'] = text['Confidence']
            row_thumb_text['Area'] = text['Geometry']['BoundingBox']['Width'] * text['Geometry']['BoundingBox']['Height']
            row_thumb_text['x'] = text['Geometry']['BoundingBox']['Left']
            row_thumb_text['y'] = text['Geometry']['BoundingBox']['Top']
            thumb_text.append(row_thumb_text)
    return thumb_text

response_dir = f'data/responses/{dir_key}'
response_files = os.listdir(response_dir)

thumbnails = {}
for file in tqdm(response_files):
    with open(f'{response_dir}/{file}') as f:
        data = json.load(f)
        thumbnail = file.replace('.json', '')
        thumbnails[thumbnail] = data

with open('data/metadata/labels.json', 'w') as f:
    json.dump(thumbnails, f, indent=4)

label_dict = {}
text_dict = {}

for key, value in tqdm(thumbnails.items()):
    label_data = value['Label Response']['Labels']
    text_data = value['Text Response']['TextDetections']
    
    parsed_labels = parse_labels(label_data)
    parsed_text = parse_text(text_data)
    
    label_dict[key] = parsed_labels
    text_dict[key] = parsed_text
import pandas as pd

label_s = {}
cat_s = {}

for key, value in tqdm(label_dict.items()):
    temp_label_df = pd.DataFrame.from_dict(value, orient='index')
    label_s[key] = temp_label_df['Confidence']
    temp_label_df = temp_label_df.groupby('Categories').max()
    cat_s[key] = temp_label_df['Confidence']
    
label_df = pd.DataFrame(label_s).T.fillna(0)
cat_df = pd.DataFrame(cat_s).T.fillna(0)
Feature selection based on correlation
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from tqdm import tqdm

# Find pairwise correlations

def calculate_pairwise_correlations(df):
    correlations = {}
    cols = df.columns
    n_cols = len(cols)
    
    # Convert dataframe to numpy array for faster processing
    data = df.to_numpy()
    
    progress_bar = tqdm(total=n_cols * (n_cols - 1) // 2)
    
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            progress_bar.update(1)
            col1, col2 = data[:, i], data[:, j]
            corr, _ = pointbiserialr(col1, col2)
            correlations[(cols[i], cols[j])] = corr
            # correlations[(cols[j], cols[i])] = corr  # This ensures symmetry
    
    progress_bar.close()
    return correlations

feature_selection_df = label_df.copy()
feature_selection_df = (feature_selection_df > 0).astype(int)
correlations = calculate_pairwise_correlations(feature_selection_df)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    vif_data = pd.Series()
    # Adding a small identity matrix to prevent singularity issues
    X = df.values  
    epsilon = 1e-5
    XTX_inv = np.linalg.inv(np.dot(X.T, X) + epsilon * np.eye(X.shape[1]))
    vif_values = np.diag(XTX_inv) * np.sum(X ** 2, axis=0)
    vif_data = pd.Series(vif_values, index=df.columns, name='VIF')
    return vif_data

def calculate_mean_corr(df):
    mean_corr = df.corr().abs().mean()
    return mean_corr

vif_data = calculate_vif(feature_selection_df)
corr_data = calculate_mean_corr(feature_selection_df)
drop_list = []

for pair, correlation in tqdm(correlations.items()):
    label1, label2 = pair
    if not (label1 in drop_list or label2 in drop_list):
        if abs(correlation) == 1:
            if vif_data[label1] > vif_data[label2]:
                drop_list.append(label1)
            else:
                drop_list.append(label2)
        elif abs(correlation) > 0.9:
            if corr_data[label1].mean() > corr_data[label2].mean():
                drop_list.append(label2)
            else:
                drop_list.append(label1)
                
feature_selection_df.drop(drop_list, axis=1, inplace=True)

# recalculate mean_corr and vif
vif_data = calculate_vif(feature_selection_df)
corr_data = calculate_mean_corr(feature_selection_df)
# dont use scientific notation
pd.options.display.float_format = '{:.4f}'.format
# drop columns with high VIF
feature_selection_df.drop(vif_data[vif_data > 5].index, axis=1, inplace=True)
feature_selection_df
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import PolynomialFeatures


merged_df = pd.read_csv(f"data/metadata/{dir_key}.csv")




# # Initialize PolynomialFeatures with interaction_only=True
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# # Fit and transform the data to create interaction terms
# X_interaction = poly.fit_transform(label_df)

# # Convert the transformed data back to a DataFrame with appropriate column names
# interaction_columns = poly.get_feature_names_out(label_df.columns)
# label_interaction_df = pd.DataFrame(X_interaction, columns=interaction_columns)

# label_interaction_df.index = label_df.index
import pandas as pd

def prep_data(frame_a, frame_b):
    # join frame a on id.videoId and frame b on index
    data_df = frame_a.join(frame_b, how='inner', on='id.videoId')
    data_features = data_df[frame_b.columns]
    target = data_df['statistics.viewCount']
    data_df = data_features
    # Convert all values greater than 0 to 1
    data_df[data_df > .9] = 1
    # rest to 0
    data_df[data_df <= .9] = 0
    data_df = data_df.astype(int)
    # Get column-wise counts
    column_counts = data_df.sum(axis=0)
    column_counts = column_counts.sort_values(ascending=False)
    # get counts above 1
    column_counts = column_counts[column_counts > 8]
    # Get the column names
    column_names = column_counts.index
    # Filter the data
    data_df = data_df[column_names]
    return data_df, target
    
merged_df = pd.read_csv(f"data/metadata/{dir_key}.csv")
label_data_df, label_target = prep_data(merged_df, feature_selection_df)
label_target = np.log1p(label_target)


import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

X_train, X_val_test, y_train, y_val_test = train_test_split(label_data_df, label_target, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Add a constant to the model (required for statsmodels)
X_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_const).fit()
influence = model.get_influence()

# Get leverage and Cook's distance
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]

# Plot leverage vs. Cook's distance
plt.scatter(leverage, cooks_d)
plt.xlabel('Leverage')
plt.ylabel("Cook's Distance")
plt.show()

# Potentially remove points with high leverage and Cook's distance
influential_points = leverage > (2 * (X_train.shape[1] + 1) / X_train.shape[0])
X_influential_removed = X_train[~influential_points]
y_influential_removed = y_train[~influential_points]
print(f"Removed {sum(influential_points)} influential points")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# Step 4: Standardize Training Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_influential_removed)
X_val_scaled = scaler.transform(X_val)  # Standardize the validation set with the same scaler
X_test_scaled = scaler.transform(X_test)  # Standardize the test set with the same scaler

# Step 5: Lasso for Feature Selection
lasso = LassoCV(cv=5, random_state=0)
lasso.fit(X_train_scaled, y_influential_removed)

# Ensure the correct feature selection
selected_features = X_train.columns[lasso.coef_ != 0]
X_train_selected = pd.DataFrame(X_train_scaled, columns=X_train.columns)[selected_features]
X_val_selected = pd.DataFrame(X_val_scaled, columns=X_val.columns)[selected_features]  # Select the same features in the validation set
X_test_selected = pd.DataFrame(X_test_scaled, columns=X_test.columns)[selected_features]  # Select the same features in the test set
from sklearn.ensemble import RandomForestRegressor

# RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

params = {
    'n_estimators': [8, 16, 32, 64, 128, 256, 512],
    'max_depth': [8, 16, 32, 64, 128, None],
    'min_samples_split': [2, 4, 8, 16, 32, 64, 128, 256, 512],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor()
rf_search = RandomizedSearchCV(rf, param_distributions=params, n_iter=128, cv=8, verbose=2, random_state=42, n_jobs=-1)
rf_search.fit(X_train_selected, y_influential_removed)

# Best parameters
print(rf_search.best_params_)
# Best score
print(rf_search.best_score_)
# Predictions
y_pred = rf_search.predict(X_test_selected)
# Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# R2 Score
r2 = rf_search.best_estimator_.score(X_test_selected, y_test)
print(f"R2 Score: {r2}")
import matplotlib.pyplot as plt

#print 100 rows
pd.set_option('display.max_rows', 100)

# plot feature importances that are larger than 0
importances = rf_search.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(X_train_selected.shape[1]), importances[indices])
plt.xticks(range(X_train_selected.shape[1]), X_train_selected.columns[indices], rotation=90)
plt.title("Feature Importances")
plt.show()
# save features to list
features = X_train_selected.columns[indices].tolist()

label_important_df = label_data_df[features]

X_imp_train, X_imp_val_test, y_imp_train, y_imp_val_test = train_test_split(label_important_df, label_target, test_size=0.2, random_state=42)
X_imp_val, X_imp_test, y_imp_val, y_imp_test = train_test_split(X_imp_val_test, y_imp_val_test, test_size=0.5, random_state=42)

model = rf_search.best_estimator_
model.fit(X_imp_train, y_imp_train)
y_pred = model.predict(X_imp_val)
mse = mean_squared_error(y_imp_val, y_pred)
print(f"Mean Squared Error: {mse}")

# R2 Score
r2 = model.score(X_imp_val, y_imp_val)
print(f"R2 Score: {r2}")

# build keras regressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(16, input_dim=X_imp_train.shape[1], activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer=Adam(0.001))
model.fit(X_train_selected, y_train, epochs=2048, batch_size=32, validation_data=(X_val_selected, y_val), verbose=0)

# predict the target on the test
y_pred = model.predict(X_test_selected)

# Root Mean Squared Error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# calculate R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Assuming X and y are your features and target variable

# Step 1: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(label_data_df, label_target, test_size=0.2, random_state=0)

# Step 2: Initial Perfect Correlation Check on Training Data
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] == 1)]
X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)  # Drop the same columns from the test set

# Step 3: VIF Calculation and Dropping High VIF Features in Training Data
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

vif_data = calculate_vif(X_train)
high_vif_features = vif_data[vif_data['VIF'] > 5]['feature']
X_train = X_train.drop(columns=high_vif_features)
X_test = X_test.drop(columns=high_vif_features)  # Drop the same columns from the test set

# Step 4: Standardize Training Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Standardize the test set with the same scaler

# Step 5: Lasso for Feature Selection
lasso = LassoCV(cv=5, random_state=0)
lasso.fit(X_train_scaled, y_train)

# Ensure the correct feature selection
selected_features = X_train.columns[lasso.coef_ != 0]
X_train_selected = pd.DataFrame(X_train_scaled, columns=X_train.columns)[selected_features]
X_test_selected = pd.DataFrame(X_test_scaled, columns=X_test.columns)[selected_features]  # Select the same features in the test set

# Step 6: Random Forest for Feature Importance
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train_selected, y_train)
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_selected.columns,
    'importance': importances
})

# Set your threshold for feature importance
threshold = 0.01
important_features = feature_importance_df[feature_importance_df['importance'] > threshold]['feature']
X_train_final = X_train_selected[important_features]
X_test_final = X_test_selected[important_features]  # Select the same features in the test set

# Step 7: Model Building and Cook’s Distance
# Add constant to the model
X_train_const = sm.add_constant(X_train_final)
X_test_const = sm.add_constant(X_test_final)

# Fit the model
model = sm.OLS(y_train, X_train_final).fit()

# Calculate Cook’s distance
influence = model.get_influence()
cooks = influence.cooks_distance[0]
# Typical threshold for influential points
threshold_cooks = 4 / len(y_train)
# Filter out influential points
non_influential_indices = np.where(cooks < threshold_cooks)[0]
X_train_clean = X_train_final.iloc[non_influential_indices]
y_train_clean = y_train.iloc[non_influential_indices]

# Refit the model with clean data
model_clean = sm.OLS(y_train_clean, sm.add_constant(X_train_clean)).fit()

# Evaluate the model on the test set
y_pred = model_clean.predict(X_test_const)

# Optional: Evaluate performance
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

#### Analyze Titles
#### Analyze Description
#### Analyze Transcript
#### Analyze Comments
# For each creator in the dataset, get all their videos and statistics to compare their video on this list to typical performance
# Quantify or categorize descriptions, titles, and pairing. A local LLM could likely help here.

# Dont use until data is reworked to have creator folders
# Dont use until data is reworked to have creator folders
# Dont use until data is reworked to have creator folders


# upload thumbnails to s3 if not already there
import boto3
from tqdm import tqdm

import os

s3 = boto3.client('s3')

bucket = 'thumbnailstoragebucket'

thumbnail_loc = './data'
creators = os.listdir(thumbnail_loc)
bucket_contents = s3.list_objects(Bucket=bucket)['Contents']
# Dont use until data is reworked to have creator folders
for creator in creators:
    for thumbnail in os.listdir(thumbnail_loc + '/' + creator):
        key = creator + '/' + thumbnail
        if key not in bucket_contents:
            s3.upload_file(thumbnail_loc + '/' + creator + '/' + thumbnail, bucket, key)
import os
import json
from tqdm import tqdm
import boto3

def check_for_saved_response(file_key):
    split_key = file_key.split('/')
    creator = split_key[0]
    thumbnail = split_key[1].replace('.jpg', '')
    dir_path = f'./output_data/labels/{creator}'
    if not os.path.exists(dir_path):
        return False
    return os.path.exists(f'{dir_path}/{thumbnail}.json')

bucket = 'thumbnailstoragebucket'
s3 = boto3.resource('s3')
thumbnail_bucket = s3.Bucket(bucket)
objects = thumbnail_bucket.objects.all()
file_keys = [obj.key for obj in objects if not check_for_saved_response(obj.key)]

for key in tqdm(file_keys):
    session = boto3.Session(profile_name='default')
    client = session.client('rekognition')
    label_response = client.detect_labels(
        Image={
            'S3Object': {
                'Bucket': bucket,
                'Name': key
            }
        },
        Features=['GENERAL_LABELS', 'IMAGE_PROPERTIES']
    )
    
    text_response = client.detect_text(
        Image={
            'S3Object': {
                'Bucket': bucket, 
                'Name': key
            }
        }
    )
    
    combined_response = {
        'Label Response': label_response,
        'Text Response': text_response
    }
    
    split_key = key.split('/')
    creator = split_key[0]
    thumbnail = split_key[1].replace('.jpg', '')
    
    # Ensure the directory path is correct
    dir_path = f'./output_data/labels/{creator}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # Corrected to create the intended path
    
    # Save the JSON response
    with open(f'{dir_path}/{thumbnail}.json', 'w') as f:
        json.dump(combined_response, f, indent=4)
label_data = './output_data/labels'
creator_data = os.listdir(label_data)
creators = {}
for creator in tqdm(creator_data):
    thumbnail_data = os.listdir(label_data + '/' + creator)
    thumbnails = {}
    for thumbnail in thumbnail_data:
        with open(label_data + '/' + creator + '/' + thumbnail) as f:
            data = json.load(f)
            thumbnails[thumbnail.replace('.json', '')] = data
    creators[creator] = thumbnails
videos = {}
for key, value in tqdm(creators.items()):
    for video_id, data in value.items():
        if video_id not in videos:
            videos[video_id] = data
def filter_by_instance(data):
    labels = data['Label Response']['Labels']
    has_instances = [label for label in labels if check_if_label_has_instances(label)]
    return has_instances

def check_if_label_has_instances(label):
    return label['Instances'] != []

has_instance_data = {}
for key, value in tqdm(videos.items()):
    filtered_data = filter_by_instance(value)
    has_instance_data[key] = filtered_data
import pandas as pd
metadata_df = pd.read_csv('metadata.csv')
# min-max scale view count for each creator
metadata_df['view_count_scaled'] = metadata_df.groupby('creator')['view_count'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))


# Create dict of video_id and view_count_scaled
view_count_scaled_dict = metadata_df.set_index('video_id')['view_count_scaled'].to_dict()


view_count_scaled_dict
instance_data = {}
for key, value in tqdm(has_instance_data.items()):
    if key not in view_count_scaled_dict:
        continue
    
    for label in value:
        instances = label['Instances']
        for instance in instances:
            if instance['DominantColors']:
                bounding_box = instance['BoundingBox']
                dominant_colors = instance['DominantColors']
                colors = []
                for color in dominant_colors:
                    r = color['Red']
                    g = color['Green']
                    b = color['Blue']
                    p = color['PixelPercent']
                    colors.append((r, g, b, p))
                row_instance_data = {
                    'Name': label['Name'],
                    'Confidence': instance['Confidence'],
                    'Width': bounding_box['Width'],
                    'Height': bounding_box['Height'],
                    'Left': bounding_box['Left'],
                    'Top': bounding_box['Top'],
                    'Parents': label['Parents'],
                    'Categories': label['Categories'],
                    'DominantColors': colors,
                    'Vioews': view_count_scaled_dict[key]
                }
                instance_data[key] = row_instance_data
 
                
len(instance_data.items())


import os
import json
from tqdm import tqdm
import boto3

def check_for_saved_response(file_key):
    split_key = file_key.split('/')
    creator = split_key[0]
    thumbnail = split_key[1].replace('.jpg', '')
    dir_path = f'./output_data/labels/{creator}'
    if not os.path.exists(dir_path):
        return False
    return os.path.exists(f'{dir_path}/{thumbnail}.json')

bucket = 'thumbnailstoragebucket'
s3 = boto3.resource('s3')
thumbnail_bucket = s3.Bucket(bucket)
objects = thumbnail_bucket.objects.all()
file_keys = [obj.key for obj in objects if not check_for_saved_response(obj.key)]

print(file_keys)

for key in tqdm(file_keys):
    session = boto3.Session(profile_name='default')
    client = session.client('rekognition')
    label_response = client.detect_labels(
        Image={
            'S3Object': {
                'Bucket': bucket,
                'Name': key
            }
        },
        Features=['GENERAL_LABELS', 'IMAGE_PROPERTIES']
    )
    
    text_response = client.detect_text(
        Image={
            'S3Object': {
                'Bucket': bucket, 
                'Name': key
            }
        }
    )
    
    combined_response = {
        'Label Response': label_response,
        'Text Response': text_response
    }
    
    split_key = key.split('/')
    creator = split_key[0]
    thumbnail = split_key[1].replace('.jpg', '')
    
    # Ensure the directory path is correct
    dir_path = f'./output_data/labels/{creator}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # Corrected to create the intended path
    
    # Save the JSON response
    with open(f'{dir_path}/{thumbnail}.json', 'w') as f:
        json.dump(combined_response, f, indent=4)
        
        
import pandas as pd

metadata_df = pd.read_csv('metadata.csv')
# metadata_df where 'live' in snippet
metadata_df = metadata_df[~metadata_df['snippet'].str.contains('live')]
metadata_df = metadata_df[['video_id', 'title', 'creator', 'view_count', 'description']]
# get df of creators and average, stdev, min, and max of views
creator_stats = metadata_df.groupby('creator')['view_count'].agg(['mean', 'std', 'min', 'max']).reset_index()
avg_adj_df = metadata_df.copy()
# Get z-score
avg_adj_df['z_score'] = avg_adj_df.groupby('creator')['view_count'].transform(lambda x: (x - x.mean()) / x.std())
# get min-max scaling
avg_adj_df['min_max'] = avg_adj_df.groupby('creator')['view_count'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
label_df = pd.read_csv('labels.csv')
label_df
# rename Unnamed: 0	to video_id
label_df.rename(columns={'Unnamed: 0': 'video_id'}, inplace=True)
# remove .jpg from video_id
label_df['video_id'] = label_df['video_id'].str.replace('.jpg', '')
# split the video_id and use second part as video_id
label_df['video_id'] = label_df['video_id'].str.split('/').str[1]
# set video_id as index
label_df.set_index('video_id', inplace=True)
# drop if video_id is not in metadata_df
label_df = label_df[label_df.index.isin(metadata_df['video_id'])]
# drop columns with all zeros
label_df = label_df.loc[:, (label_df != 0).any(axis=0)]
# convert to binary
label_df = label_df.astype(bool)
label_df
# join the metadata and labels
corr_df = avg_adj_df.merge(label_df, on='video_id', how = 'outer')

# get top corrleations with view_count
pd.options.display.float_format = '{:.2f}'.format
z_corr = corr_df.corr(numeric_only=True)['z_score'].sort_values(ascending=False).drop(['min_max', 'z_score', 'view_count'])
print(z_corr)

mm_corr = corr_df.corr(numeric_only=True)['min_max'].sort_values(ascending=False).drop(['min_max', 'z_score', 'view_count'])
print(mm_corr)

# combine z_corr and mm_corr
features = pd.concat([z_corr, mm_corr], axis=1)
features.columns = ['z_score', 'min_max']

# import regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

df = avg_adj_df.merge(label_df, on='video_id', how = 'outer').dropna()  

# Get coluns in label_df
cols = label_df.columns

X = df[cols]
y = df[['z_score', 'min_max']].astype(float)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# create a dictionary of models
models = {
    # 'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    # 'Decision Tree': DecisionTreeRegressor(),
}

# fit and evaluate the models
results = {}
for name, model in models.items():
    for target in ['z_score', 'min_max']:
        target_y_train = y_train[target]
        target_y_val = y_val[target]
        model.fit(X_train, target_y_train)
        y_pred = model.predict(X_val)
        result = mean_squared_error(target_y_val, y_pred)
        print(f'{name} {target}: {result}')
        results[(name, target)] = result
        if name == 'Linear Regression':
            feature_importances= model.coef_
        elif name == 'Decision Tree':
            feature_importances = model.feature_importances_
        elif name == 'Random Forest':
            feature_importances = model.feature_importances_
        features[name + ' ' + target] = feature_importances

# pandas display 5 digits
pd.options.display.float_format = '{:.5f}'.format

# Scale every feature to 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
# Get mean of row for features
features['mean'] = features.mean(axis=1)
# sort
features = features.sort_values('mean', ascending=False)
features

