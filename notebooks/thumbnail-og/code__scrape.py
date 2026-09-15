# Good Async "Fetch Game Data"

# Parameters
concurrent_requests = 64
batch_size = 1
wait_when_rate_limited = 60
timeout_seconds = 120
use_cookies = False

import logging

# logging levels
# NOTSET
# DEBUG - Diagnostic details.
# INFO - Confirmation that things are working as expected.
# WARNING - Functioning as expected, but unexpected behavior or future issues.
# ERROR - Failed to perform a function or action.
# CRITICAL - Terminal failure.

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.debug(f"Using {concurrent_requests} concurrent requests")
logging.debug(f"Batch size: {batch_size}")
logging.debug(f"Wait when rate limited: {wait_when_rate_limited} seconds")
logging.debug(f"Timeout: {timeout_seconds} seconds")
logging.debug(f"Use cookies: {use_cookies}")

cookies = {
    'GPS':	1,
    'PREF':	'f6=40000000&tz=America.Chicago',
    'VISITOR_INFO1_LIVE': 'Wqhk8cqh7J8',
    'VISITOR_PRIVACY_METADATA':	'CgJVUxIEGgAgDg%3D%3D',
    'YSC':	'm9y0hfYs_R4'
}

if use_cookies:
    logging.debug(f"Using cookies: {cookies}")

from tqdm.notebook import tqdm

import pandas as pd

pd.set_option('display.width', 512)

import asyncio
import aiohttp
import json


semaphore = asyncio.Semaphore(concurrent_requests)


import random


async def exponential_backoff(attempt, max_delay=60):
    delay = min(2 ** attempt + random.uniform(0, 1), max_delay)
    logging.debug(f"Backing off for {delay:.2f} seconds")
    await asyncio.sleep(delay)


async def fetch_data(session, video_id_batch, max_retries=5):
    async with semaphore:
        url = f'http://localhost:8080/videos?part={part_str}&id={video_id_batch}'
        logging.debug(f"Fetching data from: {url}")    
        
        for attempt in range(max_retries):
            logging.debug(f"Attempt {attempt + 1}/{max_retries}")
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as response:
                    if response.status == 429:  # Too Many Requests
                        logging.warning(f"Rate limited. Backing off...")
                        await exponential_backoff(attempt)
                        continue
                    
                    response.raise_for_status()  # Raise an exception for bad status codes
                    text_response = await response.text()
                
                    # Find the start of the JSON object
                    json_start = text_response.find('{')

                    if json_start != -1:
                        clean_json = text_response[json_start:]
                        try:
                            json_data = json.loads(clean_json)
                            # return json_data['items'][0]  # Return the required item
                            return json_data['items']  # Return the required item
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to decode JSON: {e}")
                    else:
                        logging.error("No JSON data found in response")
                        
                    return None
            
            except aiohttp.ClientConnectorError as e:
                logging.error(f"Connection failed: {e}. Retrying...")
            except aiohttp.ServerDisconnectedError as e:
                logging.error(f"Server disconnected: {e}. Retrying...")
            except aiohttp.ClientResponseError as e:
                logging.error(f"Client response error: {e}. Retrying...")
            except asyncio.TimeoutError:
                logging.error(f"Request timed out. Retrying...")

            # Wait before retrying with exponential backoff
            await exponential_backoff(attempt)
        
        logging.error(f"Failed to fetch after {max_retries} attempts. Storing video_id_batch: {video_id_batch}")

        # log video_id_batch to a file
        with open('failed_video_id_batches.txt', 'a') as f:
            f.write(video_id_batch + '\n')

        return None
    

from sqlalchemy.exc import ProgrammingError
from sqlalchemy import text, inspect


async def fetch_all_data(video_id_batches, engine):
    data = []
    
    session_args = {'timeout': aiohttp.ClientTimeout(total=timeout_seconds)}

    if use_cookies:
        session_args['cookies'] = cookies
    
    async with aiohttp.ClientSession(**session_args) as session:
        tasks = [fetch_data(session, batch) for batch in video_id_batches]
        
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                result = await task
            
                if result:
                    result_df = treat_result(result)
                    # Try inserting the result into the database
                    insert_attempts = 0
                    max_insert_attempts = 3
                    while insert_attempts < max_insert_attempts:
                        try:
                            result_df.to_sql('scraped', con=engine, if_exists='append', index=True)
                            video_ids = ','.join(result_df.index.unique())
                            logging.info(f"Successfully inserted data for video IDs: {video_ids}")
                            break  # If successful, break out of the while loop
                        except (ProgrammingError) as e:
                            logging.error(f"OperationalError on attempt {insert_attempts + 1}: {e}")
                            error_message = str(e).lower()
                            if "does not have" in error_message:
                                logging.info("Detected missing columns. Attempting to add them.")
                                add_missing_columns(result_df)
                            else:
                                logging.error(f"Unexpected OperationalError: {e}")
                            insert_attempts += 1
                        except Exception as e:
                            
                            print(e)
                            print(type(e))

                            logging.error(f"Unexpected error while inserting data: {e}")
                            insert_attempts += 1
                    if insert_attempts == max_insert_attempts:
                        logging.error(f"Failed to insert after {max_insert_attempts} attempts for video ID: {result.get('id', 'Unknown')}")
                    else:
                        data.append(result_df)
            # except Exception as e:
            except ValueError as e:
                logging.error(f"Error processing task: {e}")
                
    return data


def treat_result(result):
    treated_result = pd.json_normalize(result)
    if 'id' in treated_result.columns:
        treated_result.set_index('id', inplace=True)
    for col in treated_result.columns:
        treated_result[col] = treated_result[col].apply(json.dumps)
    return treated_result


def add_missing_columns(result_df, engine):
    """
    This function checks for new columns in the result_df that do not exist in the 'scraped' table.
    If it finds new columns, it alters the table to add those columns.
    """
    inspector = inspect(engine)
    existing_columns = [col['name'] for col in inspector.get_columns('scraped')]
    logging.debug(f"Existing columns: {existing_columns}")

    new_columns = set(result_df.columns) - set(existing_columns)
    logging.debug(f"New columns to add: {new_columns}")
    
    if new_columns:
        with engine.begin() as conn:
            for col in new_columns:
                logging.info(f"Adding new column: {col}")
                alter_table_query = f'ALTER TABLE scraped ADD COLUMN "{col}" TEXT'
                conn.execute(text(alter_table_query))
                logging.debug(f"Executed ALTER TABLE to add column: {col}")
        logging.info(f"Added {len(new_columns)} new columns to the table.")
    else:
        logging.info("No new columns to add.")

    # Verify that columns were added
    updated_columns = [col['name'] for col in inspect(engine).get_columns('scraped')]
    logging.debug(f"Updated columns after addition: {updated_columns}")


import duckdb
import pandas as pd


import asyncio

import nest_asyncio
import time


iteration = 0

async def main():
    
    while True:
        
        logging.info(f"Iteration {iteration}")

        query = "SELECT DISTINCT video_id FROM '../../data/get_games/read_df.csv'"

        video_ids = duckdb.query(query).fetchnumpy()['video_id']

        logging.debug(f"Initial number of videos: {len(video_ids)}")

        conn = duckdb.connect('../../data/read_games.db')

        from sqlalchemy import create_engine

        engine = create_engine('duckdb:///../../data/read_games.db')

        # Fetch the table names
        tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';").fetchall()

        table_names = [table[0] for table in tables]

        tqdm.pandas()

        if 'scraped' in table_names:
            
            logging.info(f"Found existing 'scraped' table. Removing scraped videos from list.")
            scraped_df = conn.execute("SELECT * FROM scraped").fetchdf().set_index('id')
            logging.debug(f"Number of scraped videos: {len(scraped_df)}")
            video_ids = list(set(video_ids) - set(scraped_df.index))

        else:

            logging.info("No existing 'scraped' table found")

        logging.info(f"Fetching data for {len(video_ids)} videos")
            
        parts = ['mostReplayed', 'chapters', 'activity'] # 'snippet', 'statistics', 'contentDetails'
        part_str = ','.join(parts)

        logging.debug(f"Fetching parts: {part_str}")

        # temp for testing purposes. remove slice for full run
        video_id_batches = [','.join(video_ids[i:i + batch_size]) 
                            for i in range(0, len(video_ids), batch_size)]


        logging.debug(f"Batch size: {batch_size}")
        logging.debug(f"Number of batches: {len(video_id_batches)}")

        nest_asyncio.apply()

        try:
            logging.info("Starting data fetch process...")
            start = time.time()
            results = await fetch_all_data(video_id_batches, engine)
            end = time.time()
            logging.info("Completed.")
        except Exception as e:
            logging.critical(f"An unexpected error occurred: {e}")
        finally:
            logging.info("Script execution finished.")
        
        iteration += 1