import sqlite3
import pandas as pd
import json
from tqdm import tqdm
import time

class DataHub:
    def __init__(self, db_location='../data/thumbnail_model.db'):
        # for labels, definitely create df of labels drop parents and vice versa
        # do a cat map
        # get tfidf data
        self.current_time = time.time()
        self.db_loc = db_location
        self._parse_rekognition_data()
        
    def get_data(self):
        data ={
            'statistics': self.get_statistics_data(),
            'queries': self.get_queries_data(),
            'tags': self.get_tags_data(),
            'topics': self.get_topics_data(),
            'parents_map': self.get_parents_map(),
            'aliases_map': self.get_aliases_map(),
            'categories_map': self.get_categories_map(),
            'label_data': self.get_label_data(),
            'text_detections_data': self.get_text_detections_data(),
            'image_dominant_colors_data': self.get_image_dominant_colors_data(),
            'instances_data': self.get_instances_data(),
            'quality_data': self.get_quality_data(),
            'label_dominant_colors_data': self.get_label_dominant_colors_data(),
            'foreground_quality_data': self.get_foreground_quality_data(),
            'foreground_dominant_color_data': self.get_foreground_dominant_color_data(),
            'background_quality_data': self.get_background_quality_data(),
            'background_dominant_color_data': self.get_background_dominant_color_data()
        }
        return data
    
    def get_statistics_data(self):
        return self.get_table('statistics', 'video_id')
    
    def get_queries_data(self):
        return self.get_table('queries', 'id')

    def get_tags_data(self):
        return self.get_table('tags', 'id')
    
    def get_topics_data(self):
        return self.get_table('topics', 'id')
    
    def get_parents_map(self):
        return self.parents_map.copy()
    
    def get_aliases_map(self):
        return self.aliases_map.copy()
    
    def get_categories_map(self):
        return self.categories_map.copy()
    
    def get_label_data(self):
        return self.label_data.copy()
        
    def get_text_detections_data(self):
        return self.text_detections_data.copy()
    
    def get_image_dominant_colors_data(self):
        return self.image_dominant_colors_data.copy()
    
    def get_instances_data(self):
        return self.instances_data.copy()
    
    def get_quality_data(self):
        return self.quality_data.copy()
    
    def get_label_dominant_colors_data(self):
        return self.label_dominant_colors_data.copy()
    
    def get_foreground_quality_data(self):
        return self.foreground_quality_data.copy()
    
    def get_foreground_dominant_color_data(self):
        return self.foreground_dominant_color_data.copy()

    def get_background_quality_data(self):
        return self.background_quality_data.copy()

    def get_background_dominant_color_data(self):
        return self.background_dominant_color_data.copy()
    
    def get_table(self, table_name, index=None):
        query = f"SELECT * FROM {table_name}"
        table = self._execute_query(query)
        if index:
            table = table.set_index(index)
        return table

    def _execute_query(self, query):
        connection = sqlite3.connect(self.db_loc)
        cursor = connection.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        connection.close()
        data_df = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
        return data_df
    
    def _get_table(self, table_name, index=None):
        table = self.get_table(table_name, index)
        return table
    
    def _parse_rekognition_data(self):
        rekognition_data = self.get_table('rekognition_responses', 'video_id')
        self._parse_label_response_data(rekognition_data)
        self._parse_text_response_data(rekognition_data)
    
    def _parse_label_response_data(self, rekognition_data):
        label_responses = rekognition_data['label_response']
        loaded_label_responses = label_responses.apply(json.loads)
        label_response_data = loaded_label_responses.apply(pd.Series).set_index(rekognition_data.index)
        self._parse_label_data(label_response_data)
        self._parse_image_properties_data(label_response_data)
    
    def _parse_label_data(self, label_response_data):
        label_data = label_response_data['Labels']
        label_data = label_data.apply(pd.Series).stack().apply(pd.Series)
        self._set_label_data(label_data)

    def _parse_image_properties_data(self, label_response_data):
        image_properties = label_response_data['ImageProperties']
        image_properties = image_properties.apply(pd.Series).set_index(label_response_data.index)
        self.image_properties_data = image_properties
        self._set_image_properties_data(image_properties)
    
    def _parse_image_dominant_colors_data(self, image_properties_data):
        # Normalize and set the index for dominant color data
        dominant_color_data = pd.json_normalize(image_properties_data['DominantColors'])
        dominant_color_data.index = image_properties_data.index
        # Initialize a list to collect all the rows
        color_data_list = []
        # Iterate through each row and process the color data
        for video_id, row in tqdm(dominant_color_data.iterrows(), total=len(dominant_color_data)):
            # Normalize the row values
            color_values = pd.json_normalize(row.values[0])
            color_values['video_id'] = video_id
            color_data_list.append(color_values)
        # Concatenate all collected data into a single DataFrame
        dominant_color_df = pd.concat(color_data_list, ignore_index=True)
        return dominant_color_df
    
    def _set_image_properties_data(self, image_properties_data):
        self.quality_data = pd.json_normalize(image_properties_data['Quality'])
        self.image_dominant_colors_data = self._parse_image_dominant_colors_data(image_properties_data)

        foreground_data = pd.json_normalize(image_properties_data['Foreground']).set_index(image_properties_data.index)
        fgdc_data = foreground_data['DominantColors']
        fgdc_df = pd.DataFrame()
        for video_id, row in tqdm(fgdc_data.items(), total=len(fgdc_data)):
            if type(row) == list:
                row_df = pd.DataFrame(row)
                row_df['video_id'] = video_id
                fgdc_df = pd.concat([fgdc_df, row_df], axis=0)
        fgdc_df.drop(columns=['HexCode'], inplace=True)
        fgdc_df.columns = ['Foreground.' + col for col in fgdc_df.columns if col != 'video_id'] + ['video_id']
        self.foreground_dominant_color_data = fgdc_df
        self.foreground_quality_data = foreground_data.drop(columns=['DominantColors'])
        background_data = pd.json_normalize(image_properties_data['Background']).set_index(image_properties_data.index)
        bgdc_data = background_data['DominantColors']
        bgdc_df = pd.DataFrame()
        for video_id, row in tqdm(bgdc_data.items(), total=len(bgdc_data)):
            if type(row) == list:
                row_df = pd.DataFrame(row)
                row_df['video_id'] = video_id
                bgdc_df = pd.concat([bgdc_df, row_df], axis=0)
        bgdc_df.drop(columns=['HexCode'], inplace=True)
        bgdc_df.columns = ['Background.' + col for col in bgdc_df.columns if col != 'video_id'] + ['video_id']
        self.background_dominant_color_data = bgdc_df
        self.background_quality_data = background_data.drop(columns=['DominantColors'])

    def _parse_text_response_data(self, rekognition_df):
        text_responses = rekognition_df['text_response']
        loaded_text_responses = text_responses.apply(json.loads)
        text_response_data = loaded_text_responses.apply(pd.Series)
        text_response_data = text_response_data.set_index(rekognition_df.index)
        self._parse_text_detections_data(text_response_data)
    
    def _parse_text_detections_data(self, text_response_data):
        text_detections = text_response_data['TextDetections']
        text_detections_data = text_detections.apply(pd.Series).set_index(text_response_data.index)
        normalized_text_detections = self._normalize_text_detections(text_detections_data)
        self._set_text_detections_data(normalized_text_detections)
    
    def _normalize_text_detections(self, text_detections_data):
        detection_keys = ['DetectedText', 'Type', 'Id', 'Confidence', 'Geometry']
        all_detections = []
        for video_id, text_detections in tqdm(text_detections_data.iterrows(), total=len(text_detections_data), desc='Parsing Text Detections'):
            detections = [det for det in text_detections if isinstance(det, dict)]
            for detection in detections:
                modified_detection = {key: detection.get(key, None) for key in detection_keys}
                if 'Geometry' in modified_detection:
                    geometry = modified_detection.pop('Geometry', {})
                    for geom_key, geom_value in geometry.items():
                        modified_detection[f'Geometry_{geom_key}'] = geom_value
                modified_detection['width'] = modified_detection['Geometry_BoundingBox'].get('Width', None)
                modified_detection['height'] = modified_detection['Geometry_BoundingBox'].get('Height', None)
                modified_detection['left'] = modified_detection['Geometry_BoundingBox'].get('Left', None)
                modified_detection['top'] = modified_detection['Geometry_BoundingBox'].get('Top', None)
                modified_detection['video_id'] = video_id
                all_detections.append(modified_detection)
        detected_text_df = pd.DataFrame(all_detections)
        detected_text_df = detected_text_df.drop(columns=['Geometry_BoundingBox', 'Geometry_Polygon'])
        return detected_text_df
    
    def _set_text_detections_data(self, text_detections_data):
        self.text_detections_data = text_detections_data
        
    def _set_label_data(self, data):
        
        def process_data(data):
            # Prepare container for batch operations
            normalized_label_data = [
                {'video_id': idx[0], 'label': row['Name'], 'confidence': row['Confidence']}
                for idx, row in data.iterrows()
            ]

            # Normalize Instances data in one go if possible
            instances_data = pd.DataFrame([
                {**{'video_id': idx[0], 'label': row['Name'], 'id': i}, **instance}
                for i, (idx, row) in tqdm(enumerate(data.iterrows()), desc='Parsing Instances', total=len(data))
                for instance in row['Instances']
            ])

            # Handle nested DominantColors data
            if not instances_data.empty:
                instances_data['id'] = range(len(instances_data))
                dominant_colors_data = pd.json_normalize(
                    instances_data.to_dict('records'), 'DominantColors', 
                    ['BoundingBox', 'Confidence', 'video_id', 'label', 'id'], 
                    errors='ignore'
                )

            # Process Parents, Aliases, and Categories in one go
            parents_map_data = [
                {'Label': row['Name'], 'Parent': parent['Name']}
                for _, row in tqdm(data.iterrows(), desc='Parsing Parents', total=len(data))
                for parent in row['Parents']
            ]

            aliases_map_data = [
                {'Label': row['Name'], 'Alias': alias['Name']}
                for _, row in tqdm(data.iterrows(), desc='Parsing Aliases', total=len(data))
                for alias in row['Aliases']
            ]
            
            categories_map_data = [
                {'Label': row['Name'], 'Category': category['Name']}
                for _, row in tqdm(data.iterrows(), desc='Parsing Categories', total=len(data))
                for category in row['Categories']
            ]

            return {
                'labels': normalized_label_data,
                'instances': instances_data,
                'dominant_colors': dominant_colors_data,
                'parents_map': parents_map_data,
                'aliases_map': aliases_map_data,
                'categories_map': categories_map_data
            }

        processed_results = process_data(data)
    
        self.parents_map = pd.DataFrame(processed_results['parents_map']).drop_duplicates()
        self.aliases_map = pd.DataFrame(processed_results['aliases_map']).drop_duplicates()
        self.categories_map = pd.DataFrame(processed_results['categories_map']).drop_duplicates()

        self.label_dominant_colors_data = processed_results['dominant_colors'].drop(columns=['BoundingBox', 'Confidence', 'HexCode'])
        instances_data = processed_results['instances'].drop(columns=['DominantColors'])
        self.instances_data = instances_data.join(pd.json_normalize(instances_data['BoundingBox'])).drop(columns=['BoundingBox'])

        self.label_data = pd.DataFrame(processed_results['labels'])