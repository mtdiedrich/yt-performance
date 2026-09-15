DROP VIEW IF EXISTS v_label_names;
CREATE TEMP VIEW v_label_names AS
    SELECT 
        REPLACE(video_id, '.jpg', '') AS video_id, 
        GROUP_CONCAT(json_extract(value, '$.Name')) AS labels
    FROM 
        rekognition_responses,
        json_each(json_extract(label_response, '$.Labels'))
    GROUP BY 
        video_id;
SELECT * FROM v_label_names;