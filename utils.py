import json
import re

def parse_to_json(response):
    pattern = r"```json\s*(\{.*\})"
    # print(response)
    match = re.search(pattern, response,re.DOTALL)
    json_content= ''
    if match:
        json_content = match.group(1)  # Extract the JSON string
    else:
        print("No valid JSON found.")
    
    return json_content

if __name__== "__main__":
    string= '''
    ```json
        {
        "objects": [
            "Coca-Cola can",
            "Black mug",
            "Blue water bottle"
        ],
        "Picking up": [
            {
            "start_time": "00:00",
            "end_time": "00:03",
            "object_name": "Coca-Cola can",
            "notes": "Human hand picks up the Coca-Cola can."
            },
            {
            "start_time": "00:03",
            "end_time": "00:06",
            "object_name": "Black mug",
            "notes": "Human hand picks up the black mug after putting down the can."
            },
            {
            "start_time": "00:08",
            "end_time": "00:11",
            "object_name": "Blue water bottle",
            "notes": "Human hand picks up the blue water bottle."
            }
        ],
        "Placing down": [
            {
            "start_time": "00:03",
            "end_time": "00:04",
            "object_name": "Coca-Cola can",
            "notes": "Human hand places the Coca-Cola can back on the table."
            },
            {
            "start_time": "00:06",
            "end_time": "00:08",
            "object_name": "Black mug",
            "notes": "Human hand places the black mug back on the table."
            }
        ]
        }
        ```
    '''

    parse_to_json(string)