import requests
import json

scoring_uri = "http://e381f9b4-cd2a-41f3-b935-d2cd04fa48e9.southcentralus.azurecontainer.io/score"
# because the deployed service is authenticated, the primary key will be used to access it
key = 'axg5fiIJF5iZqHDELLGfUsmTsdIdX3f4'

# Two sets of data to score, so we get two results back
data = {"data":
         [
           {
            "mean_radius": 17.99,
            "mean_texture": 10.38,
            "mean_perimeter": 122.8,
            "mean_area": 1001,
            "mean_smoothness": 0.1184
			},
		  {
            "mean_radius": 13.54,
            "mean_texture": 14.36,
            "mean_perimeter": 87.46,
            "mean_area": 566.3,
            "mean_smoothness": 0.09779
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())