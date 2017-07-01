# The package will be accessible by importing clarifai:

from clarifai import rest
from clarifai.rest import ClarifaiApp

# The client takes the `API_KEY` you created in your Clarifai
# account. You can set these variables in your environment as:

# - `CLARIFAI_API_KEY`

CLARIFAI_API_KEY = "c7e404bf2c5d4f028893ee1689b408c5"

app = ClarifaiApp(api_key=CLARIFAI_API_KEY)

# get the general model
model = app.models.get("general-v1.3")

# predict with the model
result = model.predict_by_url(url='https://samples.clarifai.com/metro-north.jpg')

print(result)