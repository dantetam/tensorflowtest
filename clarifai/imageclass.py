# The package will be accessible by importing clarifai:

from clarifai import rest
from clarifai.rest import ClarifaiApp

import json

# The client takes the `API_KEY` you created in your Clarifai
# account. You can set these variables in your environment as:
# - `CLARIFAI_API_KEY`

CLARIFAI_API_KEY = "c7e404bf2c5d4f028893ee1689b408c5"
app = ClarifaiApp(api_key=CLARIFAI_API_KEY)

# get the general model
model = app.models.get("general-v1.3")

def predictConceptsFromUrl(imageUrl):
  # predict with the model
  result = model.predict_by_url(url=imageUrl)
  #print(result)
  #print(type(result))

  with open('clarifai_api_results.txt', 'w') as outfile:
    json.dump(result, outfile, sort_keys = True, indent = 4,
                 ensure_ascii = False)

  concepts = result["outputs"][0]["data"]["concepts"]   

  print("Found concepts for image url: " + imageUrl)
  for concept in concepts:
    print(concept)
   
  return concepts
           
predictConceptsFromUrl('http://cdn2.hubspot.net/hub/231966/file-1100227115-jpg/+FTMBA/Blog/Schmitz_Haas_IMG_0163.jpg?t=1405108650085')
predictConceptsFromUrl('http://i.imgur.com/Q7kFeGN.jpg')
#predictConceptsFromUrl('./testimage.jpg')