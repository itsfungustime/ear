import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from image import predict_image_classification_sample
from PIL import Image
from io import BytesIO
import requests

app = Flask(__name__)

project = "280882700549"
endpoint_id = "2896496257608450048"
location = "us-central1"
confidence_threshold = 0.2

display_name_mapping = {
    "aom": "Acute Otitis Media",
    "ote": "Otitis Externa",
    "csom": "Chronic Suppurative Otitis Media",
    "normal": "Normal",
    "earwax": "Earwax",
    "eartube": "Ear Tube",
    "foreign": "Foreign Object",
    "tympano": "Tympanosclerosis",
    "pseudomem": "Pseudo-tympanic membrane",
}

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
    """Respond to incoming messages with a simple text message."""
    resp = MessagingResponse()

    if request.values['NumMedia'] != '0':
        image_url = request.values['MediaUrl0']
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img.save("received_image.jpg")

        prediction_result = predict_image_classification_sample(
            project=project,
            endpoint_id=endpoint_id,
            filename="received_image.jpg",
            location=location,
        )

        # Filter predictions based on confidence threshold
        filtered_result = [(name, confidence) for name, confidence in prediction_result if confidence >= confidence_threshold]

        warning_message = "Always consult a doctor. This is not intended for diagnosis."
        mapped_result = [(display_name_mapping[name], confidence) for name, confidence in filtered_result]
        formatted_predictions = ', '.join([f"{name} ({confidence * 100:.0f}%)" for name, confidence in mapped_result])

        ear_problem_keywords = ['aom', 'ote', 'csom', 'foreign', 'tympano', 'pseudomem']
        if any(x.lower() in [name.lower() for name, _ in filtered_result] for x in ear_problem_keywords):
            message = f"{warning_message}\n\nThe predictions for the image show possible ear infection or problem: {formatted_predictions}"
        else:
            message = f"{warning_message}\n\nThe predictions for the image don't appear to show an ear infection: {formatted_predictions}"

        resp.message(message)
    else:
        resp.message("Please send an image for prediction.")

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
