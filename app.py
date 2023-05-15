import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from image import predict_image_classification_sample
from PIL import Image
from io import BytesIO
import requests

app = Flask(__name__)

project = "280882700549"
ear_endpoint_id = "7859322309482381312"
problem_endpoint_id = "2896496257608450048"
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

        ear_prediction_results, problem_prediction_results = predict_image_classification_sample(
            project=project,
            ear_endpoint_id=7859322309482381312,
            problem_endpoint_id=2896496257608450048,
            filename="received_image.jpg",
            location=location,
        )

        # Check if the image is classified as an ear
        if any(name.lower() == 'yesanear' for name, _ in ear_prediction_results):
            # Proceed with checking for ear problems
            ear_problem_keywords = ['aom', 'ote', 'csom', 'foreign', 'tympano', 'pseudomem']
            ear_problem_result = [name for name, _ in problem_prediction_results if name.lower() in ear_problem_keywords]

            if ear_problem_result:
                # Ear problem detected
                non_normal_results = [(display_name_mapping[name], confidence) for name, confidence in problem_prediction_results if name.lower() != 'normal']
                sorted_results = sorted(non_normal_results, key=lambda x: x[1], reverse=True)  # Sort by confidence, highest first

                if len(sorted_results) > 0:
                    formatted_predictions = " and/or ".join([f"{name} ({confidence * 100:.0f}% likelihood)" for name, confidence in sorted_results])
                    message = f"Always consult a doctor. Not intended to diagnose.\n\nResult: Possible ear infection or problem detected:\n\n{formatted_predictions}\n\nPlease consult a doctor for an accurate diagnosis."
                else:
                    message = "Always consult a doctor. Not intended to diagnose.\n\nResult: Possible ear infection or problem detected."
            else:
                # No ear problem detected
                non_normal_results = [(display_name_mapping[name], confidence) for name, confidence in problem_prediction_results if name.lower() != 'normal']
                sorted_results = sorted(non_normal_results, key=lambda x: x[1], reverse=True)  # Sort by confidence, highest first

                if len(sorted_results) > 0:
                    formatted_predictions = " and/or ".join([f"{name} ({confidence * 100:.0f}% likelihood)" for name, confidence in sorted_results])
                    message = f"Always consult a doctor. Not intended to diagnose.\n\nResult: No possible ear infection or problem detected. Indication of possible:\n\n{formatted_predictions}\n\nPlease consult a doctor for an accurate diagnosis."

                else:
                    message = "Always consult a doctor. Not intended to diagnose.\n\nResult: No possible ear infection or problem detected."
            resp.message(message)
        else:
            # Image is not an ear
            resp.message("Please send an image of an ear for analysis.")
    else:
        resp.message("Please send an image of an ear for analysis.")

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
