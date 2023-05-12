import base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.2, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    predictions = response.predictions
    prediction_results = []

    for prediction in predictions:
        for display_name, confidence in zip(prediction["displayNames"], prediction["confidences"]):
            prediction_results.append((display_name, confidence))

    return prediction_results

if __name__ == "__main__":
    project = "280882700549"
    endpoint_id = "2896496257608450048"
    location = "us-central1"
    filename = "ear.png"

    prediction_results = predict_image_classification_sample(project, endpoint_id, filename, location)
    print(prediction_results)
