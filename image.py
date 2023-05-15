import base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

def predict_image_classification_sample(
    project: str,
    ear_endpoint_id: str,
    problem_endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    encoded_content = base64.b64encode(file_content).decode("utf-8")

    ear_instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    ear_instances = [ear_instance]

    problem_instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    problem_instances = [problem_instance]

    ear_parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.2, max_predictions=5,
    ).to_value()

    problem_parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.2, max_predictions=5,
    ).to_value()

    ear_endpoint = client.endpoint_path(
        project=project, location=location, endpoint=ear_endpoint_id
    )
    problem_endpoint = client.endpoint_path(
        project=project, location=location, endpoint=problem_endpoint_id
    )

    ear_response = client.predict(
        endpoint=ear_endpoint, instances=ear_instances, parameters=ear_parameters
    )
    problem_response = client.predict(
        endpoint=problem_endpoint, instances=problem_instances, parameters=problem_parameters
    )

    ear_predictions = ear_response.predictions
    problem_predictions = problem_response.predictions

    ear_prediction_results = []
    problem_prediction_results = []

    for ear_prediction in ear_predictions:
        for display_name, confidence in zip(ear_prediction["displayNames"], ear_prediction["confidences"]):
            ear_prediction_results.append((display_name, confidence))

    for problem_prediction in problem_predictions:
        for display_name, confidence in zip(problem_prediction["displayNames"], problem_prediction["confidences"]):
            problem_prediction_results.append((display_name, confidence))

    return ear_prediction_results, problem_prediction_results

if __name__ == "__main__":
    project = "280882700549"
    ear_endpoint_id = "7859322309482381312"
    problem_endpoint_id = "2896496257608450048"
    location = "us-central1"
    filename = "ear.png"

    ear_prediction_results, problem_prediction_results = predict_image_classification_sample(
        project, ear_endpoint_id, problem_endpoint_id, filename, location
    )
    print("Ear predictions:", ear_prediction_results)
    print("Problem predictions:", problem_prediction_results)
