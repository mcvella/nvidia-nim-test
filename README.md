# NVIDIA NIM Vision Service

This Viam module implements the `rdk:service:vision` API using NVIDIA's NIM for image classification. It allows you to send images to an NVIDIA NIM endpoint (specifically models like LLaMA for vision) and receive classification results.

This model leverages NVIDIA's NIM inference endpoints to allow for image classification and querying. NVIDIA API access is required.

## Build and Run

1.  **Prerequisites**: Ensure you have Python 3.9 or higher installed.
2.  **Clone the repository** (if you haven't already).
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    This will install the Viam SDK, `aiohttp` for asynchronous API calls, and `Pillow` for image manipulation.
4.  **Set up NVIDIA API Key**: You will need an API key for the NVIDIA NIM endpoint you intend to use. This key will be configured in the Viam app.
5.  **Run the module**:
    The module can be run directly using the `run.sh` script:
    ```bash
    ./run.sh
    ```
    This script handles setting up a virtual environment and starting the module. When run locally, the module will typically be available at `localhost:8080` (or the address specified in your Viam configuration).

## Configure Your Vision Service

To use this vision service in your robot:

1.  **Add the Module**: Ensure this module is added to your robot's configuration, either by deploying it from the Viam registry (if published) or by running it locally and connecting to it.
2.  **Create a Vision Service**:
    *   Navigate to the **Config** tab of your robot's page in the Viam app.
    *   Click on the **Components** subtab and then **Create component**.
    *   Select the `vision` type.
    *   Select the `nvidia-demo:service:nvidia-nim-test` model.
    *   Enter a name for your vision service (e.g., `my-nvidia-vision`).
    *   Click **Create**.

3.  **Configure Attributes**:
    On the new service panel, copy and paste the following attribute template into your vision service's **Attributes** box, then customize it with your values:

    ```json
    {
      "api_key": "YOUR_NVIDIA_API_KEY_HERE",
      "model_name": "meta/llama-3.2-11b-vision-instruct",
      "default_question": "Describe this image in one sentence.",
      "invoke_url": "https://integrate.api.nvidia.com/v1/chat/completions",
      "max_tokens": 100,
      "temperature": 0.7,
      "top_p": 1.0,
      "cameras": []
    }
    ```

### Attributes

The following attributes are available for the `nvidia-demo:service:nvidia-nim-test` model:

| Name               | Type          | Inclusion    | Description                                                                                                | Default Value                                      |
| ------------------ | ------------- | ------------ | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `api_key`          | string        | **Required** | Your NVIDIA API key for accessing the NIM endpoint.                                                        | N/A                                                |
| `model_name`       | string        | Optional     | The specific model identifier for the NVIDIA NIM endpoint.                                                 | `meta/llama-3.2-11b-vision-instruct`               |
| `default_question` | string        | Optional     | The default question to ask about the image if no specific question is provided in the API call.           | `describe this image`                              |
| `invoke_url`       | string        | Optional     | The full URL for the NVIDIA NIM chat completions API.                                                      | `https://integrate.api.nvidia.com/v1/chat/completions` |
| `max_tokens`       | number        | Optional     | The maximum number of tokens to generate in the response.                                                  | `512`                                              |
| `temperature`      | number        | Optional     | Controls randomness. Lower values make the model more deterministic.                                       | `1.0`                                              |
| `top_p`            | number        | Optional     | Controls nucleus sampling.                                                                                 | `1.0`                                              |
| `cameras`          | array[string] | Optional     | A list of camera component names that this service can use with `get_classifications_from_camera`.         | `[]`                                               |

**Note**: If you use the `cameras` attribute, ensure that each camera listed is also added to the `depends_on` array for this vision service component in the Viam app configuration. For example:
```json
{
  "components": [
    {
      "name": "my-camera",
      "type": "camera",
      // ... other camera config ...
    },
    {
      "name": "my-nvidia-vision",
      "model": "nvidia-demo:service:nvidia-nim-test",
      "type": "vision",
      "namespace": "rdk",
      "attributes": {
        "api_key": "YOUR_NVIDIA_API_KEY_HERE",
        "cameras": ["my-camera"]
      },
      "depends_on": ["my-camera"]
    }
  ]
}
```

## API

The `nvidia-demo:service:nvidia-nim-test` resource provides the following methods from Viam's built-in `rdk:service:vision` API:

### `get_classifications(image, count, *, extra, timeout)`

*   Sends an image to the NVIDIA NIM endpoint and returns a list of classifications.
*   `image`: The `ViamImage` to classify.
*   `count`: This parameter is not strictly used by the NVIDIA API in the same way as some classifiers that return N top results with scores. The service will return the classification provided by the model.
*   `extra` (Optional): A dictionary where you can provide a custom `question` for the image. If no `question` is provided in `extra`, the `default_question` from the attributes will be used.
    *   **Special YES/NO Question Handling**: If you format your question as `"Is this a <description>? Answer YES or NO."` (or `"Is this an <description>, Answer YES or NO."`), the service will attempt to parse this. If the NVIDIA API responds affirmatively (e.g., "YES", "Yes, it is..."), the returned `Classification` will have its `class_name` set to the `<description>` you provided (e.g., "a man wearing glasses"). If the API responds negatively (e.g., "NO"), no classification will be returned for this specific query. For all other question formats, the raw text response from the API will be used as the `class_name`.

    Example (Python SDK):
    ```python
    # Assuming 'my_vision_service' is your configured vision service
    # and 'my_image' is a ViamImage object
    classifications = await my_vision_service.get_classifications(
        my_image,
        1, # count is a required argument for the Viam SDK method
        extra={"question": "What color is the object in the center?"}
    )
    for c in classifications:
        print(f"Classification: {c.class_name}, Confidence: {c.confidence}")
    ```

### `get_classifications_from_camera(camera_name, count, *, extra, timeout)`

*   Captures an image from the specified camera and then calls `get_classifications` with that image.
*   `camera_name`: The name of the camera component to use (must be listed in the `cameras` attribute and `depends_on` for the service).
*   The `count` and `extra` parameters behave the same as in `get_classifications`.
    *   **Special YES/NO Question Handling**: (See description under `get_classifications` for details on how questions like `"Is this a <description>? Answer YES or NO."` are processed).

    Example (Python SDK):
    ```python
    classifications = await my_vision_service.get_classifications_from_camera(
        "my-camera",
        1,
        extra={"question": "Is there a person in this image? Answer YES or NO."}
    )
    for c in classifications:
        print(f"Classification: {c.class_name}, Confidence: {c.confidence}")
    ```

### `get_properties(*, extra, timeout)`
* Returns the properties of this vision service, indicating that it supports classifications.
