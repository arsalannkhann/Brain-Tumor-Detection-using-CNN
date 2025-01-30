# Image Prediction App

This project is an image prediction application that uses a pre-trained model to predict the class of an input image.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/image-prediction-app.git
    cd image-prediction-app
    ```

2. Create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the pre-trained model and place it in the project directory.

## Usage

1. Ensure the virtual environment is activated:
    ```sh
    source venv/bin/activate
    ```

2. Run the application:
    ```sh
    python app.py
    ```

3. Use the application to predict the class of an image by calling the `get_prediction` function:
    ```python
    from PIL import Image
    import numpy as np
    from app import get_prediction

    image = Image.open('path_to_image.jpg')
    class_name, confidence = get_prediction(image)
    print(f"Predicted class: {class_name}, Confidence: {confidence}")
    ```

## Dependencies

- Python 3.x
- numpy
- Pillow
- tensorflow (or keras, depending on the model used)

## Example

```python
from PIL import Image
import numpy as np
from app import get_prediction

image = Image.open('path_to_image.jpg')
class_name, confidence = get_prediction(image)
print(f"Predicted class: {class_name}, Confidence: {confidence}")
