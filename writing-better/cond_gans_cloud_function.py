import sys
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def main():
    """
    Main function to load a trained GAN generator model, generate an image based on a 
    random noise vector and a specified class label, and return the image as a base64 
    encoded string.
    """
    try:
        # Read input data from stdin (expects a JSON with the class label)
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        number = int(data['data'])  # Extract the class label from input data

        # Load the pre-trained generator model
        testing_gen = load_model('/app/volume/model/cond_gans_generator_model.h5')

        # Prepare a one-hot encoded label for the specified class
        one_hot_labels = tf.one_hot(tf.constant([int(number)]), 10)
        
        # Generate a random noise vector of shape (1, 100)
        random_latent_vectors = tf.random.normal(shape=(1, 100))

        # Concatenate the noise vector and one-hot encoded label
        generated_images = testing_gen(tf.concat([random_latent_vectors, one_hot_labels], axis=-1))

        # Convert the generated image tensor to a NumPy array and remove extra dimensions
        img_array = generated_images.numpy()[0]
        img_array = np.squeeze(img_array, axis=-1)

        # Create an image from the NumPy array and convert it to a PNG format
        img = Image.fromarray((img_array * 255).astype('uint8'))  
        buffered = BytesIO()
        img.save(buffered, format="PNG")

        # Encode the image to base64 string
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Close the buffer to release resources
        buffered.close()
        
        # Return the generated image in base64 encoded format as JSON
        print(json.dumps({"image": img_str}))

    except Exception as e:
        # Handle exceptions by returning the error message as JSON
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
