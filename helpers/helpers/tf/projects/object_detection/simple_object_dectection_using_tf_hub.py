"""Download the model from Tensorflow Hub

Tensorflow Hub is a repository of trained machine learning models which you can reuse in your own projects. 
- You can see the domains covered [here](https://tfhub.dev/) and its subcategories. 
- For this lab, you will want to look at the [image object detection subcategory](https://tfhub.dev/s?module-type=image-object-detection). 
- You can select a model to see more information about it and copy the URL so you can download it to your workspace. 
- We selected a [inception resnet version 2](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1)
- You can also modify this following cell to choose the other model that we selected, [ssd mobilenet version 2](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2)
"""
import tensorflow as tf
import tensorflow_hub as hub
import PIL 
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

def get_model_from_tf_hub(module_handle):
    print("Starting download of model from TF Hub...")
    model = hub.load(module_handle)
    print("Model downloaded")
    detector = model.signatures['default'] # you can use model.signatures.keys() to see all the signatures available
    return detector

def download_and_resize_image(url, new_width=256, new_height=256):
    '''
    Fetches an image online, resizes it and saves it locally.
    
    Args:
        url (string) -- link to the image
        new_width (int) -- size in pixels used for resizing the width of the image
        new_height (int) -- size in pixels used for resizing the length of the image
        
    Returns:
        (string) -- path to the saved image
    '''
    
    
    # create a temporary file ending with ".jpg"
    _, filename = tempfile.mkstemp(suffix=".jpg")
    
    # opens the given URL
    response = urlopen(url)
    
    # reads the image fetched from the URL
    image_data = response.read()
    
    # puts the image data in memory buffer
    image_data = BytesIO(image_data)
    
    # opens the image
    pil_image = PIL.Image.open(image_data)
    
    # resizes the image. will crop if aspect ratio is different.
    pil_image = PIL.ImageOps.fit(pil_image, (new_width, new_height), PIL.Image.ANTIALIAS)
    
    # converts to the RGB colorspace
    pil_image_rgb = pil_image.convert("RGB")
    
    # saves the image to the temporary file created earlier
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    
    print("Image downloaded to %s." % filename)
    
    return filename

def load_img(path):
    '''
    Loads a JPEG image and converts it to a tensor.
    
    Args:
        path (string) -- path to a locally saved JPEG image
    
    Returns:
        (tensor) -- an image tensor
    '''

    # read the file
    img = tf.io.read_file(path)

    # convert the tensor
    img = tf.image.decode_jpeg(img, channels=3)

    return img 

def run_detector(detector, path):
    """
    Runs inference on a local file using an object detection model downloaded from TF Hub

    Args:
        detector (function) -- object detection model
        path (string) -- path to an image saved locally
    """

    img = load_img(path)

    # add a batch dimension in front of the tensor
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    
    # run inference using the model
    result = detector(converted_img)

    # save the results in a dictionary
    result = {key:value.numpy() for key,value in result.items()}

    # print results
    print("Found %d objects." % len(result["detection_scores"]))

    print(result["detection_scores"])
    print(result["detection_class_entities"])
    print(result["detection_boxes"])

def main():
    detector = get_model_from_tf_hub(module_handle) 
    # download the image and use the original height and width
    downloaded_image_path = download_and_resize_image(image_url, 3872, 2592)
    run_detector(detector, downloaded_image_path)


if __name__ == "__main__":
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    # You can choose a different URL that points to an image of your choice
    image_url = "https://upload.wikimedia.org/wikipedia/commons/f/fb/20130807_dublin014.JPG"


    main()