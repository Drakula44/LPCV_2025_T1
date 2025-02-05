import requests


def get_class1k():
    sample_classes = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
    response = requests.get(sample_classes, stream=True)
    response.raw.decode_content = True
    categories = [str(s.strip()) for s in response.raw]
    return categories
