import numpy as np
import requests
import qai_hub as hub

def get_imagenet_categories():
    sample_classes = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
    response = requests.get(sample_classes, stream=True)
    response.raw.decode_content = True
    return [str(s.strip()) for s in response.raw]


# Nisam siguran da li dobro radi kada se ne posalji ndarray
def print_probablities_from_output(output, categories = None, top = 5, modelname = "", filename = ""):
    if(type(output) != np.ndarray):
        output = output.cpu().detach().numpy()
    if categories is None:
        categories = get_imagenet_categories()
    probabilities = np.exp(output) / np.sum(np.exp(output), axis=1)
    # Print top five predictions for the on-device model
    print(f"Top-{top} predictions for {modelname} on {filename}:".format(top))
    top_classes = np.argsort(probabilities[0], axis=0)[-top:]
    for clas in reversed(top_classes):
        print(f"{clas} {categories[clas]:20s} {probabilities[0][clas]:>6.1%}")

def inference_job_probabilities(inference_job_object: hub.client.InferenceJob):
    on_device_output = inference_job_object.download_output_data()
    output_name = list(on_device_output.keys())[0]
    out = on_device_output[output_name][0]
    print_probablities_from_output(out, top=5, modelname="Cloud model")