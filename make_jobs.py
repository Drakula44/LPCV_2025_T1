# Wrapper za pravljenje prostih jobova za AIHUB, trenutno samo kompajliranje, profiliranje i inferencija

import qai_hub as hub
import torch
import input_getter
import numpy as np
import copy
import requests

# Apstraktna klasa za dobijanje ulaznih slika, za sada samo koristim random i onu njihovu sliku, ideja je da se kada se napravi
# dataset da nekako izvlaci iz njega


def compile_job(traced_model, input_shape):
    cj = hub.submit_compile_job(
    model=traced_model,
    device=hub.Device("Snapdragon 8 Elite QRD"),
    input_specs=dict(image=input_shape),)
    return cj

def profile_job(qai_model: hub.client.Model):
    profile_job = hub.submit_profile_job(
    model=qai_model,
    device=hub.Device("Snapdragon 8 Elite QRD"))
    return profile_job
    
def inference_job(qai_model: hub.client.Model, input_array):
    array = np.array(input_array)
    inference_job = hub.submit_inference_job(
    model=qai_model,
    device=hub.Device("Snapdragon 8 Elite QRD"),
    inputs=dict(image=[array])
    )
    return inference_job

def compile_profile_inference(model: torch.nn.Module, input_getter: input_getter.input_getter):
    input_array = input_getter.get_input()
    input_shape = input_array.shape
    traced_model = torch.jit.trace(model, input_getter.get_input())
    compile_job_object = compile_job(traced_model, input_shape)
    qai_model = compile_job_object.get_target_model()
    profile_job_object = profile_job(qai_model)
    inference_job_object = inference_job(qai_model, input_array)

    return compile_job_object, profile_job_object, inference_job_object

def inference_job_probabilities(inference_job_object: hub.client.InferenceJob):
    on_device_output = inference_job_object.download_output_data()

    # Step 5: Post-processing the on-device output
    output_name = list(on_device_output.keys())[0]
    out = on_device_output[output_name][0]
    on_device_probabilities = np.exp(out) / np.sum(np.exp(out), axis=1)

    # Read the class labels for imagenet
    sample_classes = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
    response = requests.get(sample_classes, stream=True)
    response.raw.decode_content = True
    categories = [str(s.strip()) for s in response.raw]

    # Print top five predictions for the on-device model
    print("Top-5 On-Device predictions:")
    top5_classes = np.argsort(on_device_probabilities[0], axis=0)[-5:]
    for c in reversed(top5_classes):
        print(f"{c} {categories[c]:20s} {on_device_probabilities[0][c]:>6.1%}")