# Wrapper za pravljenje prostih jobova za AIHUB, trenutno samo kompajliranje, profiliranje i inferencija

import qai_hub as hub
import torch
import input_getter
import numpy as np
import copy
import requests
import helper


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
    input_shape = input_getter.get_input_numpy().shape
    traced_model = torch.jit.trace(model, input_getter.get_input_torch())
    compile_job_object = compile_job(traced_model, input_shape)
    qai_model = compile_job_object.get_target_model()
    profile_job_object = profile_job(qai_model)
    inference_job_object = inference_job(qai_model, input_getter.get_input_numpy())

    return compile_job_object, profile_job_object, inference_job_object
