import tensorflow.lite as tflite

def download_model_from_compile_job(compile_job, download_path):
    if(not download_path.endswith(".tflite")):
        download_path += ".tflite"
    target_model = compile_job.get_target_model()
    target_model.download(download_path)
    return download_path

class TFHelper:

    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        
    def get_interpreter(self):
        return self.interpreter
    
    def run_inference(self, input_array):
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_array)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data
