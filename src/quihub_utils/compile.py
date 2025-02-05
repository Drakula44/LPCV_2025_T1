from typing import Union, Tuple

import torch
import pydantic
import qai_hub as hub


class CompileJobConfig(pydantic.BaseModel):
    device: str
    input_shape: Tuple[int, int, int, int]

class CompileJob:
    def __init__(self, config: CompileJobConfig):
        self.device = hub.Device(config.device)
        self.input_shape: dict[str, tuple[int]] = {"image": config.input_shape}
        self.compile_job: List[hub.CompileJob] = []

    def trace_torch(self, model: torch.nn.Module) -> torch.jit.ScriptModule:
        example_input = torch.rand(self.input_shape["image"])
        return torch.jit.trace(model, example_input)

    def run_compile_job(self, model: torch.jit.ScriptModule):
        compile_job = hub.submit_compile_job(
            model=model,
            device=self.device,
            input_specs=self.input_shape,
        )
        assert compile_job is not None, "Compile job submission failed"
        if isinstance(compile_job, list):
            compile_job.extend(compile_job)
        else:
            self.compile_job.append(compile_job)

    def get_target_model(self, index: int = 0):
        assert len(self.compile_job) > 0, "No compile job submitted"
        assert index < len(self.compile_job), "Invalid index"
        return self.compile_job[index].get_target_model()

    def run(self, model: torch.nn.Module):
        traced_model = self.trace_torch(model)
        self.run_compile_job(traced_model)
