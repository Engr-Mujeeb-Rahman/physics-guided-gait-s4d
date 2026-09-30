"""Export trained PyTorch GaitS4D checkpoint to ONNX format.

Governed by architecture.md §5 and design.md §7.
Validates numerical parity between PyTorch and ONNXRuntime outputs.
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models.backbone import GaitS4D

try:
    import onnx
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class FoldedS4DBlock(torch.nn.Module):
    """Inference-optimized S4D block with pre-folded real 1D depthwise convolution kernel."""
    def __init__(self, s4d_block, seq_len: int = 100):
        super().__init__()
        self.norm = s4d_block.norm
        self.d_model = s4d_block.s4d.d_model
        self.seq_len = seq_len
        with torch.no_grad():
            K = s4d_block.s4d.kernel(seq_len).unsqueeze(1)  # (d_model, 1, seq_len)
        self.register_buffer("K", K)
        self.D = s4d_block.s4d.D
        self.act = s4d_block.act
        self.ffn = s4d_block.ffn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm(x)
        u_conv = norm_x.transpose(1, 2)
        y_conv = torch.nn.functional.conv1d(
            u_conv, self.K, padding=self.seq_len - 1, groups=self.d_model
        )[:, :, :self.seq_len]
        y = (y_conv + self.D.unsqueeze(-1) * u_conv).transpose(1, 2)
        h = x + y
        return h + self.ffn(self.act(h))


class FoldedGaitS4D(torch.nn.Module):
    """Inference-time folded GaitS4D model with precomputed real convolution kernels."""
    def __init__(self, model, seq_len: int = 100):
        super().__init__()
        self.in_proj = model.in_proj
        self.use_film = model.use_film
        self.films = model.films
        self.blocks = torch.nn.ModuleList([
            FoldedS4DBlock(b, seq_len=seq_len) for b in model.blocks
        ])
        self.final_norm = model.final_norm
        self.head = model.head
        
    def forward(self, x: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.use_film:
                h = self.films[i](h, speed)
        h = self.final_norm(h)
        return self.head(h)


def export_model_to_onnx(
    checkpoint_path: str = "experiments/phase3_conditioning_physics/model_best.pt",
    output_onnx_path: str = "experiments/phase3_conditioning_physics/model.onnx",
    config: str = "conditioning_physics",
    d_model: int = 64,
    n_layers: int = 4,
    seq_len: int = 100,
    opset_version: int = 14,
):
    """Export GaitS4D checkpoint to ONNX with validation check."""
    print(f"==================================================")
    print(f"EXPORTING TO ONNX: {checkpoint_path}")
    print(f"Target Output: {output_onnx_path}")
    print(f"==================================================")
    
    use_film = config in ["conditioning", "conditioning_physics"]
    model = GaitS4D(
        in_channels=102,
        out_channels=3,
        d_model=d_model,
        d_state=64,
        n_layers=n_layers,
        dropout=0.0,
        use_film=use_film,
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_x = torch.randn(1, seq_len, 102, dtype=torch.float32)
    dummy_speed = torch.tensor([[0.5]], dtype=torch.float32)
    
    # Use folded inference model (precomputed real 1D depthwise convolution kernels)
    wrapper = FoldedGaitS4D(model, seq_len=seq_len)
    wrapper.eval()
    
    with torch.no_grad():
        torch_out = wrapper(dummy_x, dummy_speed).numpy()
        
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    
    input_names = ["kinematics_muscles", "speed"]
    output_names = ["joint_moments"]
    dynamic_axes = {
        "kinematics_muscles": {0: "batch_size"},
        "speed": {0: "batch_size"},
        "joint_moments": {0: "batch_size"},
    }
    
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_speed),
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    
    file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Successfully exported ONNX model ({file_size_mb:.2f} MB).")
    
    if HAS_ORT:
        # Check model with ONNX
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model passed onnx.checker.check_model() successfully.")
        
        # Verify numerical parity with ONNXRuntime
        session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {
            "kinematics_muscles": dummy_x.numpy(),
            "speed": dummy_speed.numpy(),
        }
        ort_out = session.run(None, ort_inputs)[0]
        
        max_diff = np.max(np.abs(torch_out - ort_out))
        print(f"PyTorch vs. ONNXRuntime maximum absolute discrepancy: {max_diff:.2e}")
        assert max_diff < 1e-4, f"Discrepancy too large: {max_diff}"
        print("NUMERICAL PARITY CONFIRMED: PyTorch output matches ONNXRuntime.")
        
    print("==================================================\n")
    return output_onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="experiments/phase3_conditioning_physics/model_best.pt")
    parser.add_argument("--output", type=str, default="experiments/phase3_conditioning_physics/model.onnx")
    parser.add_argument("--config", type=str, default="conditioning_physics")
    args = parser.parse_args()
    
    export_model_to_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        config=args.config,
    )
