from .interface import DynamicalSystemAdapter

__all__ = ["DynamicalSystemAdapter", "NeuralCLBFPendulum", "SablasDrone"]


def __getattr__(name: str):
    if name == "NeuralCLBFPendulum":
        from .neural_clbf_pendulum import NeuralCLBFPendulum

        return NeuralCLBFPendulum
    if name == "SablasDrone":
        from .sablas_drone import SablasDrone

        return SablasDrone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
