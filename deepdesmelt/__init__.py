__version__ = "1.0.0"
__author__ = "Luo Guang"

# Only import the core model components that are needed everywhere
from deepdesmelt.architecture.DeepDESmelt import MOE

# Other modules should be imported explicitly when needed to avoid unnecessary dependencies
# Example:
#   from deepdesmelt.data.preprocessing import make_inference_dataset

__all__ = [
    'MOE',
]
