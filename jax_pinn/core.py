# %%
"""
This module provides utility functions and neural network architectures for computational tasks.
It includes functions for gradient and Laplacian computation, neural network creation, and GPU memory management.
"""
import orbax.checkpoint as ocp
import jax
import jax.numpy as np
from jax.typing import ArrayLike
from jax import Array
from jax import grad, jit, vmap, value_and_grad, hessian
from jax import random
import os
import subprocess
from tqdm.auto import tqdm
import datetime
import matplotlib.pyplot as plt
from functools import partial
import time
from typing import List, Tuple, Union, Callable, Sequence, Any, TypeVar
import flax.linen as nn
import optax
from flax.typing import PRNGKey,  Dtype, Shape, VariableDict
import warnings
from dataclasses import field
from codename import codename


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class ComputeManager:
    def __init__(self, precision: str = "float32"):
        """
        Initializes the ComputeManager with the specified precision and identifies the GPU with the minimum memory usage.

        Args:
            precision (str): The precision for computations ("float32" or "float64"). Default is "float32".
        """
        # Check the validity of the precision argument
        if precision not in ["float32", "float64"]:
            raise ValueError("Precision must be 'float32' or 'float64'.")

        self.device_id = self.get_min_memory_gpu()
        self.precision = precision
        self.dtype = np.float32 if precision == "float32" else np.float64

    def get_min_memory_gpu(self) -> str | None:
        """Identifies the GPU with the minimum used memory."""
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
            gpu_memory_list = [int(memory) for memory in output.decode(
                'utf-8').strip().split('\n')]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        if len(gpu_memory_list) == 0:
            return None

        min_memory = min(gpu_memory_list)
        min_memory_gpu = gpu_memory_list.index(min_memory)

        return str(min_memory_gpu)

    def configure_gpu(self):
        """Configure the GPU for computing."""
        if self.device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_id
        print(f"Using GPU {self.device_id} with {self.precision} precision.")
        # Ensure the computation is done with the selected precision (dtype)
        if self.precision == "float64":
            jax.config.update('jax_enable_x64', self.precision == "float64")

    def set_precision(self, precision: str):
        """Sets the precision for the computation."""
        if precision not in ["float32", "float64"]:
            raise ValueError("Precision must be 'float32' or 'float64'.")

        self.precision = precision
        self.dtype = np.float32 if precision == "float32" else np.float64
        print(f"Precision set to {self.precision}.")

    def get_precision(self) -> str:
        """Returns the current precision setting."""
        return self.precision

    def get_dtype(self):
        """Returns the current data type (np.float32 or np.float64)."""
        return self.dtype

    def set_gpu(self, device_id: str):
        """
        Manually set the GPU device to use.

        Args:
            device_id (str): The device ID (e.g., '0' for the first GPU) to be used for computation.
        """
        # Validate the GPU id by checking if the device exists
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,nounits,noheader'])
            available_gpus = [gpu.strip() for gpu in output.decode(
                'utf-8').strip().split('\n')]
            if device_id not in available_gpus:
                raise ValueError(
                    f"Invalid GPU id: {device_id}. Available GPUs: {', '.join(available_gpus)}")
        except subprocess.CalledProcessError:
            raise RuntimeError("Failed to check available GPUs.")

        # Set the selected GPU device
        self.device_id = device_id
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device_id
        print(f"Manually selected GPU {self.device_id} for computation.")


# compute_manager = ComputeManager(precision="float32")
# compute_manager.configure_gpu()


"""
Type alias for a function that takes a Tensor and returns an Array.
"""
Tensor = Array
Function = Callable[[Tensor], Array]
NN = Callable[[Tensor, VariableDict], Array]
NNModule = TypeVar("NNModule", bound=nn.Module)
# key: PRNGKey = random.PRNGKey(42)
# pi = np.pi
if 'Timetxt' not in locals() and 'Timetxt' not in globals():
    Timetxt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
else:
    Timetxt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    warnings.warn("Timetxt has been updated", stacklevel=2)

if 'runcode' not in locals() and 'runcode' not in globals():
    runcode = codename(separator='-')
else:
    runcode = codename(separator='-')
    warnings.warn("runcode has been updated", stacklevel=2)
# %%


def CreateGrad(fun: Function, dim: int) -> Function:
    """
    Creates a gradient function for a given function.

    Args:
        fun: The function to compute gradient for.
        dim: The dimension of input tensor.

    Returns:
        Function: The gradient function.
    """

    return jit(lambda _xx: ((vmap(grad(lambda _x: fun(_x.reshape(-1, dim)).reshape())))(_xx)))


def CreateLaplace(fun: Function, dim: int) -> Function:
    """
    Creates a Laplacian function for a given function.

    Args:
        fun: The function to compute Laplacian for.
        dim: The dimension of input tensor.

    Returns:
        Function: The Laplacian function.
    """

    return jit(lambda _xx: ((vmap(lambda _x: np.trace(hessian(lambda _x: fun(_x.reshape(-1, dim)).reshape())(_x))))(_xx)))
    # return jit(lambda _xx:((vmap(lambda _x:(hessian(lambda _x:fun(_x.reshape(-1,dim)).reshape())(_x))))(_xx)))


@jit
def L2Norm(x: Tensor) -> Array:
    """
    Computes the L2 norm of a tensor.

    Args:
        x: Input tensor.

    Returns:
        Array: The L2 norm value.
    """

    # return np.sqrt(np.mean((np.square(x))))
    return np.sqrt((np.sum((np.square(x)))))

# %%
# def TestFun(x: Tensor) -> Array:
#     return np.prod(np.sin(x), axis=1)


# dy = jit(lambda _xx: (
#     (vmap(grad(lambda _x: TestFun(_x.reshape(-1, 2)).reshape())))(_xx)).sum(axis=1))
# # ddy=vmap(grad(grad(TestFun)))
# # ddy=jit(lambda _xx:((vmap(lambda _x:np.trace(hessian(lambda _x:TestFun(_x.reshape(-1,2)).reshape())(_x))))(_xx)))
# dy = CreateGrad(TestFun, 10)
# ddy = CreateLaplace(TestFun, 10)
# X = np.linspace(0, pi, 1000)
# X = np.stack(np.meshgrid(X, X), axis=-1).reshape(-1, 2)
# X2 = np.zeros((1000*1000, 8))+0.5*pi
# X = np.concatenate([X, X2], axis=1)
# Y = ddy(X)
# plt.imshow(Y.reshape(1000, 1000))
# plt.colorbar()
# %%

# Y2=ddy(X)
# # Y=TestFun(X)
# plt.imshow(Y2.reshape(1000,1000))
# plt.colorbar()
# %%


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) neural network architecture.

    Attributes:
        layer_sizes: Sequence of layer sizes.
    """

    layer_sizes: Sequence[int] = field(
        default_factory=list)  # 类型标注信息 Sequence[int]
    act_function: Function = nn.tanh

    def setup(self):
        self.layers = [nn.Dense(features=size)
                       for size in self.layer_sizes[1:]]
        self.act = self.act_function

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        return self.layers[-1](x)


class ResNet(nn.Module):
    """
    A residual neural network (ResNet) architecture.

    Attributes:
        layer_sizes: Sequence of layer sizes.
    """

    layer_sizes: Sequence[int] = field(default_factory=list)
    act_function: Function = nn.tanh

    def setup(self):
        self.layers = [nn.Dense(features=size)
                       for size in self.layer_sizes[1:]]
        self.act = self.act_function

    def __call__(self, x):
        x = self.act(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.act(layer(x)) + x
        return self.layers[-1](x)

# %%


def CreateNN(NN: NNModule, InputDim: int, OutputDim: int, Depth: int, width: int, key: PRNGKey, Activation=nn.tanh) -> Tuple[NNModule, Array]:
    """
    Creates a neural network with specified architecture.

    Args:
        NN: Neural network class to instantiate.
        InputDim: Input dimension.
        OutputDim: Output dimension.
        Depth: Number of hidden layers.
        width: Width of hidden layers.
        Activation: Activation function (default: tanh).

    Returns:
        Tuple[nn.Module, Array]: The neural network and its parameters.
    """

    _nn = NN(layer_sizes=[InputDim]+[width]*Depth +
             [OutputDim], act_function=Activation)
    _x = np.zeros((1, InputDim))
    params = _nn.init(key, _x)
    return _nn, params


# net=MLP(layer_sizes=[10,10,10,1])
# key, init_key = random.split(key)  # init_key used for initialization
# dummy_x = random.uniform(init_key, (784, ))
# key, init_key = random.split(key)

# # init_key
# params = net.init(init_key, X)
# net,params=CreateNN(MLP,10,1,3,10)
# # %%
# Y=net.apply(params,X)
# %%
def CreateGradNN(fun: NN, dim: int) -> NN:
    """
    Creates a gradient function for a neural network.

    Args:
        fun: Neural network function.
        dim: Input dimension.

    Returns:
        NN: Gradient function.
    """

    return jit(lambda _xx, para: ((vmap(grad(lambda _x: fun(_x.reshape(-1, dim), para).reshape())))(_xx)))


def CreateLaplaceNN(fun: NN, dim: int) -> NN:
    """
    Creates a Laplacian function for a neural network.

    Args:
        fun: Neural network function.
        dim: Input dimension.

    Returns:
        NN: Laplacian function.
    """

    return jit(lambda _xx, para: ((vmap(lambda _x: np.trace(hessian(lambda _x: fun(_x.reshape(-1, dim), para).reshape())(_x))))(_xx)))


ActivationDict = {
    'tanh': nn.tanh,
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'swish': nn.swish,
    'gelu': nn.gelu,
    'silu': nn.silu,
    'elu': nn.elu,
    'leaky_relu': nn.leaky_relu,
    'softplus': nn.softplus,
    'softsign': nn.soft_sign
}


def get_chpt(path: str | None = None) -> Tuple[ocp.StandardCheckpointer, str | None]:
    if path is not None:
        checkpath = os.path.abspath(path)
        checkpath = ocp.test_utils.erase_and_create_empty(checkpath)
    else:
        checkpath = None
    checkpointer = ocp.StandardCheckpointer()
    return checkpointer, checkpath


def save_chpt(checkpointer: ocp.StandardCheckpointer, checkpath: str, name: str, params):
    checkpointer.save(checkpath/name, params)
    print(f"Checkpoint saved at  {checkpath}")


def load_chpt(checkpointer: ocp.StandardCheckpointer, checkpath: str, params):
    params = checkpointer.restore(checkpath, params)
    print(f"Checkpoint loaded from {checkpath}")
    return params

# %%
