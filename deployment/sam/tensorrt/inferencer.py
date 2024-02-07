from abc import ABC, abstractmethod

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import ImageDraw


class HostDeviceMem(object):
    """Clean data structure to handle host/device memory."""

    def __init__(self, host_mem, device_mem, npshape, name: str = None):
        """Initialize a HostDeviceMem data structure.

        Args:
            host_mem (cuda.pagelocked_empty): A cuda.pagelocked_empty memory buffer.
            device_mem (cuda.mem_alloc): Allocated memory pointer to the buffer in the GPU.
            npshape (tuple): Shape of the input dimensions.

        Returns:
            HostDeviceMem instance.
        """
        self.host = host_mem
        self.device = device_mem
        self.numpy_shape = npshape
        self.name = name

    def __str__(self):
        """String containing pointers to the TRT Memory."""
        return "Name: " + self.name + "\nHost:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        """Return the canonical string representation of the object."""
        return self.__str__()


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1, execute_v2=False, return_raw=False):
    """Generalization for multiple inputs/outputs.

    inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    # Run inference.
    if execute_v2:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    else:
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    # Synchronize the stream
    stream.synchronize()

    if return_raw:
        return outputs

    # Return only the host outputs.
    return [out.host for out in outputs]


def allocate_buffers(engine, context=None):
    """Allocates host and device buffer for TRT engine inference.

    Args:
        engine (trt.ICudaEngine): TensorRT engine
        context (trt.IExecutionContext): Context for dynamic shape engine
        reshape (bool): To reshape host memory or not (FRCNN)

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        binding_id = engine.get_binding_index(str(binding))
        binding_name = engine.get_binding_name(binding_id)
        if context:
            size = trt.volume(context.get_binding_shape(binding_id))
            dims = context.get_binding_shape(binding_id)
        else:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dims = engine.get_binding_shape(binding)
        size = engine.max_batch_size if size == 0 else size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, dims, name=binding_name))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, dims, name=binding_name))
    return inputs, outputs, bindings, stream


def load_engine(trt_runtime, engine_path):
    """Helper funtion to load an exported engine."""
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


class TRTInferencer(ABC):
    """Base TRT Inferencer."""

    def __init__(self, engine_path):
        """Init.

        Args:
            engine_path (str): The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(self.logger)
        self.engine = load_engine(self.trt_runtime, engine_path)
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

    @abstractmethod
    def infer(self):
        return

    @abstractmethod
    def __del__(self):
        return


class SAMEncoderInferencer(TRTInferencer):
    """Implements inference for the EfficientViT-SAM Encoder TensorRT engine."""

    def __init__(self, engine_path, input_shape=None, batch_size=None, data_format="channel_first"):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        super().__init__(engine_path)
        self.max_batch_size = self.engine.max_batch_size
        self.execute_v2 = False
        self.context = None

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        self._input_shape = []
        for binding in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding):
                self._input_shape = self.engine.get_binding_shape(binding)[-3:]
        assert len(self._input_shape) == 3, "Engine doesn't have valid input dimensions"

        if data_format == "channel_first":
            self.height = self._input_shape[1]
            self.width = self._input_shape[2]
        else:
            self.height = self._input_shape[0]
            self.width = self._input_shape[1]

        # set binding_shape for dynamic input
        if (input_shape is not None) or (batch_size is not None):
            self.context = self.engine.create_execution_context()
            if input_shape is not None:
                self.context.set_binding_shape(0, input_shape)
                self.max_batch_size = input_shape[0]
            else:
                self.context.set_binding_shape(0, [batch_size] + list(self._input_shape))
                self.max_batch_size = batch_size
            self.execute_v2 = True

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, self.context)
        if self.context is None:
            self.context = self.engine.create_execution_context()

        input_volume = trt.volume(self._input_shape)
        self.numpy_array = np.zeros((self.max_batch_size, input_volume))

    def infer(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        max_batch_size = self.max_batch_size
        actual_batch_size = len(imgs)
        if actual_batch_size > max_batch_size:
            raise ValueError(
                f"image_paths list bigger ({actual_batch_size}) than \
                               engine max batch size ({max_batch_size})"
            )

        self.numpy_array[:actual_batch_size] = imgs.reshape(actual_batch_size, -1)
        np.copyto(self.inputs[0].host, self.numpy_array.ravel())

        results = do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
            batch_size=max_batch_size,
            execute_v2=self.execute_v2,
        )

        y_pred = [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in results]

        return y_pred

    def __del__(self):
        if self.trt_runtime:
            del self.trt_runtime

        if self.context:
            del self.context

        if self.engine:
            del self.engine

        if self.stream:
            del self.stream

        for inp in self.inputs:
            inp.device.free()

        for out in self.outputs:
            out.device.free()


class SAMDecoderInferencer(TRTInferencer):
    """Implements inference for the EfficientViT-SAM Decoder TensorRT engine."""

    def __init__(self, engine_path, num=None, batch_size=None):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            batch_size (int): batch size for dynamic shape engine
            num (int): number of points, 2x when the prompt is box
        """
        super().__init__(engine_path)
        self.max_batch_size = self.engine.max_batch_size
        self.execute_v2 = False
        self.context = None

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        self._input_shape = []
        self.context = self.engine.create_execution_context()
        for binding in range(self.engine.num_bindings):
            # set binding_shape for dynamic input
            if self.engine.binding_is_input(binding):
                _input_shape = list(self.engine.get_binding_shape(binding)[1:])
                if binding != 0:
                    _input_shape[0] = num
                self._input_shape.append(_input_shape)
                self.context.set_binding_shape(binding, [batch_size] + _input_shape)

        self.max_batch_size = batch_size
        self.execute_v2 = True
        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, self.context)
        if self.context is None:
            self.context = self.engine.create_execution_context()

        input_volumes = [trt.volume(shape) for shape in self._input_shape]
        dtypes = (float, float, float)
        self.numpy_array = [
            np.zeros((self.max_batch_size, volume), dtype=dtype) for volume, dtype in zip(input_volumes, dtypes)
        ]

    def infer(self, inputs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        max_batch_size = self.max_batch_size

        for idx, inp in enumerate(inputs):
            actual_batch_size = len(inp)
            self.numpy_array[idx][:actual_batch_size] = inp.reshape(actual_batch_size, -1)
            np.copyto(self.inputs[idx].host, self.numpy_array[idx].ravel())

        results = do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
            batch_size=max_batch_size,
            execute_v2=self.execute_v2,
        )

        y_pred = [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in results]

        return y_pred

    def __del__(self):
        if self.trt_runtime:
            del self.trt_runtime

        if self.context:
            del self.context

        if self.engine:
            del self.engine

        if self.stream:
            del self.stream

        for inp in self.inputs:
            inp.device.free()

        for out in self.outputs:
            out.device.free()
