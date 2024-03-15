from setuptools import find_packages, setup

setup(
    name="efficientvit",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "einops",
        "opencv-python",
        "timm",
        "tqdm",
        "torchprofile",
        "matplotlib",
        "transformers",
        "onnx",
        "onnxsim",
        "onnxruntime",
        "torchpack @ git+https://github.com/zhijian-liu/torchpack.git@3a5a9f7ac665444e1eb45942ee3f8fc7ffbd84e5",
        "tinyneuralnetwork @ git+https://github.com/alibaba/TinyNeuralNetwork.git",
        "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git",
    ],
)
