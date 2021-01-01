import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

# dependencies
INSTALL_REQUIRES = [
    "torch",
    "torchvision>=0.4.0",
    "numpy",
    "click",
    "matplotlib",
    "torchviz",
    "pyyaml",
    "setproctitle"
]

setuptools.setup(
    name="DLtorch",
    version="2.0.0",
    author="Junbo Zhao",
    author_email="zhaojb17@mails.tsinghua.edu.cn",
    description="Deep Learning Framework based on Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhaojb17/DLtorch",
    packages=setuptools.find_packages(),
    package_data={'': ['*.yaml']},
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
        "console_scripts": [
            "DLtorch=DLtorch.main:main"
        ]
    },
)
