import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="DLtorch",
    version="0.0.2",
    author="Junbo Zhao",
    author_email="zhaojb17@mails.tsinghua.edu.cn",
    description="Deep Learning Framework based on Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhaojb17/DLtorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)