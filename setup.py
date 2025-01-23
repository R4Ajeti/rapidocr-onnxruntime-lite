from setuptools import setup, find_packages

# Parse requirements.txt
with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="rapidocr-onnxruntime-lite",
    version="0.1",
    description="A lightweight fork of RapidOCR optimized for minimal dependencies and easy deployment",
    author="Rinor Ajeti",
    author_email="r4ajeti@gmail.com",
    python_requires=">=3.8",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "rapidocr_onnxruntime": ["config.yaml"],
    },
    url="https://github.com/R4Ajeti/rapidocr-onnxruntime-lite",
)
