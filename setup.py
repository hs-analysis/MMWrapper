from setuptools import setup, find_packages

setup(
    name="mmwrapper",
    version="0.1.0",
    author="Philipp Marquardt",
    author_email="p.marquardt15@gmail.com",
    description="A wrapper module for mmdetection and mmsegmentation",
    long_description="A wrapper module for mmdetection and mmsegmentation",
    long_description_content_type="text/markdown",
    url="https://github.com/PhilippMarquardt/MMWrapper",
    packages=find_packages(),
    classifiers=[],
    keywords="mmwrapper wrapper module",
    python_requires=">=3.6",
    install_requires=[],
    project_urls={
        "Bug Reports": "https://github.com/PhilippMarquardt/MMWrapper/issues",
        "Source": "https://github.com/PhilippMarquardt/MMWrapper",
    },
)
