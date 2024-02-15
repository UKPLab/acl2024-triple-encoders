from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
    name="triple-encoders",
    version="0.0.1",
    # anynonmous for peer review
    #author="",
    #author_email="",
    #download_url="",
    description="Distributed Sentence Transformer Representations with Triple Encoders ",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(),
    python_requires=">=3.8.0",
    install_requires=[
        'datasets==2.13.1',
        'transformers>=2.0.0'
        'torch==2.0.1',
        'numpy==1.21.5',
        'pandas==1.4.4',
        'tqdm==4.64.1'
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    keywords="PyTorch NLP deep learning Sentence Transformer Triple Encoders Dialog Systems",
)