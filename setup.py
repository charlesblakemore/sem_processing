from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
	long_description = fh.read()

setup(name="sem_processing", version=1.0, 
      package_dir={"": "lib"},
      packages=find_packages(), 
      author="Charles Blakemore", 
      author_email="chas.blakemore@gmail.com",
      description="Library for Processing Grayscale TIFF Images from SEM",
      long_description=long_description,
      url="https://github.com/charlesblakemore/sem_processing")

