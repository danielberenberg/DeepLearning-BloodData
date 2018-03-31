from setuptools import setup

setup(name='basic_utils',
      version='0.1',
      description='Utility functions for partitioning and organizing video frames',
      url='https://github.com/danielberenberg/DeepLearning-BloodData',
      author='Adam Barson and Daniel Berenberg',
      author_email='abarson@uvm.edu',
      license='UVM',
      packages=['basic_utils'],
      install_requires=[
          'Pillow',
          'opencv-python'
        ],
      zip_safe=False)
