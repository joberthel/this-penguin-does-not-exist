from setuptools import setup, find_packages

setup(name='dataset',
      version='1.0',
      packages=find_packages(),
      description='download source images and create dataset',
      author='Johannes Berthel',
      author_email='mail@johannesberthel.de',
      license='MIT',
      install_requires=[
          'Pillow',
          'keras',
          'matplotlib'
      ],
      zip_safe=False)
