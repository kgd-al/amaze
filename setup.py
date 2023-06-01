from setuptools import setup, find_packages

setup(name='amaze',
      version='0.0.0',
      author='Kevin Godin-Dubois',
      author_email='k.j.m.godin-dubois@vu.nl',
      packages=find_packages(where='src'),
      requires=[
            "abrain",
      ],
      scripts=[

      ])



