from setuptools import setup

setup(name='amaze',
      version='0.0.0',
      author='Kevin Godin-Dubois',
      author_email='k.j.m.godin-dubois@vu.nl',
      # packages=find_namespace_packages(),
      # packages=['amaze'],
      # package_dir={'amaze': 'src/'},
      requires=[
            "abrain",
            "numpy", "pandas", "matplotlib",
            "humanize", "colorama",
            "PyQt5"
      ],
      scripts=[])
