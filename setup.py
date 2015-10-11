try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'filters',
    'version': '0.1.2',
    'packages': ['filters'],
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'nose'],
    'author': 'Evan M. Davis',
    'author_email': 'emd@mit.edu',
    'url': '',
    'description': 'Python tools for the filtering of digital signals.'
}

setup(**config)
