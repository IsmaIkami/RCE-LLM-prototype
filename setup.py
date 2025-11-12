"""
RCE-LLM Setup Configuration

Author: Ismail Sialyen
Email: is.sialyen@gmail.com
Paper: DOI 10.5281/zenodo.17360372
"""

from setuptools import setup, find_packages
import os

# Read README
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='rce-llm',
    version='1.0.0',
    author='Ismail Sialyen',
    author_email='is.sialyen@gmail.com',
    description='Relational Coherence Engine for Consistent and Energy-Efficient Language Modeling',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/IsmaIkami/RCE-LLM-prototype',
    project_urls={
        'Paper': 'https://doi.org/10.5281/zenodo.17360372',
        'Source': 'https://github.com/IsmaIkami/RCE-LLM-prototype',
        'Bug Reports': 'https://github.com/IsmaIkami/RCE-LLM-prototype/issues',
    },
    packages=find_packages(exclude=['tests*', 'examples*', 'docs*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'spacy>=3.7.0',
        'python-dateutil>=2.8.2',
        'networkx>=3.1',
        'numpy>=1.24.0',
    ],
    extras_require={
        'optimization': [
            'pulp>=2.7.0',
            'scipy>=1.11.0',
        ],
        'api': [
            'fastapi>=0.103.0',
            'uvicorn>=0.23.0',
            'pydantic>=2.3.0',
        ],
        'rag': [
            'sentence-transformers>=2.2.2',
            'faiss-cpu>=1.7.4',
        ],
        'benchmarks': [
            'pandas>=2.1.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
        ],
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
            'sphinx>=7.1.0',
            'sphinx-rtd-theme>=1.3.0',
        ],
        'all': [
            'pulp>=2.7.0',
            'scipy>=1.11.0',
            'fastapi>=0.103.0',
            'uvicorn>=0.23.0',
            'pydantic>=2.3.0',
            'sentence-transformers>=2.2.2',
            'faiss-cpu>=1.7.4',
            'pandas>=2.1.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'rce-llm=rce_llm.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'natural language processing',
        'language models',
        'coherence',
        'relational reasoning',
        'knowledge graphs',
        'semantic consistency',
        'explainable AI',
        'energy efficiency',
    ],
)
