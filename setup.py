from setuptools import setup, find_packages

setup(
    name='counternet',
    version='0.0.1',
    description='End-to-end training of prediction and counterfactual explanation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Hangzhi Guo',
    author_email='hangz@psu.edu',
    url='https://github.com/birkhoffg/counternet',
    license='Apache Software License 2.0',

    # Automatically find the 'counternet' package
    packages=find_packages(),

    python_requires='>=3.6',

    install_requires=[
        'pytorch_lightning>=1.3.0',
        'torch>=1.6.0',
        'matplotlib',
        'scikit-learn',
        'pandas',
        'test_tube',
        'torchmetrics',
        'tabulate'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
    ],

    keywords='XAI, explainability, interpretability, counterfactual explanation',
    zip_safe=False,
)