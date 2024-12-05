from setuptools import setup, find_packages

setup(
    name='LiteSocNavGym',
    version='0.1.0',
    description='A lightweight Gymnasium environment for social navigation tasks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Goodluck Oguzie',
    author_email='goodluckoguzie1@gmail.com',
    url='https://github.com/goodluckoguzie/LiteSocNavGym',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gymnasium',
        'numpy',
        'opencv-python',
        'PyYAML'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
