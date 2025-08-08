from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'gcsfs==0.6.0',
    'pandas==0.24.2',
    'scikit-learn==0.20.4',
    'joblib==0.14.1',
    'numpy==1.18.4',
    'psutil==5.7.0',
    'memoized-property==1.0.3',
    'scipy== 1.2.2',
    'category_encoders==2.2.2']

setup(
    name='FraudDetection_model',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Fraud detection - Probability Prediction Pipeline'
)
