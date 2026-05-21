from setuptools import setup, find_packages

setup(
    name="visual_slam",
    version="1.0.0",
    description="Visual SLAM with loop closure, GTSAM pose graph optimization, and ASIFT features",
    author="Hitang",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22",
        "opencv-python>=4.5",
        "scikit-learn>=1.0",
        "scipy>=1.7",
        "matplotlib>=3.5",
    ],
    extras_require={
        "gtsam": ["gtsam>=4.0"],
    },
)
