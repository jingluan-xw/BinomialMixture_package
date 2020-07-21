from setuptools import setup
from setuptools import find_packages


exec(open("binomial_mixture_model/version.py").read())
setup(
    name = "binomial_mixture_model",
    version = __version__,
    packages = find_packages(exclude=["test*", "api"]),
    setup_requires = ["pytest-runner"],
    test_require = ["pytest"],
    license = "none",
    description = "Binomial Mixture Model",
    author = "Jing Luan",
    author_email = "jingluan.xw@gmail.com"
)
