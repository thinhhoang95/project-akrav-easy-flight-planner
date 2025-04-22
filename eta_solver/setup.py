from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define the Pybind11 extension module
ext_modules = [
    Pybind11Extension(
        "eta_solver",                    # Module name
        ["eta_solver.cpp"],             # Source file
        define_macros=[('VERSION_INFO', "0.1.0")],
        extra_compile_args=['-O3', '-std=c++14'],
    ),
]

# Setup configuration
tools = {
    'build_ext': build_ext,
}

setup(
    name="eta_solver",
    version="0.1.0",
    author="Thinh Hoang",
    author_email="thinh.hoangdinh@enac.fr",
    description="High-performance ETA solver with Cost Index",
    long_description="""
    A Python extension module for estimating aircraft waypoint pass times
    using a fast C++ implementation exposed via Pybind11.
    """,
    ext_modules=ext_modules,
    cmdclass=tools,
    install_requires=["pybind11>=2.6.0"],
    zip_safe=False,
)
