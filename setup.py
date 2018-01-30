from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['Particle_Simulation']
    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    cmdclass={'test': PyTest},
    name='Particle_Simulation',
    version='0.1.0',
    author='Maximilian Salomon, Hannes Kneiding',
    author_email='salomon@fu-berlin.de',
    url='https://github.com/BioFreak95/Particle_Simulation',
    long_description=open('README.md').read(),
    packages=['Particle_Simulation', 'Particle_Simulation.test'],
    setup_requires=['pytest-runner',],
    install_requires=['numpy', 'numba'],
    tests_require=['pytest'],
zip_safe=False)
