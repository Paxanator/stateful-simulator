import setuptools

setuptools.setup(name='stateful_simulator',
                 version='0.0.1',
                 description='Python Simulator for Stateful Models to estimate error',
                 url='https://github.com/Paxanator/stateful-simulator',
                 author='Patrick Boueri',
                 license='BSD-3',
                 packages=setuptools.find_packages(where=".", exclude=['*.tests']),
                 package_dir={'stateful_simulator': 'stateful_simulator'},
                 zip_safe=False
                 )
