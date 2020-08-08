from setuptools import setup

setup(
    name='WaveformML',
    version='1',
    packages=['src', 'src.utils', 'src.models', 'src.datasets', 'src.engineering', 'src.optimization'],
    url='https://github.com/BlaineHeffron/WaveformML',
    license='GPL',
    author='Blaine Heffron',
    author_email='bheffron@vols.utk.edu',
    description='Machine learning tools for multidetector waveform analysis.'
)
