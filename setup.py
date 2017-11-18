from distutils.core import setup

__version__="dev"

setup(
    name='segmentation',
    version=__version__,
    description='Dynamic convolutional recurrent neural networks for morhological segmentation',
    author='Geza Velkey',
    author_email='evelkey@gmail.com',
    url='math.bme.hu/~velkeyg',
    packages=['segmentation'])
