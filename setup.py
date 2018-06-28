#!/usr/bin/env python

import setuptools

setuptools.setup(name='tvb_fit',
                 version='0.2',
                 description='TVB inference package and epilepsy applications',
                 author='Denis Perdikis, Paula Popa, Lia Domide, Marmaduke Woodman',
                 author_email='<insert here>',
                 license="GPL v3",
                 url='https://github.com/the-virtual-brain/tvb-epilepsy',
                 packages=['tvb_fit'],
                 requires=["h5py", "mne", "mpldatacursor", "mpmath", "numpy", "psutil",
                           "pystan", "pytest", "SALib", "sympy"]
                 )
