#!/usr/bin/env python

import setuptools

setuptools.setup(name='tvb_epilepsy',
                 version='0.1',
                 description='TVB epilepsy applications',
                 author='Denis Perdikis, Paula Popa, Lia Domide, Marmaduke Woodman',
                 author_email='<insert here>',
                 license="GPL",
                 url='https://github.com/the-virtual-brain/tvb_epilepsy-epilepsy',
                 packages=['tvb_epilepsy'], requires=["numpy", "sympy", "h5py", "tvb", "mpldatacursor"]
                 )
