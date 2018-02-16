# TVB Epilepsy

This project contains prototype code for epilepsy applications of TVB.

Project architecture
====================

This module contains the following packages:
- base

    This package is independent and used by the other packages in tvb-epilepsy. It is holding sub-packages that define models, configurations, logger, computations, symbolic computations etc.
- io

    The I/O functionality is kept here. This package provides readers and writers for various file types, like: h5, csv, rdump, tvb_data format.
    It is mostly used to serialize models from base package in files. It is dependent on base package.
- plot

    Besides keeping data into files, tvb-epilepsy provides plotting functionality under this package.
    There are plot methods for: hypothesis, connectivity, simulation results, epileptor model, fitting results etc.
    It is dependent only on the base package.
- service

    This package provides all the logic of tvb_epilepsy. It contains builders, factories and services.
    Builders can be used to generate objects based on some conditions. There are builders like: hypothesis, model configuration, simulator.
    Factories are used to create instances based on the wanted type (usually string or enum). For example, there are factories for epileptor model, probability distribution.
    Services can be split in two big parts: services that do forward computations (Simulator, LSA, PSE) and services that do backwards computations (Model inversion/fitting).
    These are usually dependent on the base package because they are working with models, logger and computations. In some cases, there are also dependencies on the io package.
- top

    Here there are 2 sub-packages: examples and scripts.
    Inside examples there are main files with different flows that can be used with tvb-epilepsy. As an example, main_vep exemplifies steps for: reading an hypothesis -> generating a model configuration based on it -> configuring a Simulator -> launching a simulation -> plotting the simulation results -> serializing models to h5 files.
    The scrips sub-package contains some helper functions that are used within the main files.
    The package depends on all of the above, as here is where the flow is chosen.
- tests

    Inside this package there are various unit tests implemented for verifying the correctness of the code. They can also be used for tvb-epilepsy guidance, as they are smaller than examples package.


[![Build Status](https://travis-ci.org/the-virtual-brain/tvb-epilepsy.svg?branch=review)](https://travis-ci.org/the-virtual-brain/tvb-epilepsy) [![Coverage Status](https://coveralls.io/repos/github/the-virtual-brain/tvb-epilepsy/badge.svg)](https://coveralls.io/github/the-virtual-brain/tvb-epilepsy)
