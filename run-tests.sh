#!/usr/bin/env bash

# ignore extern and scripts under dev
ignores=""
for ign in examples extern; do ignores="--ignore=$ign $ignores"; done

# maybe do coverage
cov=""
if [[ $COV == "yes" ]]; then cov="--cov=tvb_fit"; fi

# run tests
py.test --cov-config .coveragec $cov $ignores tvb_fit