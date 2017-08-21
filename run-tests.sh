#!/usr/bin/env bash

# ignore extern and scripts under dev
ignores=""
for ign in examples; do ignores="--ignore=$ign $ignores"; done

# maybe do coverage
cov=""
if [[ $COV == "yes" ]]; then cov="--cov=tvb_epilepsy"; fi

# run 'em
py.test --cov-config .coveragerc $cov $ignores