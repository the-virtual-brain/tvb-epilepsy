from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
import numpy

def test_create():
    x0_indices = [20]
    x0_values = [0.9]

    hyp = DiseaseHypothesis(None, excitability_hypothesis={tuple(x0_indices): x0_values},
                            epileptogenicity_hypothesis={}, connectivity_hypothesis={})

    assert hyp.x0_values == x0_values
    assert hyp.x0_indices == x0_indices
    assert numpy.empty(hyp.propagation_strenghts)
