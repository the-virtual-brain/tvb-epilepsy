from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.top.workflow.workflow_lsa import WorkflowLSA
from tvb_fit.tvb_epilepsy.top.workflow.workflow_simulation import WorkflowSimulate


class Workflow(WorkflowSimulate, WorkflowLSA):

    def __init__(self, config=Config(), reader=None, writer=None, plotter=None):
        super(Workflow, self).__init__(config, reader, writer, plotter)
