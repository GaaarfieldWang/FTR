from algorithms.dbc import BisimAgent
from algorithms.drqv2 import DrQV2Agent
from algorithms.ftd import FTD
from algorithms.ftr_drq import FTR_DRQ
from algorithms.q2 import Q2
from algorithms.sac import SAC

algorithm = {
    'ftr_drq': FTR_DRQ,
    'sac': SAC,
    'ftd': FTD,
    'dbc': BisimAgent,
    'drqv2': DrQV2Agent,
    'q2': Q2
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
