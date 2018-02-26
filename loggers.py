# -*- coding: utf-8 -*-

from utils import setup_logger
from settings import run_folder

# Set all LOGGER_DISABLE to True to disable logging
# WARNING: the mcts log file gets big quite quickly


LOGGER_DISABLE = {
    'main': False,
    'memory': False,
    'tourney': False,
    'mcts': False,
    'model': False
}


logger_main = setup_logger('logger_main', run_folder + 'logs/logger_main.log')
logger_main.disable = LOGGER_DISABLE['main']


logger_memory = setup_logger('logger_memory', run_folder + 'logs/logger_memory.log')
logger_memory.disabled = LOGGER_DISABLE['memory']


logger_tourney = setup_logger('logger_tourney', run_folder + 'logs/logger_tourney.log')
logger_tourney.disabled = LOGGER_DISABLE['tourney']


logger_mcts = setup_logger('logger_mcts', run_folder + 'logs/logger_mcts.log')
logger_mcts.disable = LOGGER_DISABLE['mcts']


logger_model = setup_logger('logger_model', run_folder + 'logs/logger_model.log')
logger_model.disabled = LOGGER_DISABLE['model']