"""
The TileLang LSTM: a drop-in LSTM whose recurrent scan runs a tuned TileLang kernel.
"""

from caiman_asr_train.tilelstm.lstm import TileLangLSTM

__all__ = ["TileLangLSTM"]
