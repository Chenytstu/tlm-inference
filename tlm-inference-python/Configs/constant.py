from FixedPoint.FixedPoint import FXfamily

cintbit = 16          # the range of values in the task (plaintext)
cshareintbit = 48     # the range of additive shares
cfracbit = 64         # the precision of secret and additive shares
efracbit = 64         # the precision of multiplicative shares

dim = 16  # the dimension of matrix, it should be determined by the actual task, here we only set it for testing

famcfrac = FXfamily(cfracbit)
famefrac = FXfamily(efracbit)