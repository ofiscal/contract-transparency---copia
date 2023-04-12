# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:29:07 2023

@author: usuario
"""

import os
from tensorboard import program

tracking_address = os.path.join(r"C:\Users\usuario\Documents\contract-transparency-copia\python3_10\modules\models\tensor_board", 'keras_export') # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")