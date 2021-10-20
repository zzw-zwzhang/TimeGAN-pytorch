"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

train.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from options import Options
from lib.data import load_data
from lib.timegan import TimeGAN


def train():
    """ Training
    """

    # ARGUMENTS
    opt = Options().parse()

    # LOAD DATA
    ori_data = load_data(opt)

    # LOAD MODEL
    model = TimeGAN(opt, ori_data)

    # TRAIN MODEL
    model.train()

if __name__ == '__main__':
    train()