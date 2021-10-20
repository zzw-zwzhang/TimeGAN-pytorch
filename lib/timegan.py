"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .data import batch_generator
from utils import extract_time, random_generator, NormMinMax
from .model import Encoder, Recovery, Generator, Discriminator, Supervisor
from .metrics.discriminative_metrics import discriminative_score_metrics
from .metrics.predictive_metrics import predictive_score_metrics
from .metrics.visualization_metrics import visualization


class BaseModel():
  """ Base Model for timegan
  """

  def __init__(self, opt, ori_data):
    # Seed for deterministic behavior
    self.seed(opt.manualseed)

    # Initalize variables.
    self.opt = opt
    self.ori_data, self.min_val, self.max_val = NormMinMax(ori_data)
    self.ori_time, self.max_seq_len = extract_time(self.ori_data)
    self.data_num, _, _ = np.asarray(ori_data).shape    # 3661; 24; 6
    self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
    self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
    self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

  def seed(self, seed_value):
    """ Seed

    Arguments:
        seed_value {int} -- [description]
    """
    # Check if seed is default value
    if seed_value == -1:
      return

    # Otherwise seed all functionality
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True

  def save_weights(self, epoch):
    """Save net weights for the current epoch.

    Args:
        epoch ([int]): Current epoch number.
    """

    weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
    if not os.path.exists(weight_dir): os.makedirs(weight_dir)

    torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
               '%s/netE.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()},
               '%s/netR.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
               '%s/netG.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
               '%s/netD.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()},
               '%s/netS.pth' % (weight_dir))


  def train_one_iter_er(self):
    """ Train the model for one epoch.
    """

    self.nete.train()
    self.netr.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)  # [128, 24, 6]
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

    # train encoder & decoder
    self.optimize_params_er()

  def train_one_iter_s(self):
    """ Train the model for one epoch.
    """

    self.nete.eval()
    self.nets.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)  # [128, 24, 6]
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

    # train superviser
    self.optimize_params_s()

  def train_one_iter_g(self):
    """ Train the model for one epoch.
    """

    self.netr.eval()
    self.nets.eval()
    self.netd.eval()
    self.netg.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)  # [128, 24, 6]
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

    # train superviser
    self.optimize_params_g()

  def train_one_iter_d(self):
    """ Train the model for one epoch.
    """
    self.nete.eval()
    self.netr.eval()
    self.nets.eval()
    self.netg.eval()
    self.netd.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)  # [128, 24, 6]
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

    # train superviser
    self.optimize_params_d()


  def train(self):
    """ Train the model
    """

    for iter in range(self.opt.iteration):
      # Train for one iter
      self.train_one_iter_er()

      print('Encoder training step: '+ str(iter) + '/' + str(self.opt.iteration))

    for iter in range(self.opt.iteration):
      # Train for one iter
      self.train_one_iter_s()

      print('Superviser training step: '+ str(iter) + '/' + str(self.opt.iteration))

    for iter in range(self.opt.iteration):
      # Train for one iter
      for kk in range(2):
        self.train_one_iter_g()
        self.train_one_iter_er()

      self.train_one_iter_d()

      print('Superviser training step: '+ str(iter) + '/' + str(self.opt.iteration))


    self.generated_data = self.generation()
    print('Finish Synthetic Data Generation')

    self.evaluation()


  def evaluation(self):
    ## Performance metrics
    # Output initialization
    metric_results = dict()

    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(self.opt.metric_iteration):
      temp_disc = discriminative_score_metrics(self.ori_data, self.generated_data)
      discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = list()
    for tt in range(self.opt.metric_iteration):
      temp_pred = predictive_score_metrics(self.ori_data, self.generated_data)
      predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(self.ori_data, self.generated_data, 'pca')
    visualization(self.ori_data, self.generated_data, 'tsne')

    ## Print discriminative and predictive scores
    print(metric_results)


  def generation(self):
    ## Synthetic data generation
    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)
    self.E_hat = self.netg(self.Z)    # [?, 24, 24]
    self.H_hat = self.nets(self.E_hat)  # [?, 24, 24]
    generated_data_curr = self.netr(self.H_hat)  # [?, 24, 24]

    generated_data = list()

    for i in range(self.data_num):
      temp = generated_data_curr[i, :self.ori_time[i], :]
      generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * self.max_val
    generated_data = generated_data + self.min_val

    return generated_data



class TimeGAN(BaseModel):
    """TimeGAN Class
    """

    @property
    def name(self):
      return 'TimeGAN'

    def __init__(self, opt, ori_data):
      super(TimeGAN, self).__init__(opt, ori_data)

      # -- Misc attributes
      self.epoch = 0
      self.times = []
      self.total_steps = 0

      # Create and initialize networks.
      self.nete = Encoder(self.opt).to(self.device)
      self.netr = Recovery(self.opt).to(self.device)
      self.netg = Generator(self.opt).to(self.device)
      self.netd = Discriminator(self.opt).to(self.device)
      self.nets = Supervisor(self.opt).to(self.device)

      if self.opt.resume != '':
        print("\nLoading pre-trained networks.")
        self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
        self.nete.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netE.pth'))['state_dict'])
        self.netr.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netR.pth'))['state_dict'])
        self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
        self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
        self.nets.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netS.pth'))['state_dict'])
        print("\tDone.\n")

      # loss
      self.l_mse = nn.MSELoss()
      self.l_r = nn.L1Loss()
      self.l_bce = nn.BCELoss()

      # Setup optimizer
      if self.opt.isTrain:
        self.netg.train()
        self.netd.train()
        self.optimizer_e = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_r = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_s = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))


    def forward_e(self):
      """ Forward propagate through netE
      """
      self.H = self.nete(self.X)     # [?, 24, 24]

    def forward_er(self):
      """ Forward propagate through netR
      """
      self.H = self.nete(self.X)         # [?, 24, 24]
      self.X_tilde = self.netr(self.H)   # [?, 24, 6]

    def forward_g(self):
      """ Forward propagate through netG
      """
      self.E_hat = self.netg(self.Z)    # [?, 24, 24]

    def forward_dg(self):
      """ Forward propagate through netD
      """
      self.Y_fake = self.netd(self.H_hat)   # [?, 24, 1]
      self.Y_fake_e = self.netd(self.E_hat)  # [?, 24, 1]

    def forward_rg(self):
      """ Forward propagate through netG
      """
      self.X_hat = self.netr(self.H_hat)    # [?, 24, 24]

    def forward_s(self):
      """ Forward propagate through netS
      """
      self.H_supervise = self.nets(self.H)  # [?, 24, 24]

    def forward_sg(self):
      """ Forward propagate through netS
      """
      self.H_hat = self.nets(self.E_hat)  # [?, 24, 24]

    def forward_d(self):
      """ Forward propagate through netD
      """
      self.Y_real = self.netd(self.H)
      self.Y_fake = self.netd(self.H_hat)   # [?, 24, 1]
      self.Y_fake_e = self.netd(self.E_hat)  # [?, 24, 1]


    def backward_er(self):
      """ Backpropagate through netE
      """
      self.err_er = self.l_mse(self.X, self.X_tilde)    # [128, 24]
      self.err_er.backward(retain_graph=True)

      print("Loss: ", self.err_er)

    def backward_g(self):
      """ Backpropagate through netG
      """
      self.err_g_U = self.l_bce(torch.ones_like(self.Y_fake), self.Y_fake)
      self.err_g_U_e = self.l_bce(torch.ones_like(self.Y_fake_e), self.Y_fake_e)
      self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(self.X_hat,[0])[1] + 1e-6) - torch.sqrt(torch.std(self.X,[0])[1] + 1e-6)))   # |a^2 - b^2|
      self.err_g_V2 = torch.mean(torch.abs((torch.mean(self.X_hat,[0])[0]) - (torch.mean(self.X,[0])[0])))  # |a - b|
      self.err_g = self.err_g_U + \
                   self.err_g_U_e * self.opt.w_gamma + \
                   self.err_g_V1 * self.opt.w_g + \
                   self.err_g_V2 * self.opt.w_g + \
      self.err_g.backward(retain_graph=True)

    def backward_s(self):
      """ Backpropagate through netS
      """
      self.err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:])
      self.err_s.backward(retain_graph=True)

    def backward_d(self):
      """ Backpropagate through netD
      """
      self.err_d_real = self.l_bce(torch.ones_like(self.Y_real), self.Y_real)
      self.err_d_fake = self.l_bce(torch.ones_like(self.Y_fake), self.Y_fake)
      self.err_d_fake_e = self.l_bce(torch.ones_like(self.Y_fake_e), self.Y_fake_e)
      self.err_d = self.err_d_real + \
                   self.err_d_fake + \
                   self.err_d_fake_e * self.opt.w_gamma
      if self.err_d > 0.15:
        self.err_d.backward(retain_graph=True)


    def optimize_params_er(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_er()

      # Backward-pass
      # nete & netr
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_s(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_s()

      # Backward-pass
      # nets
      self.optimizer_s.zero_grad()
      self.backward_s()
      self.optimizer_s.step()

    def optimize_params_g(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_g()
      self.forward_sg()
      self.forward_rg()
      self.forward_dg()

      # Backward-pass
      # nets
      self.optimizer_g.zero_grad()
      self.backward_g()
      self.optimizer_g.step()

    def optimize_params_d(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_g()
      self.forward_sg()
      self.forward_dg()

      # Backward-pass
      # nets
      self.optimizer_d.zero_grad()
      self.backward_d()
      self.optimizer_d.step()