#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 00:12:22 2022

@author: mitchellrimerman
"""

import numpy as np
import torch 
import torchvision
import matplotlib.pyplot as plt



def dxb(u):
  return u-np.roll(u,1,axis=0)

def dyb(u): 
  return u-np.roll(u,1,axis=1)

def dxf(u):
  return np.roll(u,-1,axis=0)-u

def dyf(u): 
  return np.roll(u,-1,axis=1)-u

gtown=torchvision.io.read_image('gtown.jpg')
gtown=torchvision.transforms.Resize((400,400))(gtown)
plt.imshow(gtown.permute(1, 2, 0))

gtown_gray = torchvision.transforms.Grayscale(num_output_channels=1)(gtown)
gtown_gray = torch.reshape(gtown_gray, (400,400))
plt.imshow(gtown_gray, cmap='Greys_r')

noise = torch.zeros(400,400, dtype=torch.float64)
torch.manual_seed(123)
noise = noise + (10)*torch.randn(400, 400)
gtown_gn = gtown_gray+noise
plt.imshow(gtown_gn, cmap='Greys_r')

gtown_gray_np = gtown_gray.numpy()
gtown_gn_np = gtown_gn.numpy()

def matrix_dif_normed(u):
  eps = 2**(-52)
  uxf = dxf(u)
  uyf = dyf(u)
  denom = np.sqrt(uxf*uxf+uyf*uyf+eps**2)
  p1 = uxf/denom
  p2 = uyf/denom
  return np.array([p1,p2])

def Df(u,f,mu):
  p = matrix_dif_normed(u)
  D = dxb(p[0])+dyb(p[1])
  return -D+mu*(u-f)


alpha=.01
mu= .01
f=gtown_gn_np
u=gtown_gn_np

for i in range(10):
  print(i)
  
  
  
  
  

for i in range(10000):
  print(i)
  u_next = u - alpha*Df(u,f,mu)
  nromdiff = np.linalg.norm(u_next-u,ord=2)
  u = u_next

sum(sum(((u-gtown_gray_np)**2)))
sum(sum(np.square(u - gtown_gray_np)))

plt.imshow(u, cmap='Greys_r')