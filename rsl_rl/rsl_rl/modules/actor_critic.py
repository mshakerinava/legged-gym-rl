# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

#--- mehran ---#
def g_output(x):
    return torch.cat((x[..., -9:-6], x[..., -12:-9], x[..., -3:], x[..., -6:-3]), dim=-1)


def g_input(x):
    return torch.cat((-x[..., 0:1], x[..., 1:3], -x[..., 3:4], x[..., 4:6], g_output(x[..., -3 * 12: -2 * 12]), g_output(x[..., -2 * 12: -12]), g_output(x[..., -12:])), dim=-1)


def reflect(x):
    return x[..., ::-1]


class LinearInActor(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearInActor, self).__init__()
        self.lin = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return 0.5 * (self.lin(x) + reflect(self.lin(g_input(x))))


class LinearMid(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearMid, self).__init__()
        self.lin = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return 0.5 * (self.lin(x) + reflect(self.lin(reflect(x))))


class LinearOutActor(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearOutActor, self).__init__()
        self.lin = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return 0.5 * (self.lin(x) + g_output(self.lin(reflect(x))))


class LinearInCritic(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearInCritic, self).__init__()
        self.lin = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return 0.5 * (self.lin(x) + reflect(self.lin(g_output(x))))

             
class LinearOutCritic(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearOutCritic, self).__init__()
        self.lin = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return 0.5 * (self.lin(x) + self.lin(reflect(x)))
#--- mehran ---#


class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(LinearInActor(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(LinearOutActor(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(LinearMid(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(LinearInCritic(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(LinearOutCritic(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(LinearMid(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
