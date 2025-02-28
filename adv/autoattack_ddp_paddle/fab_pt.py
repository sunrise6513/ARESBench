# Copyright (c) 2019-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import torch
import paddle

from .other_utils import zero_gradients
from .fab_base import FABAttack

class FABAttack_PT(FABAttack):
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044
    
    :param predict:       forward pass function
    :param norm:          Lp-norm to minimize ('Linf', 'L2', 'L1' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           epsilon for the random restarts
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    """

    def __init__(
            self,
            predict,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=None,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            loss_fn=None,
            verbose=False,
            seed=0,
            targeted=False,
            device=None,
            n_target_classes=9):
        """ FAB-attack implementation in pytorch """

        self.predict = predict
        super().__init__(norm,
                         n_restarts,
                         n_iter,
                         eps,
                         alpha_max,
                         eta,
                         beta,
                         loss_fn,
                         verbose,
                         seed,
                         targeted,
                         device,
                         n_target_classes)

    def _predict_fn(self, x):
        return self.predict(x)

    def _get_predicted_label(self, x):
        with paddle.no_grad():
            outputs = self._predict_fn(x)
        y = paddle.argmax(outputs, axis=1)
        return y

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone().stop_gradient=False

        with paddle.enable_grad():
            y = self.predict(im)

        g2 = paddle.zeros([y.shape[-1], *imgs.size()])
        grad_mask = paddle.zeros_like(y)
        for counter in range(y.shape[-1]):
            # zero_gradients(im)
            im.clear_grad()
            grad_mask[:, counter] = 1.0
            y.backward(grad_mask, retain_graph=True)
            grad_mask[:, counter] = 0.0
            g2[counter] = im.grad.detach()

        g2 = paddle.transpose(g2, (0, 1)).detach()
        #y2 = self.predict(imgs).detach()
        y2 = y.detach()
        df = y2 - y2[paddle.arange(imgs.shape[0]), la].unsqueeze(1)
        dg = g2 - g2[paddle.arange(imgs.shape[0]), la].unsqueeze(1)
        df[paddle.arange(imgs.shape[0]), la] = 1e10

        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        u = paddle.arange(imgs.shape[0])
        im = imgs.clone()
        im.stop_gradient=False
        with paddle.enable_grad():
            y = self.predict(im)
            diffy = -(y[u, la] - y[u, la_target])
            sumdiffy = paddle.sum(diffy)

        # zero_gradients(im)
        im.clear_grad()
        sumdiffy.backward()
        graddiffy = im.grad
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)

        return df, dg
