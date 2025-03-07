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

from .fab_projections import projection_linf, projection_l2,\
    projection_l1

DEFAULT_EPS_DICT_BY_NORM = {'Linf': .3, 'L2': 1., 'L1': 5.0}


class FABAttack():
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044
    
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

        self.norm = norm
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.eps = eps if eps is not None else DEFAULT_EPS_DICT_BY_NORM[norm]
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.targeted = False
        self.verbose = verbose
        self.seed = seed
        self.target_class = None
        self.device = device
        self.n_target_classes = n_target_classes

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def _predict_fn(self, x):
        raise NotImplementedError("Virtual function.")

    def _get_predicted_label(self, x):
        raise NotImplementedError("Virtual function.")

    def get_diff_logits_grads_batch(self, imgs, la):
        raise NotImplementedError("Virtual function.")

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
       raise NotImplementedError("Virtual function.")

    def attack_single_run(self, x, y=None, use_rand_start=False, is_targeted=False):
        """
        :param x:             clean images
        :param y:             clean labels, if None we use the predicted labels
        :param is_targeted    True if we ise targeted version. Targeted class is assigned by `self.target_class`
        """

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone()
        #assert next(self.predict.parameters()).device == x.device

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone()
        else:
            y = y.detach().clone()
        pred = y_pred == y[:, 0]
        corr_classified = paddle.sum(pred)
        if self.verbose:
            print('Clean accuracy: {:.2%}'.format(paddle.mean(pred)))
        if paddle.sum(pred) == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        if is_targeted:
            output = self._predict_fn(x)
            la_target = paddle.argsort(output, axis=-1)[:, -self.target_class]
            la_target2 = la_target[pred].detach().clone()

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[:, 0][pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = paddle.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * paddle.ones([bs])
        res_c = paddle.zeros([x.shape[0]])
        x1 = im2.clone()
        x0 = paddle.reshape(im2.clone(), [bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                if self.norm == 'Linf':
                    t = 2 * paddle.rand(x1.shape) - 1
                    x1 = im2 + (paddle.reshape(paddle.fmin(res2,
                                          self.eps * paddle.ones(res2.shape)
                                          ),[-1, *[1]*self.ndims])
                                ) * t / (paddle.reshape(paddle.fmax(paddle.abs(paddle.reshape(t, [t.shape[0], -1])), keepdim=True),[-1, *[1]*self.ndims])) * .5
                elif self.norm == 'L2':
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / ((t ** 2)
                                         .view(t.shape[0], -1)
                                         .sum(dim=-1)
                                         .sqrt()
                                         .view(t.shape[0], *[1]*self.ndims)) * .5
                elif self.norm == 'L1':
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / (t.abs().view(t.shape[0], -1)
                                         .sum(dim=-1)
                                         .view(t.shape[0], *[1]*self.ndims)) / 2

                x1 = paddle.clip(x1, 0, 1)

            counter_iter = 0
            while counter_iter < self.n_iter:
                with paddle.no_grad():
                    if is_targeted:
                        df, dg = self.get_diff_logits_grads_batch_targeted(x1, la2, la_target2)
                    else:
                        df, dg = self.get_diff_logits_grads_batch(x1, la2)
                    # print(df.shape)
                    # print(dg.shape)
                    if self.norm == 'Linf':
                        dist1 = paddle.abs(df) / (1e-12 +
                                            paddle.sum(paddle.reshape(paddle.abs(dg), (dg.shape[0], dg.shape[1], -1)), axis=-1))
                    elif self.norm == 'L2':
                        dist1 = df.abs() / (1e-12 + (dg ** 2)
                                            .reshape(dg.shape[0], dg.shape[1], -1)
                                            .sum(dim=-1).sqrt())
                    elif self.norm == 'L1':
                        dist1 = df.abs() / (1e-12 + dg.abs().reshape(
                            [df.shape[0], df.shape[1], -1]).max(dim=2)[0])
                    else:
                        raise ValueError('norm not supported')
                    
                    ind = paddle.argmin(dist1, axis=1)
                    # ind = dist1.min(dim=1)[1]
                    dg2 = dg[u1, ind]
                    b = (- df[u1, ind] + paddle.sum(paddle.reshape((dg2 * x1), (x1.shape[0], -1)), axis=-1))
                    w = paddle.reshape(dg2, [bs, -1])

                    if self.norm == 'Linf':
                        tmpa=paddle.concat((x1.reshape([bs, -1]), x0), 0)
                        tmpb=paddle.concat((w, w), 0)
                        tmpc=paddle.concat((b, b), 0)

                        d3 = projection_linf(
                            tmpa,
                            tmpb,
                            tmpc)
                    elif self.norm == 'L2':
                        d3 = projection_l2(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    elif self.norm == 'L1':
                        d3 = projection_l1(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    d1 = paddle.reshape(d3[:bs], x1.shape)
                    d2 = paddle.reshape(d3[-bs:], x1.shape)
                    if self.norm == 'Linf':
                        a0 = paddle.reshape(paddle.max(paddle.abs(d3), axis=1, keepdim=True), (-1, *[1]*self.ndims))
                    elif self.norm == 'L2':
                        a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()\
                            .view(-1, *[1]*self.ndims)
                    elif self.norm == 'L1':
                        a0 = d3.abs().sum(dim=1, keepdim=True)\
                            .view(-1, *[1]*self.ndims)
                    a0 = paddle.fmax(a0, 1e-8 * paddle.ones(
                        a0.shape))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = paddle.fmin(paddle.fmax(a1 / (a1 + a2),
                                                paddle.zeros(a1.shape)
                                                ),
                                      self.alpha_max * paddle.ones(a1.shape)
                                      )
                    x1 = paddle.clip((x1 + self.eta * d1) * (1 - alpha) +
                          (im2 + d2 * self.eta) * alpha, 0, 1)

                    is_adv = self._get_predicted_label(x1) != la2

                    if paddle.sum(is_adv) > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        if len(ind_adv.shape)>0 and ind_adv.shape[0]>0:
                            ind_adv = self.check_shape(ind_adv)
                            if self.norm == 'Linf':
                                t = paddle.max(paddle.abs(paddle.reshape(x1[ind_adv] - im2[ind_adv],
                                    [ind_adv.shape[0], -1])), axis=1)
                            elif self.norm == 'L2':
                                t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                                    .reshape(ind_adv.shape[0], -1).sum(dim=-1).sqrt()
                            elif self.norm == 'L1':
                                t = (x1[ind_adv] - im2[ind_adv])\
                                    .abs().reshape(ind_adv.shape[0], -1).sum(dim=-1)
                            adv[ind_adv] = x1[ind_adv] * paddle.reshape((t < res2[ind_adv]).astype('float32'), [-1, *[1]*self.ndims]) + \
                                adv[ind_adv] * paddle.reshape((t >= res2[ind_adv]).astype('float32'), [-1, *[1]*self.ndims])
                            res2[ind_adv] = t * (t < res2[ind_adv]).astype('float32')\
                                + res2[ind_adv] * (t >= res2[ind_adv]).astype('float32')
                            x1[ind_adv] = im2[ind_adv] + (
                                x1[ind_adv] - im2[ind_adv]) * self.beta

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(ind_succ.float().sum(), corr_classified) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        res_c[pred] = res2 * ind_succ.astype('float32') + 1e10 * (1 - ind_succ.astype('float32'))
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        if len(ind_succ.shape)>0 and ind_succ.shape[0]>0:
            adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def perturb(self, x, y):
        if self.device is None:
            self.device = x.device
        adv = x.clone().detach()
        y=y.clone().detach()
        with paddle.no_grad():
            # acc = self._predict_fn(x).max(1)[1] == y
            y_pred = paddle.argmax(self._predict_fn(x), axis=1)
            acc = y_pred == y[:, 0]

            startt = time.time()

            # torch.random.manual_seed(self.seed)
            # torch.cuda.random.manual_seed(self.seed)

            if not self.targeted:
                for counter in range(self.n_restarts):
                    # ind_to_fool = acc.nonzero().squeeze()
                    # if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    # if ind_to_fool.numel() != 0:
                        # x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    if True:
                        x_to_fool, y_to_fool=x,y
                        adv_curr = self.attack_single_run(x_to_fool, y_to_fool, use_rand_start=(counter > 0), is_targeted=False)

                        acc_curr = paddle.max(self._predict_fn(adv_curr), axis=1) == y_to_fool[:, 0]
                        if self.norm == 'Linf':
                            res = paddle.max(paddle.reshape(paddle.abs(x_to_fool - adv_curr), (x_to_fool.shape[0], -1)), axis=1)
                        elif self.norm == 'L2':
                            res = ((x_to_fool - adv_curr) ** 2).reshape(x_to_fool.shape[0], -1).sum(dim=-1).sqrt()
                        elif self.norm == 'L1':
                            res = (x_to_fool - adv_curr).abs().reshape(x_to_fool.shape[0], -1).sum(-1)
                        acc_curr = paddle.fmax(acc_curr, res > self.eps)

                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        # acc[ind_to_fool[ind_curr]] = 0
                        # adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if len(ind_curr.shape)>0 and ind_curr.shape[0]>0:
                            acc[ind_curr] = 0
                            adv[ind_curr] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), self.eps, time.time() - startt))

            else:
                for target_class in range(2, self.n_target_classes + 2):
                    self.target_class = target_class
                    for counter in range(self.n_restarts):
                        # ind_to_fool = acc.nonzero().squeeze()
                        # if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                        # if ind_to_fool.numel() != 0:
                        #     x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        if True:
                            x_to_fool, y_to_fool=x.clone(), y.clone()
                            adv_curr = self.attack_single_run(x_to_fool, y_to_fool, use_rand_start=(counter > 0), is_targeted=True)

                            # acc_curr = self._predict_fn(adv_curr).max(1)[1] == y_to_fool
                            acc_curr = paddle.argmax(self._predict_fn(adv_curr), axis=1) == y_to_fool[:, 0]
                            if self.norm == 'Linf':
                                res = paddle.max(paddle.reshape(paddle.abs(x_to_fool - adv_curr), (x_to_fool.shape[0], -1)), axis=1)
                            elif self.norm == 'L2':
                                res = ((x_to_fool - adv_curr) ** 2).reshape(x_to_fool.shape[0], -1).sum(dim=-1).sqrt()
                            elif self.norm == 'L1':
                                res = (x_to_fool - adv_curr).abs().reshape(x_to_fool.shape[0], -1).sum(-1)
                            acc_curr = paddle.fmax(acc_curr.astype('int64'), (res > self.eps).astype('int64'))

                            ind_curr = (acc_curr == 0).nonzero().squeeze()
                            # acc[ind_to_fool[ind_curr]] = 0
                            # adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                            if len(ind_curr.shape)>0 and ind_curr.shape[0]>0:
                                acc[ind_curr] = 0
                                adv[ind_curr] = adv_curr[ind_curr].clone()

                            if self.verbose:
                                print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                    counter, self.target_class, acc.float().mean(), self.eps, time.time() - startt))

        return adv
