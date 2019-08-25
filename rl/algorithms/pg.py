# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import functools
from abc import abstractmethod
import numpy as np
from rl.algorithms.algorithm import Algorithm, PolicyAgent
from rl.algorithms.utils import get_learner, get_optimizer, get_pred_oracle
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.rl_oracles import ValueBasedPolicyGradient
from rl import online_learners as ol
from rl.policies import Policy
from rl.core.utils.misc_utils import timed
from rl.core.utils import logz
from rl.core.online_learners.base_algorithms import TrustRegionSecondOrderUpdate


class AbstractPolicyGradient(Algorithm):
    """ Basic policy gradient method. """

    def __init__(self, policy, vfn,
                 with_dist_fun=False,
                 optimizer_kwargs=None,
                 lr=1e-3,
                 horizon=None, gamma=1.0, delta=None, lambd=0.99,
                 max_n_batches=2,
                 n_warm_up_itrs=None,
                 n_pretrain_itrs=1):

        assert isinstance(policy, Policy)
        self.vfn = vfn
        self.policy = policy

        # Create oracle.
        self.ae = ValueBasedAE(policy, vfn, gamma=gamma, delta=delta, lambd=lambd,
                               horizon=horizon, use_is='one', max_n_batches=max_n_batches)
        self.oracle = ValueBasedPolicyGradient(policy, self.ae)

        # Create online learner.
        scheduler = ol.scheduler.PowerScheduler(lr)
        self._optimizer = get_optimizer(policy=policy, scheduler=scheduler, **optimizer_kwargs)

        # Misc.
        self._with_dist_fun = with_dist_fun  # whether use also the loss function in trpo
        self._n_pretrain_itrs = n_pretrain_itrs
        if n_warm_up_itrs is None:
            n_warm_up_itrs = float('Inf')
        self._n_warm_up_itrs = n_warm_up_itrs
        self._itr = 0

    @property
    @abstractmethod
    def learner(self):
        pass

    @abstractmethod
    def update_learner(self, g):
        pass

    def get_policy(self):
        return self.policy

    def agent(self, mode):
        return PolicyAgent(self.policy)

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_itrs):
                ros, _ = gen_ro(self.agent('behavior'))
                ro = self.merge(ros)
                self.oracle.update(ro, self.policy)
                self.policy.update(xs=ro['obs_short'])

    def update(self, ros, agents):
        # Aggregate data
        ro = self.merge(ros)

        # Update input normalizer for whitening
        if self._itr < self._n_warm_up_itrs:
            self.policy.update(xs=ro['obs_short'])

        with timed('Update oracle'):
            _, ev0, ev1 = self.oracle.update(ro, self.policy)

        with timed('Compute policy gradient'):
            g = self.oracle.grad(self.policy.variable)

        with timed('Policy update'):
            self.update_learner(g)
            self.policy.variable = self.learner.x

        # log
        logz.log_tabular('stepsize', self.learner.stepsize)
        if hasattr(self.policy, 'lstd'):
            logz.log_tabular('std', np.mean(np.exp(2.*self.policy.lstd)))
        logz.log_tabular('g_norm', np.linalg.norm(g))
        logz.log_tabular('ExplainVarianceBefore(AE)', ev0)
        logz.log_tabular('ExplainVarianceAfter(AE)', ev1)

        self._itr += 1

    @staticmethod
    def merge(ros):
        """ Merge a list of Dataset instances. """
        return functools.reduce(lambda x, y: x+y, ros)


class PolicyGradient(AbstractPolicyGradient):
    def __init__(self, policy, vfn, **kwargs):
        super().__init__(policy, vfn, **kwargs)
        self._learner = get_learner(optimizer=self._optimizer, policy=policy)

    @property
    def learner(self):
        return self._learner

    def update_learner(self, g):
        if isinstance(self.learner, ol.FisherOnlineOptimizer):
            if isinstance(self._optimizer, TrustRegionSecondOrderUpdate) and self._with_dist_fun:
                print('trpo with dist fun')
                self.learner.update(g, ro=ro, policy=self.policy, loss_fun=self.oracle.fun)
            else:
                self.learner.update(g, ro=ro, policy=self.policy)
        else:
            self.learner.update(g)


class PiccoloPolicyGradient(AbstractPolicyGradient):
    def __init__(self, policy, vfn, pred_oracle_kwargs=None, **kwargs):
        assert pred_oracle_kwargs is not None
        super().__init__(policy, vfn, **kwargs)
        self.pred_oracle = get_pred_oracle(base_oracle=self.oracle, **pred_oracle_kwargs)
        self._learner = get_learner(optimizer=self._optimizer, policy=policy,
                                    pred_oracle=self.pred_oracle)

    @property
    def learner(self):
        return self._learner

    def update_learner(self, g):
        self.learner.update(g)
        # Update prediction oracle if necessary.
        # self.pred_oracle.update(x=self.policy.variable, ro=ro, policy=self.policy)
        self.pred_oracle.update(x=self.policy.variable)
