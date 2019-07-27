from abc import abstractmethod
import numpy as np
import copy
from .performance_estimate import PerformanceEstimate as PE
from rl.core.function_approximators.policies import Policy
from rl.core.function_approximators import FunctionApproximator
from rl.core.function_approximators.supervised_learners import SupervisedLearner
from rl.core.datasets import Dataset
from rl.core.utils.misc_utils import zipsame

class AdvantageEstimator(FunctionApproximator):
    """ An abstract advantage function estimator based on replay buffer.

        The user needs to implement methods required by `FunctionApproximator`
            `update`, `predict`, `variable` (setter and getter)
        and the new methods
            `advs`, `qfns`, `vfns`
    """
    # NOTE We overload the interfaces here to work with policies and rollouts.
    # This class can no longer use as a wrapper of usual `FunctionApproximator`.
    def __init__(self, ref_policy, name='advantage_func_app',
                 max_n_rollouts=float('Inf'),  # number of samples (i.e. rollouts) to keep
                 max_n_batches=0,  # number of batches (i.e. iterations) to keep
                 **kwargs):
        # replay buffer (the user should append ro)
        self.buffer = Dataset(max_n_batches=max_n_batches, max_n_samples=max_n_rollouts)
        assert isinstance(ref_policy, Policy)
        self._ref_policy = ref_policy  # reference policy
        self._ob_shape = ref_policy.x_shape
        self._ac_shape = ref_policy.y_shape
        super().__init__([self._ob_shape, self._ac_shape], (1,), name=name, **kwargs)

    @property
    def max_n_batches(self):
        return self.buffer.max_n_batches

    @property
    def max_n_rollouts(self):
        return self.buffer.max_n_samples

    @abstractmethod
    def update(self, ro, *args, **kwargs):
        """ based on rollouts """

    @abstractmethod
    def advs(self, ro, *args, **kwargs):  # advantage function
        """ Return a list of rank-1 nd.arrays, one for each rollout. """

    @abstractmethod
    def qfns(self, ro, *args, **kwargs):  # Q function
        """ Return a list of rank-1 nd.arrays, one for each rollout. """

    @abstractmethod
    def vfns(self, ro, *args, **kwargs):  # value function
        """ Return a list of rank-1 nd.arrays, one for each rollout. """

    # _ref_policy should not be deepcopy or saved
    def __getstate__(self):
        if hasattr(super(), '__getstate__'):
            d = super().__getstate__()
        else:
            d = self.__dict__
        d = dict(d)
        del d['_ref_policy']
        return d




class ValueBasedAE(AdvantageEstimator):
    """ An estimator based on value function. """

    DELTA_MAX=0.9999

    def __init__(self, ref_policy,  # the reference policy
                 vfn,  # value function estimator (SupervisedLearner)
                 gamma,  # discount in the problem definition
                 delta,  # discount in defining value function to make learning well-behave, or to reduce variance
                 lambd,  # mixing rate of different K-step adv/Q estimates (e.g. 0 for actor-critic, 0.98 GAE)
                 horizon=None,
                 pe_lambd=1.0,  # lambda for policy evaluation in [0,1] or None
                 n_pe_updates=5,  # number of iterations in policy evaluation
                 use_is='one',  # 'one' or 'multi' or None
                 name='value_based_adv_func_app',
                 **kwargs):
        """ Create an advantage estimator wrt ref_policy. """
        self._ref_policy = ref_policy  # Policy object
        if horizon is None and delta is None and np.isclose(gamma,1.):
            delta = min(gamma, DELTA_MAX)  # to make value learning well-defined
        self.horizon = float('Inf') if horizon is None else horizon
        self._pe = PE(gamma=gamma, lambd=lambd, delta=delta) # a helper function
        # importance sampling
        assert use_is in ['one', 'multi', None]
        self.use_is = use_is
        # policy evaluation
        assert 0<=pe_lambd<=1 or pe_lambd is None
        self.pe_lambd = pe_lambd  # user-defined self.pe_lambda-weighted td error
        if np.isclose(self.pe_lambd, 1.0):
            n_pe_updates = 1
        assert n_pe_updates >= 1, 'Policy evaluation needs at least one udpate.'
        self._n_pe_updates = n_pe_updates
        assert isinstance(vfn, SupervisedLearner)
        self._vfn = vfn
        self._vfn._dataset.max_n_samples=0  # since we aggregate rollouts here
        self._vfn._dataset.max_n_batches=0
        super().__init__(ref_policy, name=name, **kwargs)

    @property
    def gamma(self):
        return self._pe.gamma

    @property
    def delta(self):
        return self._pe.delta

    @property
    def lambd(self):
        return self._pe.lambd

    def weights(self, ro, policy=None):  # importance weight
        # ro is a Dataset or list of rollouts
        policy = policy or self._ref_policy
        assert isinstance(policy, Policy)
        return [np.exp(policy.logp(rollout.obs_short, rollout.acs) - rollout.lps) for rollout in ro]

    def update(self, ro, **kwargs):
        """ Policy evaluation """
        if len(ro)>0:
            self.buffer.append(ro)  # update the replay buffer
        ro = self.buffer[None] # join all the ros in the replay buffer
        if len(ro)>0:
            print('Replay buffer: {} batches, {} rollouts, {} samples'.format(len(self.buffer), len(ro), ro.n_samples))
            w = np.concatenate(self.weights(ro)) if self.use_is else 1.0
            for i in range(self._n_pe_updates):
                v_hat = (w*np.concatenate(self.qfns(ro, self.pe_lambd))).reshape([-1, 1])  # target
                results, ev0, ev1 = self._vfn.update(ro['obs_short'], v_hat, **kwargs)
            return results, ev0, ev1
        else:
            return None, None, None

    def advs(self, ro, lambd=None, use_is=None, ref_policy=None):  # advantage function
        """ Compute adv (evaluated at ro) wrt to ref_policy.

            Note `ref_policy` argument is only considered when `self.use_is`
            is True; in this case, if `ref_policy` is None, it is wrt to
            `self._ref_policy`. Otherwise, when `self.use_is`_is is False, the
            adv is biased toward the behavior policy that collected the data.
        """
        use_is = use_is or self.use_is
        vfns = self.vfns(ro)
        if use_is is 'multi':
            ws = self.weights(ro, ref_policy)  # importance weight
            advs = [self._pe.adv(rollout.rws, vf, rollout.done, w=w, lambd=lambd)
                    for rollout, vf, w in zipsame(ro, vfns, ws)]
        else:
            advs = [self._pe.adv(rollout.rws, vf, rollout.done, w=1.0, lambd=lambd)
                    for rollout, vf in zipsame(ro, vfns)]
        return advs, vfns

    def qfns(self, ro, lambd=None, use_is=None, ref_policy=None):  # Q function
        advs, vfns = self.advs(ro, lambd=lambd, use_is=use_is, ref_policy=ref_policy)
        qfns = [adv + vfn[:-1] for adv, vfn in zip(advs, vfns)]
        return qfns

    def vfns(self, ro):  # value function
        return [np.squeeze(self._vfn.predict(rollout.obs)) for rollout in ro]

    # Required methods of FunctionApproximator
    def predict(self, xs, **kwargs):
        raise np.zeros((len(xs),1))

    @property
    def variable(self):
        return self._vfn.variable

    @variable.setter
    def variable(self, val):
        self._vfn.variable = val


class QBasedAE(ValueBasedAE):
    pass
