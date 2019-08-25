# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from rl.core import oracles as Or
from rl.oracles.oracle import rlOracle


class MvAvgOracle(rlOracle, Or.MvAvgOracle):
    def ro(self):
        return None


class AdversarialOracle(rlOracle, Or.AdversarialOracle):
    def ro(self):
        return None
