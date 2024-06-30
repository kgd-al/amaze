import json
from random import Random
from typing import List
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .base import BaseController, InputType, OutputType, Action, State, Robot


class TabularController(BaseController):
    _simple = False
    _savable = True

    def __init__(
        self,
        robot_data: Robot.BuildData,
        epsilon,
        seed,
        actions=BaseController.discrete_actions,
    ):
        super().__init__(robot_data)
        self._actions = actions
        self._actions_ix = {a: i for i, a in enumerate(actions)}
        self._data = {}
        self.epsilon = epsilon
        self._rng = Random(seed)
        self._updates = {}
        self.init_value = 0
        self.min_value, self.max_value = None, None

    def value(self, state, action):
        return self.values(state)[self._actions_ix[action]]

    def states(self):
        return self._data.keys()

    def reset(self):
        pass

    def values(self, state: np.ndarray):
        b_state = tuple(state)
        if b_state not in self._data:
            self._data[b_state] = [self.init_value for _ in self._actions]
        return self._data[b_state]

    def __updated(self, s: State, a: Action, value):
        b_state = tuple(s)
        if b_state not in self._updates:
            self._updates[b_state] = [0 for _ in self._actions]
        self._updates[b_state][self._actions_ix[a]] += 1

        def _update(k, v, f):
            setattr(self, k, (value if not (a_ := getattr(self, k)) else f(v, a_)))

        _update("max_value", value, max)
        _update("min_value", value, min)

    def __call__(self, state: State):
        # Use epsilon-greedy policy
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.choice(self._actions)
        else:
            return self.greedy_action(state)

    def greedy_action(self, state: State):
        """Requests the best possible action (without exploration)"""
        values = self.values(state)
        a_indices = np.flatnonzero(values == np.max(values))
        return self._rng.choice([self._actions[i] for i in a_indices])

    def __repr__(self):
        return f"Table({len(self._data)} states)"

    @staticmethod
    def __signed_tuple(t):
        return "(" + ", ".join(f"{v:+g}" for v in t) + ")"

    def __pretty_format(self, table):
        return pd.DataFrame.from_dict(
            data={self.__signed_tuple(k): table[k] for k in sorted(table.keys())},
            columns=[self.__signed_tuple(a) for a in self._actions],
            orient="index",
        )

    def pretty_format(self):
        return self.__pretty_format(self._data)

    def pretty_print(self, show_updates=False):
        header = f"Table ({len(self._data)} states):"
        body = str(self.pretty_format())
        separator = "-" * max(len(line) for line in body.split("\n"))

        print("-" * len(header))
        print(header)
        print("max:", self.max_value)
        print("min:", self.min_value)
        print(separator)
        print(body)
        print(separator)
        if show_updates:
            print(self.__pretty_format(self._updates))

    def details(self) -> dict:
        return {
            "": str(self),
            "init": self.init_value,
            "min": self.min_value,
            "max": self.max_value,
        }

    def _save_to_archive(self, archive: ZipFile, *args, **kwargs) -> bool:
        def fmt(f_):
            return self.__pretty_format(f_).to_dict(orient="index")

        with archive.open("data.json", "w") as file:
            data = json.dumps(
                dict(
                    actions=[a.tuple() for a in self._actions],
                    init_val=self.init_value,
                    min_val=self.min_value,
                    max_val=self.max_value,
                    data=fmt(self._data),
                    updates=fmt(self._updates),
                    epsilon=self.epsilon,
                    seed=self._rng.randint(0, 2**32),
                )
            ).encode("utf-8")
            file.write(data)
            return True

    @classmethod
    def _load_from_archive(
        cls, archive: ZipFile, robot: Robot.BuildData, *args, **kwargs
    ) -> "TabularController":
        def parse_val(v):
            try:
                return int(v)
            except ValueError:
                return float(v)

        def parse_tuple(s):
            return tuple([parse_val(v) for v in s.replace("(", "").replace(")", "").split(",")])

        def parse_dict(d):
            return {parse_tuple(k): list(v.values()) for k, v in d.items()}

        with archive.open("data.json", "r") as file:
            dct = json.loads(file.read().decode("utf-8"))
            actions = [Action(*t) for t in dct["actions"]]
            c = TabularController(
                robot_data=robot,
                actions=actions,
                epsilon=dct["epsilon"],
                seed=dct["seed"],
            )
            c._data = parse_dict(dct["data"])
            c._updates = parse_dict(dct["updates"])
            c.init_value = dct["init_val"]
            c.min_value = dct["min_val"]
            c.max_value = dct["max_val"]
        return c

    @staticmethod
    def inputs_types() -> List[InputType]:
        return [InputType.DISCRETE]

    @staticmethod
    def outputs_types() -> List[OutputType]:
        return [OutputType.DISCRETE]

    def __bellman(self, s: State, a: Action, r, alpha, gamma, q_value):
        __v = self.values(s)
        delta = alpha * (r + gamma * q_value - self.value(s, a))
        ix = self._actions_ix[a]
        __v[ix] += delta
        self.__updated(s, a, __v[ix])

    def sarsa(self, s: State, a: Action, r, s_, a_, alpha, gamma):
        self.__bellman(s, a, r, alpha, gamma, self.value(s_, a_))

    def q_learning(self, s: State, a: Action, r, s_, _, alpha, gamma):
        self.__bellman(s, a, r, alpha, gamma, max(self.values(s_)))

    @staticmethod
    def assert_equal(lhs: "TabularController", rhs: "TabularController"):
        for (lhs_k, lhs_v), (rhs_k, rhs_v) in zip(lhs.__dict__.items(), rhs.__dict__.items()):
            assert lhs_k == rhs_k
            if lhs_k == "_rng":
                continue
            assert lhs_v == rhs_v
