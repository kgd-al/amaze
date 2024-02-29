""" Collection of functions to generate a user-friendly representation of
a neural network from stable baselines 3"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import graphviz
from stable_baselines3.common.policies import BaseModel, ActorCriticCnnPolicy
from torch.nn import Linear, Conv2d, ReLU, Flatten, Sequential


class _Module(ABC):
    def __init__(self): pass
    @abstractmethod
    def input(self) -> Tuple[str, int]: pass
    @abstractmethod
    def output(self) -> Tuple[str, int]: pass


class _AtomicModule(_Module):
    def __init__(self, m, n_id):
        super().__init__()
        if isinstance(m, Linear):
            self.i_arity, self.o_arity = m.in_features, m.out_features
            self.name = "Linear"
        elif isinstance(m, Conv2d):
            self.i_arity, self.o_arity = m.in_channels, m.out_channels
            self.name = f"Conv2d(k={m.kernel_size}, s={m.stride})"
        elif isinstance(m, ReLU) or isinstance(m, Flatten):
            self.i_arity, self.o_arity = -1, -1
            self.name = m.__class__.__name__
        else:
            raise ValueError(f"Unhandled module {m}")
        self.id = n_id

    def input(self) -> Tuple[str, int]: return self.id, self.i_arity
    def output(self) -> Tuple[str, int]: return self.id, self.o_arity


class _SubGraph(_Module):
    def __init__(self, children: List[Tuple[str, _Module]]):
        super().__init__()
        self.children = dict(children)
        self._input = children[0][1].input()
        self._output = children[-1][1].output()

    def input(self) -> Tuple[str, int]: return self._input
    def output(self) -> Tuple[str, int]: return self._output
    def __getitem__(self, item): return self.children[item]

    def pretty(self, indent=0):
        for name, c in self.children.items():
            print(f"{indent*2*' '}{name}")
            if isinstance(c, _SubGraph):
                c.pretty(indent+1)


def __edge(graph: graphviz.Digraph,
           lhs: Union[_Module, str], rhs: Union[_Module, str]):
    if isinstance(lhs, str):
        lhs_name, lhs_arity = lhs, -1
    else:
        lhs_name, lhs_arity = lhs.output()

    if isinstance(rhs, str):
        rhs_name, rhs_arity = rhs, -1
    else:
        rhs_name, rhs_arity = rhs.input()

    # assert lhs_arity < 0 or rhs_arity < 0 or lhs_arity == rhs_arity
    label = str(lhs_arity) if lhs_arity > 0 \
        else str(rhs_arity) if rhs_arity > 0 \
        else ""
    graph.edge(lhs_name, rhs_name, label)


def __to_dot(module, g: graphviz.Digraph, name, p_name=""):
    named_children = list(module.named_children())
    full_name = name if not p_name else f"{p_name}_{name}"
    if len(named_children):
        cluster_name = f"{name}: {module.__class__.__name__}"
        with g.subgraph(name="cluster_" + full_name,
                        graph_attr={'label': cluster_name,
                                    'labeljust': 'l'}) as sg:
            # print(f"{s_str}  label=\"{name}\"")
            children = []
            for c_name, child in named_children:
                m = __to_dot(child, sg, c_name, full_name)
                if m:
                    children.append((c_name, m))
                # if isinstance(m, AtomicModule) and m.name != "ReLU":
            if len(children) == 0:
                return None
            if len(children) > 1 and isinstance(module, Sequential):
                for (_, lhs), (_, rhs) in zip(children[:-1], children[1:]):
                    __edge(sg, lhs, rhs)
        return _SubGraph(children)
    elif not isinstance(module, Sequential):
        child = _AtomicModule(module, full_name)
        g.node(full_name, label=child.name)
        return child
    else:
        return None


def to_dot(policy):
    """ Generates a (dot) graph from the underlying pytorch elements """
    # print("="*80)
    # print("== Graph")
    # print("="*80)

    graph = graphviz.Digraph(
        format='pdf', node_attr={'shape': 'box'},
    )

    pytorch_total_params = (
        sum(p.numel() for p in policy.parameters() if p.requires_grad))

    graph.node("obs", str(policy.observation_space))

    if isinstance(policy, ActorCriticCnnPolicy):
        a_g = __to_dot(policy.action_net, graph, "c_action")
        v_g = __to_dot(policy.value_net, graph, "c_value")
        __edge(graph, a_g, "action")
        __edge(graph, v_g, "value")

        if policy.share_features_extractor:
            fe_g = __to_dot(policy.features_extractor, graph,
                            "features_extractor")
            mlp_g = __to_dot(policy.mlp_extractor, graph,
                             "mlp_extractor")
            __edge(graph, "obs", fe_g)
            __edge(graph, fe_g["cnn"], fe_g["linear"])
            if mlp_g:
                __edge(graph, fe_g, mlp_g["value_net"])
                __edge(graph, mlp_g["value_net"], a_g)
                __edge(graph, fe_g, mlp_g["policy_net"])
                __edge(graph, mlp_g["policy_net"], v_g)
            else:
                __edge(graph, fe_g, a_g)
                __edge(graph, fe_g, v_g)

        else:
            pi_fe_g = __to_dot(policy.pi_features_extractor, graph,
                               "pi_features_extractor")
            pi_mlp_g = __to_dot(policy.mlp_extractor.policy_net, graph,
                                "pi_mlp_extractor")
            __edge(graph, "obs", pi_fe_g)
            __edge(graph, pi_fe_g, pi_mlp_g)
            __edge(graph, pi_mlp_g, a_g)

            vf_fe_g = __to_dot(policy.vf_features_extractor, graph,
                               "vf_features_extractor")
            vf_mlp_g = __to_dot(policy.mlp_extractor.value_net, graph,
                                "vf_mlp_extractor")
            __edge(graph, "obs", vf_fe_g)
            __edge(graph, vf_fe_g, vf_mlp_g)
            __edge(graph, vf_mlp_g, v_g)

    # print(graph.source)

    graph.attr('graph',
               label=f"{policy.__class__.__name__}"
                     f" ({pytorch_total_params} parameters)\n\n",
               labelloc='t', labeljust='l')

    return graph
