'''
Julian Mazzier and Cameron Scolari

ad_engine.py
Advertisement Selection Engine that employs a Decision Network
to Maximize Expected Utility associated with different Decision
variables in a stochastic reasoning environment.
'''
import math
import itertools
import unittest
import numpy as np
import pandas as pd
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

class AdEngine:

    def __init__(self, data: "pd.DataFrame", structure: list[tuple[str, str]], dec_vars: list[str], util_map: dict[str, dict[int, int]]):
        """
        Responsible for initializing the Decision Network of the
        AdEngine by taking in the dataset, structure of network,
        any decision variables, and a map of utilities
        
        Parameters:
            data (pd.DataFrame):
                Pandas data frame containing all data on which the decision
                network's chance-node parameters are to be learned
            structure (list[tuple[str, str]]):
                The Bayesian Network's structure, a list of tuples denoting
                the edge directions where each tuple is (parent, child)
            dec_vars (list[str]):
                list of string names of variables to he agent. Example:
                ["Ad1", "Ad2"]be
                considered decision variables for t
            util_map (dict[str, dict[int, int]]):
                Discrete, tabular, utility map whose keys
                are variables in network that are parents of a utility node, and
                values are dictionaries mapping that variable's values to a utility
                score, for example:
                  {
                    "X": {0: 20, 1: -10}
                  }
                represents a utility node with single parent X whose value of 0
                has a utility score of 20, and value 1 has a utility score of -10
        """
        self.data: "pd.DataFrame" = data
        self.new_model: BayesianNetwork = BayesianNetwork()
        self.new_model.add_nodes_from(list(data.columns))
        self.new_model.add_edges_from([edge for edge in structure])
        self.new_model.fit(data)
        self.dec_vars: list[str] = dec_vars
        self.util_map: dict[str, dict[int, int]] = util_map
        self.inference: CausalInference = CausalInference(self.new_model)

    def meu(self, evidence: dict[str, int]) -> tuple[dict[str, int], float]:
        """
        Computes the Maximum Expected Utility (MEU) defined as the choice of
        decision variable values that maximize expected utility of any evaluated
        chance nodes given in the agent's utility map.
        
        Parameters:
            evidence (dict[str, int]):
                dict mapping network variables to their observed values, 
                of the format: {"Obs1": val1, "Obs2": val2, ...}
        
        Returns: 
            tuple[dict[str, int], float]:
                A 2-tuple of the format (a*, MEU) where:
                [0] is a dictionary mapping decision variables to their MEU states
                [1] is the MEU value (a float) of that decision combo
        """

        a_star: dict[str, int] = {"": 0}
        max_utility: float = float("-inf")

        decision_values: dict[str, list[int]] = {var: self.new_model.get_cpds(var).state_names[var] for var in self.dec_vars}
        all_assignments: list[tuple[int, ...]] = list(itertools.product(*decision_values.values()))

        for assignment in all_assignments:
            decision_assignment: dict[str, int] = dict(zip(self.dec_vars, assignment))

            do_interventions: dict[str, int] = {key: value for key, value in decision_assignment.items() if key not in evidence}
             
            query_result = self.inference.query(
                list(self.util_map.keys()),
                evidence=evidence,
                do=do_interventions,
                show_progress=False
            )

            expected_utility: float = 0.0

            for util_parent, util_mapping in self.util_map.items():
                for value, utility in util_mapping.items():

                    if util_parent in query_result.variables:
                        prob: float = query_result.get_value(**{util_parent: value})
                    else:
                        prob = 0.0

                    expected_utility += prob * utility

            if expected_utility > max_utility:
                max_utility = expected_utility
                a_star = decision_assignment

        return a_star, max_utility

    def vpi(self, potential_evidence: str, observed_evidence: dict[str, int]) -> float:
        """
        Given some observed demographic "evidence" about a potential
        consumer, returns the Value of Perfect Information (VPI)
        that would be received on the given "potential" evidence about
        that consumer.

        Parameters:
            potential_evidence (str):
                string representing the variable name of the variable
                under consideration for potentially being obtained
            observed_evidence (tuple[dict[str, int], float]):
                dict mapping network variables
                to their observed values, of the format:
                {"Obs1": val1, "Obs2": val2, ...}

        Returns:
            float:
                float value indicating the VPI(potential | observed)
        """

        observed_meu: float = self.meu(observed_evidence)[1]
        possible_values: list[int] = self.new_model.get_cpds(potential_evidence).state_names[potential_evidence]

        prob_distribution = self.inference.query(
            variables=[potential_evidence],
            evidence=observed_evidence,
            show_progress=False
        )

        expected_meu: float = 0.0

        for value in possible_values:
            extended_evidence: dict[str, int] = observed_evidence.copy()

            # Skip if potential evidence is already in observed evidence
            if potential_evidence in observed_evidence:
                return 0.0

            extended_evidence[potential_evidence] = value
            conditional_meu: float = self.meu(extended_evidence)[1]
            prob_value: float = prob_distribution.get_value(**{potential_evidence: value})
            expected_meu += prob_value * conditional_meu

        vpi_value: float = max(0, expected_meu - observed_meu)

        return vpi_value
    
    
    def most_likely_consumer(self, evidence: dict[str, int]) -> dict[str, int]:
        """
        Given some known traits about a particular consumer, makes the best guess
        of the values of any remaining hidden variables and returns the completed
        data point as a dictionary of variables mapped to their most likely values.
        (Observed evidence will always have the same values in the output).
        
        Parameters:
            evidence (dict[str, int]):
                dict mapping network variables 
                to their observed values, of the format: 
                {"Obs1": val1, "Obs2": val2, ...}
        
        Returns:
            dict[str, int]:
                The most likely values of all variables given what's already
                known about the consumer.
        """
        
        hidden_variables: list[str] = []

        for variable in list(self.data.columns):
            if variable in self.dec_vars:
                continue
            if variable in evidence.keys():
                continue
            hidden_variables.append(variable)

        possible_values: dict[str, int] = VariableElimination(self.new_model).map_query(variables=hidden_variables, evidence=evidence, show_progress=False)
        possible_values.update(evidence)

        return possible_values
    