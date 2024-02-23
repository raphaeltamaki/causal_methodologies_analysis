import numpy as np
import pandas as pd
from typing import List


class CausalExample:
    def __init__(self, random_seed=None, sample_size=10000) -> None:

        self.rng = np.random.default_rng(random_seed)
        self.sample_size = sample_size

    def generate_effects(self):
        raise NotImplementedError

    def plot_graph(self):
        raise NotImplementedError

    def show_causal_effects(self):
        raise NotImplementedError


class BloodPressureExample(CausalExample):
    def generate_effects(self):
        took_drug = self.rng.binomial(1, 0.5, self.sample_size)
        low_blood_pressure = self.rng.binomial(1, 0.5 * (1 + 0.5 * took_drug))
        heart_attack = self.rng.binomial(
            1, 0.01 * (1 + 0.1 * took_drug - 0.5 * low_blood_pressure)
        )
        self.dag = {
            "took_drug": took_drug,
            "low_blood_pressure": low_blood_pressure,
            "heart_attack": heart_attack,
        }

        return pd.DataFrame(self.dag)


class DiscountTicket(CausalExample):
    def generate_effects(self):
        income = self.rng.normal(1000, 200, self.sample_size)
        ticket = (income < 800).astype(int) * self.rng.beta(2, 8, self.sample_size)
        spend = 200 + 0.2 * income + 200 * ticket
        self.dag = {
            "income": income,
            "ticket": ticket,
            "spend": spend,
        }
        return pd.DataFrame(self.dag)


class GenderDrugDiseaseExample(CausalExample):
    def generate_effects(self):
        is_female = self.rng.binomial(1, 0.5, self.sample_size)
        took_drug = self.rng.binomial(1, 0.1 + 0.4 * is_female)
        heart_attack = self.rng.binomial(1, 0.01 * (1 + 10 * is_female - took_drug))
        self.dag = {
            "took_drug": took_drug,
            "is_female": is_female,
            "heart_attack": heart_attack,
        }
        return pd.DataFrame(self.dag)
