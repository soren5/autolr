import copy
import hashlib
import json
from pathlib import Path


class DeterministicFitnessEvaluator:
    """Small evaluator for SGE mechanics tests without TensorFlow training."""

    def __init__(self):
        self.calls = []

    def evaluate(self, phen, parameters):
        from utils.smart_phenotype import smart_phenotype

        key = smart_phenotype(phen)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        self.calls.append(key)
        return -score, {"source": "deterministic-test-evaluator"}

    def init_net(self, parameters):
        pass

    def init_data(self, parameters):
        pass

    def init_evaluation(self, parameters):
        pass


def population_projection(population):
    """Strip volatile/debug fields before comparing deterministic runs."""

    projected = []
    for indiv in population:
        item = copy.deepcopy(indiv)
        if "other_info" in item:
            item["other_info"].pop("duration", None)
        projected.append(item)
    return projected


def population_signature(population):
    """Return stable behavior fields for resume/reproducibility checks."""

    return [
        {
            "id": indiv.get("id"),
            "fitness": indiv.get("fitness"),
            "smart_phenotype": indiv.get("smart_phenotype"),
            "key": indiv.get("key"),
            "parent": indiv.get("parent"),
        }
        for indiv in population
    ]


def run_dump_dir(parameters):
    return (
        Path(parameters["DUMPS_DIR"])
        / parameters["EXPERIMENT_NAME"]
        / f"run_{parameters['RUN']}"
    )


def load_archive(parameters, generation):
    archive_path = run_dump_dir(parameters) / f"z-archive_{generation}.json"
    with archive_path.open("r") as archive_file:
        return json.load(archive_file)


def load_iteration(parameters, generation):
    iteration_path = run_dump_dir(parameters) / f"iteration_{generation}.json"
    with iteration_path.open("r") as iteration_file:
        return json.load(iteration_file)
