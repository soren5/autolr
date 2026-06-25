from types import SimpleNamespace


class FakeHistory:
    history = {"accuracy": [0.1, 0.2], "val_accuracy": [0.3, 0.4]}


class FakeModel:
    def __init__(self):
        self.weights = ["final"]
        self.evaluate_saw_weights = None

    def compile(self, **kwargs):
        self.compile_kwargs = kwargs

    def fit(self, *args, **kwargs):
        for callback in kwargs["callbacks"]:
            if isinstance(callback, FakeEarlyStopping):
                callback.best_weights = ["best-validation"]
        return FakeHistory()

    def evaluate(self, *args, **kwargs):
        self.evaluate_saw_weights = list(self.weights)
        return [0.0, 0.9]

    def set_weights(self, weights):
        self.weights = list(weights)


class FakeEarlyStopping:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.best_weights = None


class FakeCallback:
    def __init__(self, *args, **kwargs):
        pass


def test_train_model_restores_best_validation_weights_before_numpy_evaluation(monkeypatch):
    import evaluators.evaluator_utils as evaluator_utils

    fake_model = FakeModel()
    evaluator = object.__new__(evaluator_utils.Evaluator)
    evaluator.model = object()
    evaluator.dataset = SimpleNamespace(
        x_train=["train"],
        y_train=["train-labels"],
        x_val=["validation"],
        y_val=["validation-labels"],
        x_fit=["fitness"],
        y_fit=["fitness-labels"],
    )
    evaluator.batch_size = 1
    evaluator.epochs = 2
    evaluator.validation_size = 1
    evaluator.validation_metric = "val_accuracy"
    evaluator.min_delta = 0.0
    evaluator.patience = 10
    evaluator.csv_log_file = "unused.csv"

    monkeypatch.setattr(evaluator_utils.tf.keras.models, "clone_model", lambda _: fake_model)
    monkeypatch.setattr(
        evaluator_utils.keras.callbacks,
        "EarlyStopping",
        lambda **kwargs: FakeEarlyStopping(**kwargs),
    )
    monkeypatch.setattr(evaluator_utils.keras.callbacks, "TerminateOnNaN", FakeCallback)
    monkeypatch.setattr(evaluator_utils.keras.callbacks, "CSVLogger", FakeCallback)

    fitness, results = evaluator.train_model("", optimizer=object())

    assert fake_model.evaluate_saw_weights == ["best-validation"]
    assert fitness == 0.9
    assert results["test_score"] == 0.9
