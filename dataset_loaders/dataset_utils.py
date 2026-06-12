def validate_benchmark_test_size(x_test, y_test, expected_test_size=None):
    """Validate the held-out split produced by a benchmark dataset loader."""

    actual_test_size = len(x_test)
    label_test_size = len(y_test)
    if label_test_size != actual_test_size:
        raise ValueError(
            "Benchmark dataset x_test/y_test sizes differ: "
            f"{actual_test_size} != {label_test_size}"
        )

    if expected_test_size is None:
        from sge.parameters import params

        if "TEST_SIZE" not in params:
            raise ValueError(
                "TEST_SIZE must be provided before loading benchmark test data"
            )
        expected_test_size = params["TEST_SIZE"]

    if actual_test_size != expected_test_size:
        raise ValueError(
            f"TEST_SIZE={expected_test_size} does not match the loaded "
            f"held-out test size of {actual_test_size}"
        )
