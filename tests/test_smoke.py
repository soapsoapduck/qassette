def test_imports():
    import qassette  # noqa: F401
    import qiskit  # noqa: F401
    import qiskit_aer  # noqa: F401


def test_version():
    import qassette

    assert qassette.__version__
