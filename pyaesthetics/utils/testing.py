"""
Provides a base class for test cases in the Pyaesthetics project.

The `PyaestheticsTestCase` class provides path constants that can be used in test cases for the
Pyaesthetics project. It includes the paths to the project root, module root, test root, and fixtures root.

Classes
-------
PyaestheticsTestCase
    A base class for test cases in the Pyaesthetics project.

Example
-------
To use this module, import it and use the `PyaestheticsTestCase` class as a base class for your test cases:

    import pyaesthetics_test_case_module
    from pyaesthetics.utils import PyaestheticsTestCase

    class MyTestCase(PyaestheticsTestCase):
        def test_something(self):
            fixture_path = self.FIXTURES_ROOT / 'my_fixture.txt'
            # rest of the test code...
"""  # NOQA: E501

import pathlib


class PyaestheticsTestCase(object):
    """
    A base class for test cases in the Pyaesthetics project.

    This class provides path constants that can be used in test cases for the Pyaesthetics project.
    It includes the paths to the project root, module root, test root, and fixtures root.

    Attributes
    ----------
    PROJECT_ROOT : pathlib.Path
        The path to the project root.
    MODULE_ROOT : pathlib.Path
        The path to the module root.
    TEST_ROOT : pathlib.Path
        The path to the test root.
    FIXTURES_ROOT : pathlib.Path
        The path to the fixtures root.
    """

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    MODULE_ROOT = PROJECT_ROOT / "pyaesthetics"
    TEST_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
