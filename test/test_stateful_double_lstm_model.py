from unittest import TestCase
from core import stateful_double_lstm_model


class Test(TestCase):
    def test_generate_stateful_lstm_model(self):
        model = stateful_double_lstm_model.generate_stateful_lstm_model()
        print(model)

    def test_switch_instead(self):
        dict1 = {1: '1', 2: 2}
        print(dict1.get(5, "3"))

    def test_fun(self):
        fun(1, 2, 3, 4, A='a', B='b', C='c')

    def test_func(self):
        func(1, 2, 3, kwarg1=1, kwarg2=3)


def fun(a, b, *args, **kwargs):
    print("args=", args)
    print("kwargs=", kwargs)
    print(type(args))
    print(type(kwargs))


def func(arg1, arg2, arg3, *, kwarg1, kwarg2):
    pass
