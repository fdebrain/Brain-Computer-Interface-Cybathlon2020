from abc import ABCMeta, abstractmethod


class Observable:
    def __init__(self):
        self._observers = set()

    def attach(self, observer):
        self._observers.add(observer)
        observer._observables.add(self)

    def detach(self, observer):
        self._observers.discard(observer)
        observer._observables.discard(self)


class Observer(metaclass=ABCMeta):
    def __init__(self):
        self._observables = set()

    @abstractmethod
    def update(self):
        pass
