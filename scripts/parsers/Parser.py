from abc import ABCMeta, abstractmethod

"""Base class for parser, need be implemented by each benchmark"""
class Parser(metaclass=ABCMeta):


    @abstractmethod
    def parseErr(self, errString): raise NotImplementedError()

    @abstractmethod
    def relativeErrorParser(self, errList): raise NotImplementedError()

    @abstractmethod
    def header(self): raise NotImplementedError()



