import abc
import numpy as np

from functools import reduce

from noise import pnoise1 as perlin


class Flow(metaclass=abc.ABCMeta):
    def __init__(self, time_sec: int):
        self._fake_time_limit = time_sec

    def __getitem__(self, item):
        return self._calc_value(item)

    def time(self):
        return self._fake_time_limit

    def channels(self):
        return 1

    @abc.abstractmethod
    def _calc_value(self, t):
        pass


class ConstFlow(Flow):
    def __init__(self, time_sec: int
                 , constant: float):
        super(self.__class__, self).__init__(time_sec)

        self._constant = constant

    def _calc_value(self, t):
        return self._constant


class SinFlow(Flow):
    def __init__(self, time_sec: int
                 , factor: float, period: float, lag: float):
        super(self.__class__, self).__init__(time_sec)

        self._factor = factor
        self._period = period
        self._lag = lag

    def _calc_value(self, t):
        return np.sin((t - self._lag) * (2 * np.pi) / self._period) * self._factor


class PerlinFlow(Flow):
    def __init__(self, time_sec: int
                 , factor: float, period: float, complexity: int, seed: int = 0, octaves: float = 10):
        super(self.__class__, self).__init__(time_sec)

        self._factor = factor
        self._period = period
        self._complexity = complexity
        self._seed = seed
        self._octaves = octaves

    def _calc_value(self, t):
        """
        noise1(x, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024, base=0.0)

        noise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0repeatx=1024, repeaty=1024, repeatz=1024, base=0.0)

        return perlin "improved" noise value for specified coordinate

        octaves -- specifies the number of passes for generating fBm noise,
        defaults to 1 (simple noise).

        persistence -- specifies the amplitude of each successive octave relative
        to the one below it. Defaults to 0.5 (each higher octave's amplitude
        is halved). Note the amplitude of the first pass is always 1.0.

        lacunarity -- specifies the frequency of each successive octave relative
        to the one below it, similar to persistence. Defaults to 2.0.

        repeatx, repeaty, repeatz -- specifies the interval along each axis when
        the noise values repeat. This can be used as the tile size for creating
        tileable textures

        base -- specifies a fixed offset for the input coordinates. Useful for
        generating different noise textures with the same repeat interval
        """

        x = t * 2 / self._period
        repeat = self._complexity
        octaves = self._octaves
        factor = 2 * self._factor

        return perlin(x, octaves=octaves, repeat=repeat, base=self._seed) * factor


class SumFlow(Flow):
    def __init__(self, flows):
        super(self.__class__, self).__init__(flows[0].time())

        assert all(flows[0].time() == flow.time() for flow in flows)
        assert all(flows[0].channels() == flow.channels() for flow in flows)

        self._flows = flows

    def _calc_value(self, t):
        return reduce(lambda x, y: x + y, [flow[t] for flow in self._flows])


class HazardFlow(Flow):
    def __init__(self, time_sec, start, finish):
        super(HazardFlow, self).__init__(time_sec)

        self._start = start
        self._finish = finish

    @abc.abstractmethod
    def _calc_value(self, t):
        pass


class HazardTrueFalse(HazardFlow):
    def __init__(self, time_sec, start, finish
                 , upper_limit):
        super(self.__class__, self).__init__(time_sec, start, finish)

        self._start = start
        self._finish = finish
        self._upper_limit = upper_limit

    def _calc_value(self, t):
        if t < self._start or t > self._finish:
            return 0
        return self._upper_limit


class HazardLinearGrow(HazardFlow):
    def __init__(self, time_sec, start, finish
                 , upper_limit):
        super(HazardLinearGrow, self).__init__(time_sec, start, finish)

        self._start = start
        self._finish = finish
        self._upper_limit = upper_limit

    def _calc_value(self, t):
        if t < self._start:
            return 0
        if t > self._finish:
            return self._upper_limit
        return np.interp(t, [self._start, self._finish], [0, self._upper_limit])


class HazardParabolicGrow(HazardFlow):
    def __init__(self, time_sec, start, finish
                 , upper_limit):
        super(HazardParabolicGrow, self).__init__(time_sec, start, finish)

        self._start = start
        self._finish = finish
        self._upper_limit = upper_limit

    def _calc_value(self, t):
        if t < self._start:
            return 0
        if t > self._finish:
            return self._upper_limit
        upper_arg = self._upper_limit ** 0.5
        return np.interp(t, [self._start, self._finish], [0, upper_arg]) ** 2


class HazardSinGrow(HazardFlow):
    def __init__(self, time_sec, start, finish
                 , upper_limit):
        super(HazardSinGrow, self).__init__(time_sec, start, finish)

        self._start = start
        self._finish = finish
        self._upper_limit = upper_limit

    def _calc_value(self, t):
        if t < self._start:
            return 0
        if t > self._finish:
            return self._upper_limit

        return 0.5 + np.sin(np.interp(t, [self._start, self._finish],
                                [np.pi * 3 / 2, np.pi * 5 / 2])) * self._upper_limit / 2


class VectorFlow(Flow):
    def __init__(self, flows):
        super(self.__class__, self).__init__(flows[0].time())

        assert all(flows[0].time() == flow.time() for flow in flows)
        assert all(flows[0].channels() == flow.channels() for flow in flows)

        self._flows = flows

    def _calc_value(self, t):
        return [flow[t] for flow in self._flows]

    def channels(self):
        return len(self._flows)
