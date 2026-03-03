"""
File Name      : classes.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 4/5/2017
Python version : 3.5
Description    : Script contains implementation of the Option and Futures classes, 
                 as well as helper methods that set/store/manipulate instance variables. 
                 This class is used in simulation.py.

"""

import numpy as np
from .constants import multipliers
from .calc import _compute_greeks, _compute_value, _compute_iv


class Option:

    def __init__(self, strike: float, tau: float, option_type: str, vol: float,
                 underlying: 'Future', payoff: str, shorted: bool, month: str,
                 direction: str = None, barrier: str = None, lots: float = 1000,
                 bullet: bool = True, knock_in: float = None,
                 knock_out: float = None, rebate: float = 0,
                 ordering: float = 1e5, settlement: str = 'futures',
                 barrier_vol: float = None, barrier_vol2: float = None,
                 dailies: list = None) -> None:
        self.month = month
        self.barrier = barrier
        self.payoff = payoff
        self.underlying = underlying
        self.bullet = bullet
        self.dailies = dailies
        self.lots = lots
        self.desc = 'option'
        self.knock_in = knock_in
        self.knock_out = knock_out
        self.knockedout = None
        self.knockedin = None
        self.option_type = option_type
        self.strike = strike
        self.dbarrier = None

        if self.barrier == 'euro':
            mult = -1 if option_type == 'call' else 1
            product = self.underlying.get_product()
            ticksize = multipliers[product][2]
            barlevel = knock_in if knock_in is not None else knock_out
            self.dbarrier = barlevel + mult * ticksize

        self.direction = direction
        self.tau = tau
        self.barrier_vol = barrier_vol
        self.barrier_vol2 = barrier_vol2
        self.vol = vol
        self.shorted = shorted
        self.price = self.get_price()
        self.ordering = ordering
        self.init_greeks()
        self.active = self.check_active()
        self.expired = False
        self.rebate = rebate
        self.product = self.get_product()
        self.settlement = settlement
        self.strike_type = 'callstrike' if self.strike >= self.underlying.get_price() else 'putstrike'
        self.partners = set()
        self.prev_dailies = None

    def __str__(self) -> str:
        string = '<<'
        string += self.product + ' ' + self.month + \
            '.' + self.underlying.get_month() + ' '

        string += str(self.strike) + ' '
        if self.barrier is None:
            string += 'Vanilla '
        elif self.barrier == 'euro':
            string += 'E'
        string += 'C' if self.option_type == 'call' else 'P'
        if self.knock_in:
            if self.direction == 'up':
                string += 'UI '
            if self.direction == 'down':
                string += 'DI '
            string += str(self.knock_in)
        if self.knock_out:
            if self.direction == 'up':
                string += 'UO'
            if self.direction == 'down':
                string += 'DO'
            string += ' ' + str(self.knock_out)
        string += ' S ' if self.shorted else ' L '
        string += str(self.underlying.get_price())
        mult = -1 if self.shorted else 1
        string += ' | lots - ' + str(int(self.lots*mult)) + ' |'
        string += ' ttm - ' + str(round(self.tau * 365)) + ' |'
        string += ' order - [c_' + str(self.ordering) + '] |'
        string += ' price - ' + str(self.price) + ' |'
        string += ' delta - ' + str(abs(self.delta / self.lots)) + ' |'
        string += ' vol - ' + str(self.vol) + ' |'
        string += ' bvol - '
        string += str(self.barrier_vol) if self.barrier_vol is not None else 'None'
        string += ' | '
        string += ' | dbarrier - %s | ' % self.dbarrier
        string += 'bvol2 - ' + str(self.barrier_vol2) if self.barrier_vol2 is not None else 'None'
        string += ' | '
        string += ' len_ttms - %s | ' % len(self.dailies) if self.dailies is not None else '0'
        string += ' | strike type: ' + str(self.strike_type) + ' '
        string += '>>'
        return string

    def set_strike(self, strike: float) -> None:
        self.strike = strike
        self.update()

    def set_partners(self, ops: set) -> None:
        self.partners = ops

    def set_ordering(self, val: float) -> None:
        self.ordering = val

    def get_ordering(self) -> float:
        return self.ordering

    def decrement_ordering(self, i: float) -> None:
        self.ordering -= i
        self.expired = self.check_expired()

    def get_op_month(self) -> str:
        return self.month

    def update_bvol(self, vol: float, vol2: float = None) -> None:
        self.barrier_vol = vol
        if vol2 is not None:
            self.barrier_vol2 = vol2

    def get_greek(self, name: str) -> float:
        if name == 'delta':
            return self.delta
        if name == 'vega':
            return self.vega
        if name == 'gamma':
            return self.gamma
        if name == 'theta':
            return self.theta

    def get_m2m(self) -> float:
        mult = -1 if self.shorted else 1
        if self.is_bullet():
            return self.lots * self.price * multipliers[self.product][-1] * mult
        else:
            return self.lots * multipliers[self.product][-1] * (self.price * len(self.get_ttms())) * mult

    def get_ttms(self) -> list:
        return self.dailies

    def set_ttms(self, lst: list) -> None:
        self.dailies = lst

    def is_bullet(self) -> bool:
        return self.bullet

    def check_active(self) -> bool:
        spot = self.underlying.get_price()
        expired = self.check_expired()
        if expired:
            return False
        if self.knockedin:
            return True
        if self.knockedout:
            if self.tau == 0:
                return False
            elif self.barrier == 'amer':
                return False
            else:
                return True

        if self.knock_in:
            active = not expired
            if self.direction == 'up':
                self.knockedin = (spot >= self.knock_in)
            if self.direction == 'down':
                self.knockedin = (spot <= self.knock_in)
            if self.barrier == 'amer' and self.knockedin:
                self.knock_in, self.knock_out, self.barrier, self.direction = None, None, None, None

        if self.knock_out:
            if self.barrier == 'amer':
                if self.direction == 'up':
                    active = spot < self.knock_out
                if self.direction == 'down':
                    active = spot > self.knock_out
                self.knockedout = not active
                if self.knockedout:
                    self.dailies = []
                    self.expired = True
            elif self.barrier == 'euro':
                active = not expired
                if self.direction == 'up':
                    self.knockedout = (spot >= self.knock_out)
                if self.direction == 'down':
                    self.knockedout = (spot <= self.knock_out)
        else:
            active = not expired
        return active

    def get_underlying(self) -> 'Future':
        return self.underlying

    def get_month(self) -> str:
        return self.underlying.get_month()

    def get_desc(self) -> str:
        return self.desc

    def _sum_greeks_over_ttms(self, vol: float, barrier_vol: float,
                              barrier_vol2: float) -> tuple:
        product = self.get_product()
        spot = self.underlying.get_price()
        ttms = [self.tau] if self.bullet else self.dailies
        d, g, t, v = 0, 0, 0, 0
        for ttm in ttms:
            delta, gamma, theta, vega = \
                _compute_greeks(self.option_type, self.strike, ttm, vol,
                                spot, 0, product, self.payoff, self.lots,
                                knock_in=self.knock_in, knock_out=self.knock_out,
                                barrier=self.barrier, direction=self.direction,
                                order=self.ordering, barrier_vol=barrier_vol,
                                barrier_vol2=barrier_vol2,
                                dbarrier=self.dbarrier)
            d += delta
            g += gamma
            t += theta
            v += vega
        return d, g, t, v

    def init_greeks(self) -> None:
        try:
            delta, gamma, theta, vega = self._sum_greeks_over_ttms(
                self.vol, self.barrier_vol, self.barrier_vol2)
        except TypeError as e:
            raise TypeError(str(e)) from e

        if self.shorted:
            delta, gamma, theta, vega = -delta, -gamma, -theta, -vega

        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega

    def update_greeks(self, vol: float = None, barrier_vol: float = None,
                      barrier_vol2: float = None) -> None:
        active = self.check_active()
        self.active = active
        if active:
            sigma = vol if vol is not None else self.vol
            b_sigma = barrier_vol if barrier_vol is not None else self.barrier_vol
            b_sigma2 = barrier_vol2 if barrier_vol2 is not None else self.barrier_vol2

            d, g, t, v = 0, 0, 0, 0
            ttms = [self.tau] if self.bullet else self.dailies
            product = self.get_product()
            spot = self.underlying.get_price()

            for tau in ttms:
                delta, gamma, theta, vega = \
                    _compute_greeks(self.option_type, self.strike, tau, sigma,
                                    spot, 0, product, self.payoff,
                                    self.lots, knock_in=self.knock_in,
                                    knock_out=self.knock_out,
                                    barrier=self.barrier, direction=self.direction,
                                    order=self.ordering, barrier_vol=b_sigma,
                                    barrier_vol2=b_sigma2,
                                    dbarrier=self.dbarrier)
                if self.shorted:
                    delta, gamma, theta, vega = -delta, -gamma, -theta, -vega
                d += delta
                g += gamma
                t += theta
                v += vega

            self.delta, self.gamma, self.theta, self.vega = d, g, t, v
            self.vol = sigma
            self.barrier_vol = b_sigma
            self.barrier_vol2 = b_sigma2
            self.price = self.compute_price()
            self.strike_type = 'callstrike' if self.strike >= spot else 'putstrike'
        else:
            self.zero_option()

    def greeks(self) -> tuple:
        self.update_greeks()
        return self.delta, self.gamma, self.theta, self.vega

    def compute_vol(self, underlying: float, price: float, strike: float,
                    tau: float, r: float) -> float:
        product = self.get_product()
        if self.barrier is None:
            return _compute_iv(underlying, price, strike, tau, r, product)

    def compute_price(self) -> float:
        ttms = [self.tau] if self.bullet else self.dailies
        s = self.underlying.get_price()
        product = self.underlying.get_product()
        val = 0
        for tau in ttms:
            val += _compute_value(self.option_type, tau, self.vol, self.strike,
                                  s, 0, self.payoff,
                                  knock_in=self.knock_in, knock_out=self.knock_out,
                                  barrier=self.barrier,
                                  d=self.direction, product=product,
                                  barrier_vol=self.barrier_vol,
                                  barrier_vol2=self.barrier_vol2,
                                  dbarrier=self.dbarrier)
        val = val/len(ttms) if val != 0 else 0
        self.price = val
        return val

    def get_price(self) -> float:
        active = self.check_active()
        self.active = active
        if self.active:
            price = self.compute_price()
            self.price = price
            return self.price
        else:
            return 0

    def update_tau(self, diff: float) -> None:
        self.tau -= diff
        if not self.bullet:
            if diff >= 0:
                self.prev_dailies = self.dailies
                self.dailies = [x - diff if x-diff > 0 else 0
                                for x in self.dailies]
            else:
                if self.prev_dailies is not None:
                    self.dailies = self.prev_dailies
                    self.prev_dailies = None

    def remove_expired_dailies(self) -> None:
        self.dailies = [x for x in self.dailies if not np.isclose(x, 0)]

    def get_product(self) -> str:
        return self.underlying.get_product()

    def exercise(self) -> bool:
        worth = self.moneyness()
        return (worth is not None) and (worth > 0)

    def _is_itm(self, spot: float) -> bool:
        if self.option_type == 'call':
            return self.strike < spot
        return self.strike > spot

    def moneyness(self) -> int:
        active = self.check_active()
        if self.knockedout:
            return -1
        self.active = active
        if not active:
            return -1
        spot = self.underlying.get_price()
        if self.strike == spot:
            return 0
        if self.knock_out is not None and self.knockedout:
            return -1
        if self.knock_in is not None and not self.knockedin:
            return -1
        return 1 if self._is_itm(spot) else -1

    def update(self) -> None:
        self.update_greeks()
        self.price = self.get_price()
        self.strike_type = 'callstrike' if self.strike >= self.underlying.get_price() else 'putstrike'

    def zero_option(self) -> None:
        self.delta, self.gamma, self.theta, self.vega = 0, 0, 0, 0
        if self.exercise():
            if self.option_type == 'call' and self.strike < self.underlying.get_price():
                self.delta = 1 if not self.shorted else -1
            if self.option_type == 'put' and self.strike > self.underlying.get_price():
                self.delta = -1 if not self.shorted else 1
            self.delta *= self.lots

    def check_expired(self) -> bool:
        if self.bullet:
            ret = np.isclose(self.tau, 0) or self.tau <= 0
        else:
            ret = not self.dailies
        self.expired = ret
        return ret

    def update_lots(self, lots: float) -> None:
        self.lots = lots
        self.update_greeks()

    def get_strike_type(self) -> str:
        return self.strike_type

    def get_vol_id(self) -> str:
        return self.get_product() + '  ' + self.get_op_month() + '.' + self.get_month()

    def get_uid(self) -> str:
        return self.underlying.get_uid()

    def get_properties(self) -> dict:
        return {'month': self.month, 'barrier': self.barrier, 'payoff': self.payoff,
                'underlying': self.underlying, 'lots': self.lots,
                'knock_in': self.knock_in, 'knock_out': self.knock_out,
                'direction': self.direction, 'strike': self.strike,
                'option_type': self.option_type, 'vol': self.vol,
                'shorted': self.shorted, 'ordering': self.ordering,
                'rebate': self.rebate, 'barrier_vol': self.barrier_vol,
                'barrier_vol2': self.barrier_vol2, 'settlement': self.settlement}


class Future:

    def __init__(self, month: str, price: float, product: str,
                 shorted: bool = None, lots: float = 1000,
                 ordering: int = None) -> None:
        self.product = product
        self.ordering = ordering
        self.lots = lots
        self.desc = 'future'
        self.month = month
        self.shorted = shorted
        if price >= 0:
            self.price = price
        else:
            raise ValueError("Price cannot be negative")
        mult = -1 if shorted else 1
        self.delta = 1 * lots * mult

    def __str__(self) -> str:
        string = self.product + ' ' + self.month + ' '
        string += str(self.price)
        string += ' S' if self.shorted else ' L'
        string += ' ' + str(int(self.lots))
        string += ' [c_' + str(self.ordering) + ']'
        return string

    def get_ordering(self) -> int:
        return self.ordering

    def set_ordering(self, i: int) -> None:
        self.ordering = i

    def decrement_ordering(self, i: int) -> None:
        self.ordering -= i

    def get_price(self) -> float:
        return self.price

    def get_desc(self) -> str:
        return self.desc

    def update_price(self, price: float) -> None:
        self.price = price

    def get_month(self) -> str:
        return self.month

    def get_lots(self) -> float:
        return self.lots

    def get_product(self) -> str:
        return self.product

    def update_lots(self, lots: float) -> None:
        self.lots = lots
        mult = -1 if self.shorted else 1
        self.delta = 1 * lots * mult

    def get_delta(self) -> float:
        return self.delta

    def get_uid(self) -> str:
        return self.product + '  ' + self.month
