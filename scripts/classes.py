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

# lots = 10
import numpy as np

multipliers = {
    'LH':  [22.046, 18.143881, 0.025, 1, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'QC': [1.2153, 10, 1, 25, 12.153],
    'SB':  [22.046, 50.802867, 0.01, 0.25, 1120],
    'CC':  [1, 10, 1, 50, 10],
    'CT':  [22.046, 22.679851, 0.01, 1, 500],
    'KC':  [22.046, 17.009888, 0.05, 2.5, 375],
    'W':   [0.3674333, 136.07911, 0.25, 10, 50],
    'S':   [0.3674333, 136.07911, 0.25, 10, 50],
    'C':   [0.393678571428571, 127.007166832986, 0.25, 10, 50],
    'BO':  [22.046, 27.215821, 0.01, 0.5, 600],
    'LC':  [22.046, 18.143881, 0.025, 1, 400],
    'LRC': [1, 10, 1, 50, 10],
    'KW':  [0.3674333, 136.07911, 0.25, 10, 50],
    'SM':  [1.1023113, 90.718447, 0.1, 5, 100],
    'COM': [1.0604, 50, 0.25, 2.5, 53.02],
    'CA': [1.0604, 50, 0.25, 1, 53.02],
    'MW':  [0.3674333, 136.07911, 0.25, 10, 50]
}


class Option:

    """        
    Attributes:

        -- Implementation - specific attributes --

        active (bool): True if option contributes greeks, false otherwise.
        barrier (str/None): three potential inputs: None, amer or euro (for vanilla, american and european barriers respectively)
        dbarrier (float/None): used for European barriers. this is the value of the secondary digital strike based on which the 
                               digital component is priced. 
        bullet (bool): True if option has bullet payoff, False if option is Daily
        desc (str): description. defaults to 'option'.
        direc (str): 'up' or 'down'. to be used for barrier options. 
        expired (bool): True if option has expired, False otherwise. 
        knockedin (bool): True if option is has a knock-in barrier that has been breached, False otherwise. 
        knockedout (bool): True if option has a knock-out barrier that has been breached, False otherwise.
        ko (float): Knock-out barrier level.
        ki (float): Knock-in barrier level. 
        partners (set): Set containing other option objects for composite actions. if this option object is contract or delta-rolled, all options
        in this set are also contract/delta rolled. 
        strike_type (str): callstrike or putstrike, depending on relationship of strike to current spot price. 

        -----------------------------------------

        -- Conventional Attributes --

        K (float): strike
        lots (float): lottage
        month (str): Option month. The first component of C H8.Z8. 
        ordering (int): C1, C2 etc. 
        payoff (str): amer/euro. currently doesn't matter, since all options are evaluated like european options. 
        price (float): option price, computed using black scholes.
        product (str): Product this option is on. 
        r (int): interest rate, defaults to 0
        rebate (float): rebate value payed when a barrier is breached. defaults to 0. 
        settlement (str): cash or future 
        shorted (bool): True if short position, False otherwise. 
        bvol (float): Barrier vol
        bvol2 (float): barrier vol of the second ticksize-differing strike. used only for euro barriers.
        char (str): call or put
        tau (float): time to maturity in years. 
        underlying (object): Underlying Futures object. 
        vol (float): strike vol
        delta (float): delta of this option (between 0 and 1)
        gamma (TYPE): gamma
        theta (TYPE): theta
        vega (TYPE): vega

        ----------------------------------------

    Notes: 
        1_ ki, ko, bullet, direc and barrier default to None and must be 
           expressly overridden if an exotic option is desired.
        2_ knockedin and knockedout should be set ONLY for options that
           are knockin and knockout options respectively. the program assumes
           this assignment is done correctly.
        3_ greeks are computed using black scholes, unmodified.

    """

    def __init__(self, strike, tau, char, vol, underlying, payoff, shorted,
                 month, direc=None, barrier=None, lots=1000, bullet=True,
                 ki=None, ko=None, rebate=0, ordering=1e5, settlement='futures', 
                 bvol=None, bvol2=None, dailies=None):
        self.month = month
        self.barrier = barrier
        self.payoff = payoff
        self.underlying = underlying
        self.bullet = bullet  # daily = list of bullets.
        self.dailies = dailies
        # get the ttm list of all constituent daily options if this option is daily.
        self.lots = lots
        self.desc = 'option'
        self.ki = ki
        self.ko = ko
        # defaults to None. Is set upon first check_active call.
        self.knockedout = None
        # defaults to None. Is set upon first check_active call.
        self.knockedin = None
        self.char = char
        self.K = strike
        self.dbarrier = None

        # set the digital barrier, if applicable.
        if self.barrier == 'euro':
            mult = -1 if char == 'call' else 1
            product = self.underlying.get_product() 
            ticksize = multipliers[product][2]
            barlevel = ki if ki is not None else ko
            # print('barlevel: ', barlevel)
            # print('mult: ', mult)
            # print('ticksize: ', ticksize)
            self.dbarrier = barlevel + mult * ticksize

        self.direc = direc
        self.tau = tau
        self.bvol = bvol
        self.bvol2 = bvol2 
        self.vol = vol
        self.r = 0
        self.shorted = shorted
        self.price = self.get_price()
        self.ordering = ordering
        self.init_greeks()
        self.active = self.check_active()
        self.expired = False  # defaults to false.
        self.rebate = rebate
        self.product = self.get_product()
        self.settlement = settlement
        self.strike_type = 'callstrike' if self.K >= self.underlying.get_price() else 'putstrike'
        self.partners = set()
        self.prev_dailies = None 

    def __str__(self):
        string = '<<'
        string += self.product + ' ' + self.month + \
            '.' + self.underlying.get_month() + ' '

        string += str(self.K) + ' '
        if self.barrier is None:
            string += 'Vanilla '
        elif self.barrier == 'euro':
            string += 'E'
        string += 'C' if self.char == 'call' else 'P'
        if self.ki:
            if self.direc == 'up':
                string += 'UI '
            if self.direc == 'down':
                string += 'DI '
            string += str(self.ki)
        if self.ko:
            if self.direc == 'up':
                string += 'UO'
            if self.direc == 'down':
                string += 'DO'
            string += ' ' + str(self.ko)
        price = self.get_price()
        string += ' S ' if self.shorted else ' L '
        string += str(self.underlying.get_price())
        string += ' | lots - ' + str(int(self.lots)) + ' |'
        string += ' ttm - ' + str(round(self.tau * 365)) + ' |'
        string += ' order - [c_' + str(self.ordering) + '] |'
        string += ' price - ' + str(price) + ' |'
        string += ' delta - ' + str(abs(self.delta / self.lots)) + ' |'
        string += ' vol - ' + str(self.vol) + ' |'
        string += ' bvol - '
        string += str(self.bvol) if self.bvol is not None else 'None' 
        string += ' | '
        string += ' | dbarrier - %s | ' % self.dbarrier  
        string += 'bvol2 - ' + str(self.bvol2) if self.bvol2 is not None else 'None' 
        string += ' | '
        string += ' len_ttms - %s | ' % len(self.dailies) if self.dailies is not None else '0'
        string += ' | strike type: ' + str(self.strike_type) + ' '
        string += '>>'
        return string

    def set_partners(self, ops):
        self.partners = ops

    def set_ordering(self, val):
        self.ordering = val

    def get_ordering(self):
        return self.ordering

    def decrement_ordering(self, i):
        self.ordering -= i
        # check expiration
        self.expired = self.check_expired()

    def get_op_month(self):
        return self.month

    def update_bvol(self, vol, vol2=None):
        self.bvol = vol
        if vol2 is not None:
            self.bvol2 = vol2 

    def get_ttms(self):
        return self.dailies

    def set_ttms(self, lst):
        self.dailies = lst 

    def is_bullet(self):
        return self.bullet

    def check_active(self):
        """Checks to see if this option object is active, i.e. if it has any value/contributes greeks. 
        Cases are as follows:
        1) Knock-in barrier options are considered always active until expiry.
        2) Knock-out options with an american barrier are considered inactive when barrier is hit.
        3) Knock-out options with a European barrier are considered always active until expiry.
        4) Vanilla options are always active until expiry.
         """
        s = self.underlying.get_price()
        expired = self.check_expired()
        # expired case
        if expired:
            return False
        # base cases: if already knocked in or knocked out, return
        # appropriately.
        if self.knockedin:
            return True
        if self.knockedout:
            # first case: check for expiry.
            if self.tau == 0:
                return False
            # second: american barrier. ko = deactivate.
            elif self.barrier == 'amer':
                return False
            # final case: Euro barrier. active till exp.
            else:
                return True

        # barrier cases
        if self.ki:
            # all knockin options contribute greeks/have value until expiry.
            active = True if not expired else False
            if self.direc == 'up':        
                self.knockedin = True if (s >= self.ki) else False
            if self.direc == 'down':
                self.knockedin = True if (s <= self.ki) else False
            # if AKI is hit, set to vanilla parameters.
            if self.barrier == 'amer':    
                if self.knockedin:
                    self.ki, self.ko, self.barrier, self.direc = None, None, None, None

        if self.ko:
            if self.barrier == 'amer':
                # american up and out
                if self.direc == 'up':
                    active = False if s >= self.ko else True    
                # american down and out
                if self.direc == 'down':
                    active = False if s <= self.ko else True
                self.knockedout = not active
                # if knocked out, remove all elements from self.dailies
                if self.knockedout:
                    self.dailies = []
                    self.expired = True

            # european knockout are active until expiry.
            elif self.barrier == 'euro':
                active = True if not expired else False
                # european up and out
                if self.direc == 'up':
                    # print('Euro Up Out Hit')
                    self.knockedout = True if (s >= self.ko) else False
                # european down and out
                if self.direc == 'down':
                    self.knockedout = True if (s <= self.ko) else False
        else:
            # vanilla case. true till expiry
            active = True if not expired else False
        return active

    def get_underlying(self):
        return self.underlying

    def get_month(self):
        return self.underlying.get_month()

    def get_desc(self):
        return self.desc

    def init_greeks(self):
        from .calc import _compute_greeks
        # initializes relevant greeks. only used once, when initializing Option
        # object.
        product = self.get_product()
        # print(product)
        s = self.underlying.get_price()
        # print(s)
        try:
            assert self.tau > 0
            delta, gamma, theta, vega = \
                _compute_greeks(self.char, self.K, self.tau, self.vol,
                                s, self.r, product, self.payoff, self.lots,
                                ki=self.ki, ko=self.ko, barrier=self.barrier,
                                direction=self.direc, order=self.ordering,
                                bvol=self.bvol, bvol2=self.bvol2, 
                                dbarrier=self.dbarrier)
        except TypeError as e:
            print('char: ', self.char)
            print('strike: ', self.K)
            print('tau: ', self.tau)
            print('vol: ', self.vol)
            print('s: ', s)
            print('r: ', self.r)
            print('product: ', product)
            print('payoff: ', self.payoff)
            print('ki: ', self.ki)
            print('ko: ', self.ko)
            print('barrier: ', self.barrier)
            print('direction: ', self.direc)
            raise TypeError(getattr(e, 'message')) from e

        if self.shorted:
            # print('shorted!')
            delta, gamma, theta, vega = -delta, -gamma, -theta, -vega

        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        # return delta, gamma, theta, vega

    def update_greeks(self, vol=None, bvol=None, bvol2=None):
        from .calc import _compute_greeks
        # method that updates greeks given new values of s, vol and tau, and subsequently updates value.
        # used in passage of time step.
        sigma, b_sigma, b_sigma2 = None, None, None
        active = self.check_active()
        self.active = active
        if active:
            sigma = vol if vol is not None else self.vol
            b_sigma = bvol if bvol is not None else self.bvol 
            b_sigma2 = bvol2 if bvol2 is not None else self.bvol2
            
            product = self.get_product()
            s = self.underlying.get_price()

            d, g, t, v = 0,0,0,0 
            ttms = [self.tau] if self.bullet else self.dailies 

            # print('b_sigma: ', b_sigma)
            # print('b_sigma2: ', b_sigma2)

            for tau in ttms:
                delta, gamma, theta, vega = \
                    _compute_greeks(self.char, self.K, tau, sigma,
                                    s, self.r, product, self.payoff,
                                    self.lots, ki=self.ki, ko=self.ko,
                                    barrier=self.barrier, direction=self.direc,
                                    order=self.ordering, bvol=b_sigma, bvol2=b_sigma2,
                                    dbarrier=self.dbarrier)
                # account for shorting
                if self.shorted:
                    delta, gamma, theta, vega = -delta, -gamma, -theta, -vega

                d += delta 
                g += gamma 
                t += theta 
                v += vega 
                # if tau == 0:
                #     # print("Zero TTM detected; printing greeks")
                #     print(delta, gamma, theta, vega)
                #     print('lots: ', self.lots)

            self.delta, self.gamma, self.theta, self.vega = d, g, t, v

            self.vol = sigma
            self.bvol = b_sigma
            self.bvol2 = b_sigma2

            self.price = self.compute_price()
            self.strike_type = 'callstrike' if self.K >= s else 'putstrike'
        else:
            self.zero_option()

    def greeks(self):
        # getter method for greeks. preserves abstraction barrier. updates just
        # in case price of underlying has changed.
        self.update_greeks()
        return self.delta, self.gamma, self.theta, self.vega

    def compute_vol(self, underlying, price, strike, tau, r):
        # computes implied vol from market price data. only holds for vanilla
        # options.
        from .calc import _compute_iv
        product = self.get_product()
        if self.barrier is None:
            return _compute_iv(underlying, price, strike, tau, r, product)

    def compute_price(self):
        from .calc import _compute_value
        # computes the value of this structure from relevant information.
        ttms = [self.tau] if self.bullet else self.dailies 
        s = self.underlying.get_price()
        product = self.underlying.get_product()
        val = 0 
        for tau in ttms:
            val += _compute_value(self.char, tau, self.vol, self.K, s, self.r,
                                  self.payoff, ki=self.ki, ko=self.ko, barrier=self.barrier,
                                  d=self.direc, product=product, bvol=self.bvol, 
                                  bvol2=self.bvol2, dbarrier=self.dbarrier)
        # handle the edge case where self.ttms is empty in the event of a daily KO
        val = val/len(ttms) if val != 0 else 0 
        self.price = val
        return val

    def get_price(self):
        active = self.check_active()
        self.active = active
        if self.active:
            price = self.compute_price()
            self.price = price
            return self.price
        else:
            return 0

    def update_tau(self, diff):
        self.tau -= diff
        if not self.bullet:
            if diff >= 0:
                self.prev_dailies = self.dailies 
                self.dailies = [x - diff if x-diff > 0 else 0 
                                for x in self.dailies] 
            else:
                # this is only ever called when reversing a timestep. 
                if self.prev_dailies is not None:
                    self.dailies = self.prev_dailies 
                    self.prev_dailies = None 
                    
    def remove_expired_dailies(self):
        self.dailies = [x for x in self.dailies if not np.isclose(x, 0)]
        
    def get_product(self):
        return self.underlying.get_product()

    def exercise(self):
        worth = self.moneyness()
        if (worth is not None) and (worth > 0):
            return True
        else:
            return False

    def moneyness(self):
        """Checks to see the 'moneyness' of the option.

        Returns:
            int: returns 1, 0, -1 for ITM, ATM and OTM options respectively, 
                 pr None if check_active() returns False 
        """
        active = self.check_active()
        # degenerate case: knocked out.
        if self.knockedout:
            return -1
        self.active = active
        if active:
            s = self.underlying.get_price()
            # vanilla option case. 
            # at the money
            if self.K == s:
                return 0
            # call 
            if self.char == 'call':
                if self.barrier is None:
                    return 1 if self.K < s else -1
                # KO barrier case. 
                if self.ko is not None:
                    if self.knockedout:
                        return -1
                    else:
                        return 1 if self.K < s else -1
                # KI case.
                if self.ki is not None:
                    if self.knockedin:
                        return 1 if self.K < s else -1
                    else:
                        return -1 

            elif self.char == 'put':
                if self.barrier is None:
                    return 1 if self.K > s else -1
                # KO barrier case. 
                if self.ko is not None:
                    if self.knockedout:
                        return -1
                    else:
                        return 1 if self.K > s else -1
                # KI case.
                if self.ki is not None:
                    if self.knockedin:
                        return 1 if self.K > s else -1
                    else:
                        return -1 
        else:
            return -1

    def update(self):
        self.update_greeks()
        self.price = self.get_price()
        self.strike_type = 'callstrike' if self.K >= self.underlying.get_price() else 'putstrike'

    def zero_option(self):
        # check to see if the option is in the money. 
        self.delta, self.gamma, self.theta, self.vega = 0, 0, 0, 0
        if self.exercise():
            if self.char == 'call' and self.K < self.underlying.get_price():
                self.delta = 1 if not self.shorted else -1 
            if self.char == 'put' and self.K > self.underlying.get_price():
                self.delta = -1 if not self.shorted else 1
            self.delta *= self.lots 
        
    def check_expired(self):
        if self.bullet:
            ret = True if (np.isclose(self.tau, 0) or self.tau <=
                           0) else False
        else:
            ret = True if not self.dailies else False
        self.expired = ret
        return ret

    def update_lots(self, lots):
        self.lots = lots
        self.update_greeks()

    def get_strike_type(self):
        return self.strike_type

    def get_vol_id(self):
        return self.get_product() + '  ' + self.get_op_month() + '.' + self.get_month()

    def get_uid(self):
        return self.underlying.get_uid()

    def get_properties(self):
        return {'month': self.month, 'barrier': self.barrier, 'payoff': self.payoff,
                'underlying': self.underlying, 'lots': self.lots, 'ki': self.ki,
                'ko': self.ko, 'direc': self.direc, 'strike': self.K, 'char': self.char,
                'vol': self.vol,  'shorted': self.shorted, 'ordering': self.ordering,
                'rebate': self.rebate, 'bvol': self.bvol, 'bvol2': self.bvol2, 'settlement': self.settlement}


class Future:

    '''
    Class representing a Future object. Instance variables are:
    1) month     :  the contract month.
    2) price     :  the quoted price of the future.
    3) desc      :  string description of the object
    4) lots      :  number of lots represented by each future contract.
    5) product   :  the commodity of this future.
    6) shorted   :  bool indicating whether this future is being shorted or long
    7) delta     :  delta contribution of this future. 1 if shorted=False, -1 otherwise. 

    Instance Methods:
        1) get_desc       : returns 'future'
        2) get_price      : returns price of the future.
        3) update_price   : updates the price based on inputted data.
        5) get_month      : returns contract month.
        6) get_lots       : returns lot size
        7) get_product    : returns the name of this contract (i.e. the commodity)

    '''

    def __init__(self, month, price, product, shorted=None, lots=1000, ordering=None, instructions={}):
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

    def __str__(self):
        string = self.product + ' ' + self.month + ' '
        string += str(self.price)
        string += ' S' if self.shorted else ' L'
        string += ' ' + str(int(self.lots))
        string += ' [c_' + str(self.ordering) + ']'
        return string

    def get_ordering(self):
        return self.ordering

    def set_ordering(self, i):
        self.ordering = i

    def decrement_ordering(self, i):
        self.ordering -= i

    def get_price(self):
        return self.price

    def get_desc(self):
        return self.desc

    def update_price(self, price):
        # updates the price of the future object
        self.price = price

    def get_month(self):
        return self.month

    def get_lots(self):
        return self.lots

    def get_product(self):
        return self.product

    def update_lots(self, lots):
        self.lots = lots
        mult = -1 if self.shorted else 1
        self.delta = 1 * lots * mult

    def get_delta(self):
        return self.delta

    def get_uid(self):
        return self.product + '  ' + self.month
