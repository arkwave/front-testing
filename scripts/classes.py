# File containing all the classes required by the simulation.
from calc import _compute_value, _compute_greeks

lots = 10


class Option:

    """
    Definitions for the option class.
    Instance variables:
        1)  ki         =   knock-in value.
        2)  ko         =   knock-out value
        3)  K          =   strike price
        4)  price      =   price quoted for the option.
        5)  tau        =   time to expiry in years (as per black-scholes)
        6)  char       =   call or put option.
        7)  vol        =   current implied volatility
        8)  s          =   current price of underlying.
        9)  r          =   risk free interest rate. assumed to be 0, can be changed.
        10) desc       =   string description of the object
        11) month      =   month of expiry. used to keep greeks for different months separate.
        12) value      =   theoretical black-scholes value of the option.
        13) barrier    =   American or European barrier.
        14) lots       =   the quantity of underlying being shorted or bought upon expiry. 
        15) bullet     =   True - Bullet. False - Daily.
        16) underlying =   the underlying futures object

        Note: ki, ko, bullet and barrier default to None and must be expressly overridden if an exotic option is desired.

    Instance Methods:
    1) get_month       : returns the month of the underlying contract.
    2) init_greeks     : initializes the greeks of this option. only called on init.
    3) update_greeks   : updates and reassigns greeks upon changes in underlying price/vol.
    4) greeks          : getter method that returns a tuple of the greeks
    5) compute_vol     :  [not implemented]
    6) compute_value   : computes value of the object based on the appropriate valuation method.
    7) get_value       : getter method for the value of the object.
    8) update_tau      : updates time to expiry.
    9) exercise        : exercises the option. as of now, returns a boolean.
    10) moneyness      : returns 1, 0 , -1 depending if option is itm, atm, otm.
    11) get_future     : returns the underlying future object.
    12) get_desc       : returns 'option'
    13) get_underlying : returns the NAME of the underlying future.
    """

    def __init__(self, strike, price, tau, char, vol, s, underlying, barrier=None, lots=lots, bullet=None, ki=None, ko=None):
        self.underlying = underlying
        self.bullet = bullet
        self.lots = lots
        self.desc = 'option'
        self.ki = ki
        self.ko = ko
        self.K = strike
        self.tau = tau
        self.char = char
        self.price = price
        self.vol = vol
        self.s = s
        self.r = 0

        self.value = self.compute_value()
        self.delta, self.gamma, self.theta, self.vega = self.init_greeks()

    def get_future(self):
        return self.underlying

    def get_month(self):
        return self.underlying.get_month()

    def get_desc(self):
        return self.desc

    def init_greeks(self):
        # initializes relevant greeks. only used once, when initializing Option
        # object.
        return _compute_greeks(self.char, self.K,  self.tau, self.vol, self.s, self.r)

    def update_greeks(self, s, vol, tau):
        # method that updates greeks given new values of s, vol and tau, and subsequently updates value.
        # used in passage of time step.
        self.delta, self.gamma, self.theta, self.vega = _compute_greeks(
            self.char, self.K, tau, vol, s, self.r)
        self.value = self.compute_value()

    def greeks(self):
        # getter method for greeks. preserves abstraction barrier.
        return self.delta, self.gamma, self.theta, self.vega

    def compute_vol(underlying, price, strike, tau, r):
        # computes implied vol from market price data
        return _compute_iv(underlying, price, strike, tau, r)

    def compute_value(self):
        # computes the value of this structure from relevant information.
        return _compute_value(self.char, self.tau, self.vol, self.K, self.s, self.r, ki=self.ki, ko=self.ko)

    def get_value(self):
        return self.value

    def update_tau(self, diff):
        self.tau -= diff

    def get_underlying(self):
        return self.underlying.get_name()

    def exercise(self):
        worth = self.moneyness()
        if worth > 0:
            return True
        else:
            return False

    def moneyness(self):
        # at the money
        if self.K == self.s:
            return 0
        if self.char == 'call':
            if self.K < self.s:
                return 1
            else:
                return -1
        elif self.char == 'put':
            if self.K > self.s:
                return 1
            else:
                return -1


class Future:

    '''
    Class representing a Future object. Instance variables are:
    1) month  :  the contract month.
    2) price  :  the quoted price of the future.
    3) desc   :  string description of the object
    4) lots   :  number of lots represented by each future contract.
    6) name   :  the commodity of this future

    Instance Methods:
    1) greeks         : dummy method.
    2) get_value      : returns price of the future.
    3) update_price   : updates the price based on inputted data.
    4) update_greeks  : dummy method.
    5) get_month      : returns contract month.
    6) get_lots       : returns lot size
    7) get_name : returns the name of this contract (i.e. the commodity)
    '''

    def __init__(self, month, name, price, lots=lots):
        self.name = name
        self.lots = lots
        self.desc = 'future'
        self.month = month
        self.price = price

    def greeks(self):
        # method included for completeness. Futures and Options are both
        # treated as securities in portfolio class, requiring equivalent
        # methods.
        return 0, 0, 0, 0

    def get_desc(self):
        return self.desc

    def get_value(self):
        # getter method for price of the future.
        return self.price

    def update_price(self, price):
        # updates the price of the future object
        self.price = price

    def update_greeks(self, price, vol):
        # method intentionally left blank. Futures have no greeks.
        pass

    def get_month(self):
        return self.month

    def get_lots(self):
        return self.lots

    def get_name(self):
        return self.name


class Portfolio:

    '''
    Class representing the overall portfolio. 

    Instance variables:
    1) securities    : list of Option or Future objects that constitute this portfolio.
    2) newly_added   : list of newly added securities to this portfolio. 
    3) sec_by_month  : dictionary that maps months to securities that expire in that month. 
                       Format of the dictionary is Month: [set(securities), delta, gamma, theta, vega] 
                       where the greeks are the aggregate greeks over all securities belonging to that month.
    4) value         : value of the overall portfolio. computed by summing the value of the securities present in the portfolio.
    5) PnL           : records overall change in value of portfolio.

    Instance Methods:
    1) set_pnl                : setter method for pnl.
    2) init_sec_by_month      : initializes sec_by_month dictionary. Only called at init.
    3) add_security           : adds security to portfolio, adjusts greeks accordingly.
    4) remove_security        : removes security from portfolio, adjusts greeks accordingly.
    5) remove_expired         : removes all expired securities from portfolio, adjusts greeks accordingly.
    6) update_sec_by_month    : updates sec_by_month dictionary in the case of 1) adds 2) removes 3) price/vol changes
    7) update_greeks_by_month : updates the greek counters associated with each month's securities.
    8) compute_value          : computes overall value of portfolio. sub-calls compute_value of each security.
    9) get_securities_monthly : returns sec_by_month
    10) get_securities        : returns list of securities
    11) timestep              : moves portfolio forward one day, decrements tau for all securities.
    12) get_underlying        : returns a list of all underlying futures in this portfolio.


    '''

# TODO: Currently exercising options only happens at expiry. Figure this
# one out.

    def __init__(self, options, futures):
        self.options = options
        self.futures = futures
        self.newly_added = []
        self.toberemoved = []
        self.sec_by_month = {}
        self.value = self.compute_value()
        self.pnl = 0

        # updating initialized variables
        self.init_sec_by_month()

    def set_pnl(self, pnl):
        self.pnl = pnl

    def init_sec_by_month(self):
        # add in options
        for sec in self.options:
            month = sec.get_month()
            if month not in self.sec_by_month:
                self.sec_by_month[month] = [set([sec]), set(), 0, 0, 0, 0]
            else:
                self.sec_by_month[month][0].add(sec)
            self.update_greeks_by_month(month, sec, True)
        # add in futures
        for sec in self.futures:
            month = sec.get_month()
            if month not in self.sec_by_month:
                self.sec_by_month[month] = [set(), set([sec]), 0, 0, 0, 0]
            else:
                self.sec_by_month[month][1].add(sec)

    def add_security(self, security):
        # adds a security into the portfolio, and updates the sec_by_month
        # dictionary.
        if security.get_desc() == 'option':
            self.options.append(security)
        elif security.get_desc() == 'future':
            self.futures.append(security)
        self.newly_added.append(security)
        self.update_sec_by_month(True)

    def remove_security(self, security):
        # removes a security from the portfolio, updates sec_by_month, and
        # adjusts greeks of the portfolio.
        if security.get_desc() == 'option':
            self.options.remove(security)
        elif security.get_desc() == 'future':
            self.futures.remove(security)
        self.toberemoved.append(security)
        self.update_sec_by_month(False)

    def remove_expired(self):
        for sec in self.securities:
            if sec.tau == 0:
                self.remove_security(sec)

    def update_sec_by_month(self, added, price=None, vol=None):
        '''
        Helper method that updates the sec_by_month dictionary.
        Inputs  : None.
        Outputs : Updates sec_by_month dictionary.
        '''
        # adding/removing security to portfolio
        if price is None and vol is None:
            # adding
            if added:
                target = self.newly_added.copy()
                self.newly_added.clear()
                for sec in target:
                    month = sec.get_month()
                    if month not in self.sec_by_month:
                        if sec.get_desc() == 'option':
                            self.sec_by_month[month] = [
                                set([sec]), set(), 0, 0, 0, 0]
                        elif sec.get_desc() == 'future':
                            self.sec_by_month[month] = [
                                set(), set([sec]), 0, 0, 0, 0]
                    else:
                        if sec.get_desc() == 'option':
                            self.sec_by_month[month][0].add(sec)
                        else:
                            self.sec_by_month[month][1].add(sec)
                    self.update_greeks_by_month(month, sec, added)
            # removing
            else:
                target = self.toberemoved.copy()
                self.toberemoved.clear()
                for sec in target:
                    month = sec.get_month()
                    self.sec_by_month[month][0].remove(sec)
                    self.update_greeks_by_month(month, sec, added)

        # updating greeks per month when feeding in new prices/vols
        else:
            # updating greeks based on new price and vol data
            for sec in self.options:
                sec.update_greeks(price, vol)
            # updating cumulative greeks on a month-by-month basis.
            for sec in self.securities:
                month = sec.get_month()
                # reset all greeks
                self.sec_by_month[month][1:] = [0, 0, 0]
                # update from scratch. treated as fresh add of all existing
                # securities.
                self.update_greeks_by_month(month, sec, True)

    def update_greeks_by_month(self, month, sec, added):
        data = self.sec_by_month[month]
        delta, gamma, theta, vega = sec.greeks()
        if added:
            data[2] += delta
            data[3] += gamma
            data[4] += theta
            data[5] += vega
        else:
            data[2] -= delta
            data[3] -= gamma
            data[4] -= theta
            data[5] -= vega

    def compute_value(self):
        val = 0
        for sec in self.options:
            val += sec.get_value()
        for sec in self.futures:
            val += sec.get_value()
        return val

    def get_securities_monthly(self):
        dic = self.sec_by_month.copy()
        return dic

    def get_securities(self):
        lst1 = self.options.copy()
        lst2 = self.futures.copy()
        return lst1, lst2

    def get_underlying(self):
        u_set = set()
        for sec in self.options:
            u_set.add(sec.get_underlying())
        for sec in self.futures:
            u_set.add(sec.get_name())

        return list(u_set)

    def exercise_option(self, sec):
        for option in self.options:
            if option.moneyness() == 1:
                # convert into a future.
                underlying = option.get_future()
                self.remove_security(option)
                self.add_security(underlying)

    def timestep(self, value):
        for option in self.options:
            option.update_tau(value)
