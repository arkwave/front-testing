# File containing all the classes required by the simulation.
from calc import _compute_value, _compute_greeks

lots = 10


# TODO: Product specific information.
# product : [futures_multiplier - (dollar_mult, lot_mult),
# options_multiplier - (dollar_mult, lot_mult), futures_tick,
# options_tick, brokerage]


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
        17) payoff     =   American or European option
        18) direc      =   Indicates the direction of the barrier (up and out etc.)

        Note: ki, ko, bullet, direc and barrier default to None and must be expressly overridden if an exotic option is desired.

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
    14) check_active   : for barriers, checks to see if this barrier is active or not.
    """

    def __init__(self, strike, tau, char, vol, underlying, payoff, direc=None barrier=None, lots=lots, bullet=None, ki=None, ko=None):

        self.payoff = payoff
        self.underlying = underlying
        self.bullet = bullet  # daily = list of bullets.
        self.lots = lots
        self.desc = 'option'
        self.ki = ki
        self.ko = ko
        self.K = strike
        self.tau = tau
        self.char = char
        self.vol = vol
        self.s = self.underlying.get_price()
        self.r = 0
        self.price = self.compute_price()
        self.delta, self.gamma, self.theta, self.vega = self.init_greeks()
        self.active = self.check_active()
        self.direc = direc

    def check_active(self):
        active = True
        if self.ki:
            if self.direc == 'up':
                # check if up and in is active
                if self.s < self.ki:
                    active == False
            elif self.direc == 'down':
                if self.s > self.ki:
                    active == False
        if self.ko:
            if self.direc == 'up':
                if self.s > self.ko:
                    active == False
            elif self.direc == 'down':
                if self.s < self.ko:
                    active == False
        return active

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

    def update_greeks(self, vol):
        # method that updates greeks given new values of s, vol and tau, and subsequently updates value.
        # used in passage of time step.
        self.delta, self.gamma, self.theta, self.vega = _compute_greeks(
            self.char, self.K, self.tau, vol, self.s, self.r)
        self.vol = vol
        self.price = self.compute_value()

    def greeks(self):
        # getter method for greeks. preserves abstraction barrier.
        return self.delta, self.gamma, self.theta, self.vega

    # def compute_vol(underlying, price, strike, tau, r):
    #     # computes implied vol from market price data
    #     return _compute_iv(underlying, price, strike, tau, r)

    def compute_price(self):
        # computes the value of this structure from relevant information.
        return _compute_value(self.char, self.tau, self.vol, self.K, self.s, self.r, self.payoff, ki=self.ki, ko=self.ko)

    def get_price(self):
        return self.price

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
    5) product   :  the commodity of this future.

    Instance Methods:
    1) greeks         : dummy method.
    2) get_value      : returns price of the future.
    3) update_price   : updates the price based on inputted data.
    4) update_greeks  : dummy method.
    5) get_month      : returns contract month.
    6) get_lots       : returns lot size
    7) get_name : returns the name of this contract (i.e. the commodity)
    '''

    def __init__(self, month, price, product, lots=lots):
        self.product = product
        self.lots = lots
        self.desc = 'future'
        self.month = month
        self.price = price
        self.product = product

    def get_price(self):
        return self.price

    def get_desc(self):
        return self.desc

    def get_price(self):
        # getter method for price of the future.
        return self.price

    def update_price(self, price):
        # updates the price of the future object
        self.price = price

    def get_month(self):
        return self.month

    def get_lots(self):
        return self.lots

    def get_product(self):
        return self.product
