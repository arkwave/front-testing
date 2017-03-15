"""
File Name      : portfolio.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 15/3/2017
Python version : 3.5
Description    : Script contains implementation of the Portfolio class, as well as helper methods that set/store/manipulate instance variables. This class is used in simulation.py.

"""


from operator import sub


class Portfolio:

    """
    Class representing the overall portfolio. 

    Instance variables:
    1) long_options          : list of long options.
    2) short_options         : list of short options.
    3) long_futures          : list of long futures
    4) short_futures         : list of short futures.
    5) newly_added           : list of newly added securities to this portfolio. used for ease of bookkeeping.
    6) toberemoved           : list of securities to be removed from this portfolio. used for ease of bookkeeping.
    7) value                 : value of the overall portfolio. computed by summing the value of the securities present in the portfolio.
    8) long_pos              : dictionary of the form {product : {month: [set(options), set(futures), delta, gamma, theta, vega]}} containing long positions.
    9) short_pos             : dictionary of the form {product : {month: [set(options), set(futures), delta, gamma, theta, vega]}} containing short positions.
    10) net_greeks           : dictionary of the form {product : {month: [delta, gamma, theta, vega]}} containing net greeks, organized hierarchically by product and month.

    Instance Methods:
    1) set_pnl                : setter method for pnl.
    2) init_sec_by_month      : initializes long_pos, short_pos and net_greeks dictionaries. Only called at init.
    3) add_security           : adds security to portfolio, adjusts greeks accordingly.
    4) remove_security        : removes security from portfolio, adjusts greeks accordingly.
    5) remove_expired         : removes all expired securities from portfolio, adjusts greeks accordingly.
    6) update_sec_by_month    : updates long_pos, short_pos and net_greeks dictionaries in the case of 1) adds 2) removes 3) price/vol changes
    7) update_greeks_by_month : updates the greek counters associated with each month's securities.
    8) compute_value          : computes overall value of portfolio. sub-calls compute_value method of each security.
    9) get_securities_monthly : returns long_pos or short_pos dictionary, depending on flag inputted.
    10) get_securities        : returns a copy of (options, futures) containing either long- or short-securities, depending on flag inputted.
    11) timestep              : moves portfolio forward one day, decrementing tau for all securities.
    12) get_underlying        : returns a list of all UNDERLYING futures in this portfolio. returns the actual futures, NOT a copy.
    13) get_all_futures       : returns a list of all futures, UNDERLYING and PORTFOLIO.
    14) compute_net_greeks    : computes the net product-specific monthly greeks, using long_pos and short_pos as inputs.
    15) exercise_option       : if the option is in the money, exercises it, removing the option from the portfolio and adding/shorting a future as is appropriate.
    16) net_greeks            : returns the self.net_greeks dictionary.
    17) get_underlying_names  : returns a set of names of UNDERLYING futures.
    18) get_all_future_names  : returns a set of names of ALL futures

    """


# TODO [Future]: Currently exercising options only happens at expiry. Figure this
# one out.

    def __init__(self):

        self.long_options = []
        self.short_options = []
        self.long_futures = []
        self.short_futures = []

        self.newly_added = []
        self.toberemoved = []

        self.long_pos = {}
        self.short_pos = {}

        self.net_greeks = {}

        # updating initialized variables
        self.init_sec_by_month('long')
        self.init_sec_by_month('short')
        self.compute_net_greeks(self.long_pos, self.short_pos)
        self.value = self.compute_value()

    def set_pnl(self, pnl):
        self.pnl = pnl

    def init_sec_by_month(self, iden):
        # initialize dictionaries based on whether securities are long or
        # short.
        if iden == 'long':
            op = self.long_options
            ft = self.long_futures
            dic = self.long_pos
        elif iden == 'short':
            op == self.short_options
            ft = self.short_futures
            dic = self.short_pos

        # add in options
        for sec in op:
            month = sec.get_month()
            prod = sec.get_product()
            if prod not in dic:
                dict[prod] = {}
            if month not in dic[prod]:
                dic[prod][month] = [set([sec]), set(), 0, 0, 0, 0]
            else:
                dic[prod][month][0].add(sec)
            self.update_greeks_by_month(prod, month, sec, True)
        # add in futures
        for sec in ft:
            month = sec.get_month()
            prod = sec.get_product()
            if prod not in dic:
                dic[prod] = {}
            if month not in dic[prod]:
                dic[prod][month] = [set(), set([sec]), 0, 0, 0, 0]
            else:
                dic[prod][month][1].add(sec)

    def compute_net_greeks(self):
        # Computes net greeks organized hierarchically according to product and
        # month. Updates net_greeks by using long_pos and short_pos.
        common_products = set(
            self.long_pos.keys() & set(self.short_pos.keys()))

        long_products_unique = set(self.long_pos.keys()) - common_products
        short_products_unique = set(self.short_pos.keys()) - common_products

        # dealing with common products
        for product in common_products:
            # instantiating variables to make it neater.
            longdata = self.long_pos[product]
            shortdata = self.short_pos[product]
            common_months = set(longdata.keys()) & set(
                shortdata.keys())
            # finding unique months within this product.
            long_pos_unique_mths = set(
                longdata.keys()) - common_months
            short_pos_unique_mths = set(
                shortdata.keys()) - common_months
            # dealing with common months
            for month in common_months:
                long_greeks = longdata[month][2:]
                short_greeks = shortdata[month][2:]
                net = list(map(sub, long_greeks, short_greeks))
                self.net_greeks[product][month] = net
            # dealing with non overlapping months
            for month in longdata_unique_mths:
                self.net_greeks[product][month] = longdata[month][2:]
            for month in shortdata_unique_mths:
                self.net_greeks[product][month] = shortdata[month][2:]

        # dealing with non-overlapping products
        for product in long_products_unique:
            data = self.long_pos[product]
            # iterating over all months corresponding to non-overlapping
            # product for which we have long positions
            for month in data:
                self.net_greeks[product][month] = data[month][2:]

        for product in short_products_unique:
            data = self.long_pos[product]
            # iterating over all months corresponding to non-overlapping
            # product for which we have short positions.
            for month in data:
                self.net_greeks[product][month] = data[month][2:]

    def add_security(self, security, flag):
        # adds a security into the portfolio, and updates relevant lists and
        # adjusts greeks of the portfolio.
        if flag == 'long':
            op = self.long_options
            ft = self.long_futures
        elif flag == 'short':
            op = self.short_options
            ft = self.short_futures
        if security.get_desc() == 'option':
            op.append(security)
        elif security.get_desc() == 'future':
            ft.append(security)
        self.newly_added.append(security)
        self.update_sec_by_month(True, flag)

    def remove_security(self, security, flag):
        # removes a security from the portfolio, updates relevant list and
        # adjusts greeks of the portfolio.
        if flag == 'long':
            op = self.long_options
            ft = self.long_futures
        elif flag == 'short':
            op = self.short_options
            ft = self.short_futures
        if security.get_desc() == 'option':
            op.remove(security)
        elif security.get_desc() == 'future':
            ft.remove(security)
        self.toberemoved.append(security)
        self.update_sec_by_month(False)

    # TODO: add in condition for knockouts and discretionary for knockin
    def remove_expired(self):
        for sec in self.long_options:
            if sec.tau == 0:
                self.remove_security(sec, 'long')
        for sec in self.short_options:
            if sec.tau == 0:
                self.remove_security(sec, 'short')

    def update_sec_by_month(self, added, flag, price=None, vol=None):
        '''
        Helper method that updates the sec_by_month dictionary.

        Inputs: 
        1) added  : boolean flag that indicates if securities are being added or removed. Valid inputs: True, False.
        2) flag   : string flag that indicates if the securities to be manipulated are short or long. Valid inputs: 'short', 'long'
        3) price  : new price of underlying. If explicitly passed in, the method assumes that new price/vol data is being fed into the portfolio, and acts accordingly. defaults to None.
        4) vol    : new vol. If explicitly passed in, the method assumes that new price/vol data is being fed into the portfolio, and acts accordingly. defaults to None.

        Outputs : Updates long_pos, short_pos and net_greeks dictionaries.

        Notes: this method does 90% of all the heavy lifting in the portfolio class. Don't mess with this unless you know EXACTLY what each part is doing.
        '''
        if flag == 'long':
            dic = self.long_pos
            op = self.long_options
            ft = self.long_futures
        elif flag == 'short':
            dic = self.short_pos
            op = self.short_options
            ft = self.short_futures
        # adding/removing security to portfolio
        if price is None and vol is None:
            # adding
            if added:
                target = self.newly_added.copy()
                self.newly_added.clear()
                for sec in target:
                    product = sec.get_product()
                    month = sec.get_month()
                    if product not in dic:
                        dic[product] = {}
                    if month not in dic[product]:
                        if sec.get_desc() == 'option':
                            dic[product][month] = [
                                set([sec]), set(), 0, 0, 0, 0]
                        elif sec.get_desc() == 'future':
                            dic[product][month] = [
                                set(), set([sec]), 0, 0, 0, 0]
                    else:
                        if sec.get_desc() == 'option':
                            dic[product][month][0].add(sec)
                        else:
                            dic[product][month][1].add(sec)
                    self.update_greeks_by_month(
                        product, month, sec, added, flag)
            # removing
            else:
                target = self.toberemoved.copy()
                self.toberemoved.clear()
                for sec in target:
                    product = sec.get_product()
                    month = sec.get_month()
                    if sec.get_desc() == 'option':
                        dic[product][month][0].remove(sec)
                    else:
                        dic[product][month][1].remove(sec)
                    self.update_greeks_by_month(
                        product, month, sec, added, flag)

        # updating greeks per month when feeding in new prices/vols
        else:
            # updating greeks based on new price and vol data
            for sec in op:
                sec.update_greeks(vol)
            # updating cumulative greeks on a month-by-month basis.
            for sec in op:
                product = sec.get_product()
                month = sec.get_month()
                # reset all greeks
                dic[product][month][2:] = [0, 0, 0]
                # update from scratch. treated as fresh add of all existing
                # securities.
                self.update_greeks_by_month(product, month, sec, True, flag)

    def update_greeks_by_month(self, product, month, sec, added, flag):
        if flag == 'long':
            dic = self.long_pos
        else:
            dic = self.short_pos
        data = dic[product][month]
        if sec.get_desc() == 'option':
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

        self.compute_net_greeks()

    def compute_value(self):
        val = 0
        # try:
        for sec in self.long_options:
            val += sec.get_value()
        for sec in self.short_options:
            val -= sec.get_value()
        for sec in self.long_futures:
            val += sec.get_value()
        for sec in self.short_futures:
            val -= sec.get_value()
        # except
        return val

    def exercise_option(self, sec, flag):
        if flag == 'long':
            op = self.long_options
        else:
            op = self.short_options
        for option in op:
            if option.moneyness() == 1:
                # convert into a future.
                underlying = option.get_future()
                self.remove_security(option, flag)
                self.add_security(underlying, flag)

 ### getter/utility methods ###

    def get_securities_monthly(self, flag):
        if flag == 'long':
            dic = self.long_pos
        else:
            dic = self.short_pos
        dic = dic.copy()
        return dic

    def get_securities(self, flag):
        if flag == 'long':
            op = self.long_options
            ft = self.long_futures
        else:
            op = self.short_options
            ft = self.short_futures
        lst1 = op.copy()
        lst2 = ft.copy()
        return (lst1, lst2)

    def get_underlying(self):
        u_set = set()
        all_options = self.long_options + self.short_options
        for sec in all_options:
            u_set.add(sec.get_underlying())
        return list(u_set)

    def get_underlying_names(self):
        underlying = self.get_underlying()
        namelist = [ud.get_product() for ud in underlying]
        return set(namelist)

    def get_all_futures(self):
        retr = self.get_underlying()
        port_futures = self.long_futures + self.short_futures
        retr.extend(port_futures)
        return retr

    def timestep(self, value):
        all_options = self.long_options + self.short_options
        for option in all_options:
            option.update_tau(value)

    def net_greeks(self):
        return self.net_greeks

    def get_all_future_names(self):
        all_futures = self.get_all_futures()
        names = [ft.get_product() for ft in all_futures]
        return set(names)
