from operator import sub


class Portfolio:

    '''
    Class representing the overall portfolio. 

    Instance variables:
    1) long/short options    : list of Option or Future objects that constitute this portfolio. Divided into long/short options and long/short futures.
    2) newly_added           : list of newly added securities to this portfolio. 
    3) long/short positions  : dictionary that maps months to securities that expire in that month. long_pos and short_pos. Format of the dictionary is Month: [set(options), set(futures), delta, gamma, theta, vega] where the greeks are the aggregate greeks over all securities belonging to that month.
    4) value                 : value of the overall portfolio. computed by summing the value of the securities present in the portfolio.
    5) PnL                   : records overall change in value of portfolio.

    Instance Methods:
    1) set_pnl                : setter method for pnl.
    2) init_sec_by_month      : initializes sec_by_month dictionary. Only called at init.
    3) add_security           : adds security to portfolio, adjusts greeks accordingly.
    4) remove_security        : removes security from portfolio, adjusts greeks accordingly.
    5) remove_expired         : removes all expired securities from portfolio, adjusts greeks accordingly.
    6) update_sec_by_month    : updates sec_by_month dictionary in the case of 1) adds 2) removes 3) price/vol changes
    7) update_greeks_by_month : updates the greek counters associated with each month's securities.
    8) compute_value          : computes overall value of portfolio. sub-calls compute_value of each security.
    9) get_securities_monthly : returns sec_by_month dictionary.
    10) get_securities        : returns a copy of (options, futures)
    11) timestep              : moves portfolio forward one day, decrements tau for all securities.
    12) get_underlying        : returns a list of all underlying futures in this portfolio. returns the actual futures, NOT a copy.
    13) get_all_futures       : returns a list of all futures, underlying and portfolio.

    '''
# TODO [Future]: Currently exercising options only happens at expiry. Figure this
# one out.

    def __init__(self, long_sec, short_sec):

        self.long_options = long_sec[0]
        self.short_options = short_sec[0]
        self.long_futures = long_sec[1]
        self.short_futures = short_sec[1]

        self.newly_added = []
        self.toberemoved = []

        self.long_pos = {}
        self.short_pos = {}

        self.net_greeks = {}

        self.pnl = 0

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
            if month not in dic:
                dic[month] = [set([sec]), set(), 0, 0, 0, 0]
            else:
                dic[month][0].add(sec)
            self.update_greeks_by_month(month, sec, True)
        # add in futures
        for sec in ft:
            month = sec.get_month()
            if month not in dic:
                dic[month] = [set(), set([sec]), 0, 0, 0, 0]
            else:
                dic[month][1].add(sec)

    def compute_net_greeks(self):
        common_months = set(self.long_pos.keys()) & set(self.short_pos.keys())
        self.long_pos_unique = set(self.long_pos.keys()) - common_months
        self.short_pos_unique = set(self.short_pos.keys()) - common_months
        # dealing with overlaps
        for month in common_months:
            long_greeks = self.long_pos[month][2:]
            short_greeks = self.short_pos[month][2:]
            net = map(sub, long_greeks, short_greeks)
            self.net_greeks[month] = net
        # dealing with non overlaps
        for month in self.long_pos_unique:
            self.net_greeks[month] = self.long_pos[month][2:]
        for month in self.short_pos_unique:
            self.net_greeks[month] = self.short_pos[month][2:]

    def add_security(self, security, flag):
        # adds a security into the portfolio, and updates the sec_by_month
        # dictionary.
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
        self.update_sec_by_month(True)

    def remove_security(self, security, flag):
        # removes a security from the portfolio, updates sec_by_month, and
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
        Inputs  : None.
        Outputs : Updates sec_by_month dictionary.
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
                    month = sec.get_month()
                    if month not in dic:
                        if sec.get_desc() == 'option':
                            dic[month] = [
                                set([sec]), set(), 0, 0, 0, 0]
                        elif sec.get_desc() == 'future':
                            dic[month] = [
                                set(), set([sec]), 0, 0, 0, 0]
                    else:
                        if sec.get_desc() == 'option':
                            dic[month][0].add(sec)
                        else:
                            dic[month][1].add(sec)
                    self.update_greeks_by_month(month, sec, added, flag)
            # removing
            else:
                target = self.toberemoved.copy()
                self.toberemoved.clear()
                for sec in target:
                    month = sec.get_month()
                    dic[month][0].remove(sec)
                    self.update_greeks_by_month(month, sec, added, flag)

        # updating greeks per month when feeding in new prices/vols
        else:
            # updating greeks based on new price and vol data
            for sec in op:
                sec.update_greeks(vol)
            # updating cumulative greeks on a month-by-month basis.
            for sec in op:
                month = sec.get_month()
                # reset all greeks
                dic[month][2:] = [0, 0, 0]
                # update from scratch. treated as fresh add of all existing
                # securities.
                self.update_greeks_by_month(month, sec, True, flag)

    def update_greeks_by_month(self, month, sec, added, flag):
        if flag == 'long':
            dic = self.long_pos
        else:
            dic = self.short_pos
        data = dic[month]
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
        for sec in self.options:
            val += sec.get_value()
        for sec in self.futures:
            val += sec.get_value()
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
        return lst1, lst2

    def get_underlying(self):
        u_set = set()
        all_options = self.long_options + self.short_options
        for sec in all_options:
            u_set.add(sec.get_underlying())
        return list(u_set)

    def get_all_futures(self):
        retr = self.get_underlying()
        port_futures = self.long_futures + self.short_futures
        retr.extend(port_futures)
        return retr

    def timestep(self, value):
        all_options = self.long_options + self.short_options
        for option in all_options:
            option.update_tau(value)
