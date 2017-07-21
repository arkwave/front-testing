"""
File Name      : portfolio.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 17/4/2017
Python version : 3.5
Description    : Script contains implementation of the Portfolio class, as well as helper methods that set/store/manipulate instance variables. This class is used in simulation.py.

"""


# Dictionary of multipliers for greeks/pnl calculation.
# format  =  'product' : [dollar_mult, lot_mult, futures_tick,
# options_tick, pnl_mult]

multipliers = {
    'LH':  [22.046, 18.143881, 0.025, 0.05, 400],
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
    'OBM': [1.0604, 50, 0.25, 1, 53.02],
    'MW':  [0.3674333, 136.07911, 0.25, 10, 50]
}

from timeit import default_timer as timer
from operator import add
import pprint
import numpy as np
from collections import deque

seed = 7
np.random.seed(seed)


class Portfolio:

    """
    Class representing the overall portfolio.

    Instance variables:
    1) OTC_options           : list of OTC options.
    2) hedge_options         : list of hedge options.
    3) OTC_futures           : list of OTC futures
    4) hedge_futures         : list of hedge futures.
    5) newly_added           : list of newly added securities to this portfolio. used for ease of bookkeeping.
    6) toberemoved           : list of securities to be removed from this portfolio. used for ease of bookkeeping.
    7) value                 : value of the overall portfolio. computed by summing the value of the securities present in the portfolio.
    8) OTC                   : dictionary of the form {product : {month: [set(options), set(futures), delta, gamma, theta, vega]}} containing OTC positions.
    9) hedges                : dictionary of the form {product : {month: [set(options), set(futures), delta, gamma, theta, vega]}} containing hedge positions.
    10) net_greeks           : dictionary of the form {product : {month: [delta, gamma, theta, vega]}} containing net greeks, organized hierarchically by product and month.

    Instance Methods:
    1) set_pnl                : setter method for pnl.
    2) init_sec_by_month      : initializes OTC, hedges and net_greeks dictionaries. Only called at init.
    3) add_security           : adds security to portfolio, adjusts greeks accordingly.
    4) remove_security        : removes security from portfolio, adjusts greeks accordingly.
    5) remove_expired         : removes all expired securities from portfolio, adjusts greeks accordingly.
    6) update_sec_by_month    : updates OTC, hedges and net_greeks dictionaries in the case of 1) adds 2) removes 3) price/vol changes
    7) update_greeks_by_month : updates the greek counters associated with each month's securities.
    8) compute_value          : computes overall value of portfolio. sub-calls compute_value method of each security.
    9) get_securities_monthly : returns OTC or hedges dictionary, depending on flag inputted.
    10) get_securities        : returns a copy of (options, futures) containing either OTC- or hedge-securities, depending on flag inputted.
    11) timestep              : moves portfolio forward one day, decrementing tau for all securities.
    12) get_underlying        : returns a list of all UNDERLYING futures in this portfolio. returns the actual futures, NOT a copy.
    13) get_all_futures       : returns a list of all futures, UNDERLYING and PORTFOLIO.
    14) compute_net_greeks    : computes the net product-specific monthly greeks, using OTC and hedges as inputs.
    15) exercise_option       : if the option is in the money, exercises it, removing the option from the portfolio and adding/hedgeing a future as is appropriate.
    16) net_greeks            : returns the self.net_greeks dictionary.
    17) get_underlying_names  : returns a set of names of UNDERLYING futures.
    18) get_all_future_names  : returns a set of names of ALL futures

    """

    def __init__(self):

        self.OTC_options = deque()
        self.hedge_options = deque()
        self.OTC_futures = []
        self.hedge_futures = []

        # utility litst
        self.newly_added = []
        self.toberemoved = []

        self.OTC = {}
        self.hedges = {}

        self.net_greeks = {}

        # updating initialized variables
        self.init_sec_by_month('OTC')
        self.init_sec_by_month('hedge')
        self.compute_net_greeks()
        self.value = self.compute_value()

    def __str__(self):
        # custom print representation for this class.
        otcops = [op.__str__() for op in self.OTC_options]
        otcft = [op.__str__() for op in self.OTC_futures]
        hedgeops = [op.__str__() for op in self.hedge_options]
        hedgeft = [op.__str__() for op in self.hedge_futures]
        nets = self.net_greeks
        # otcs = self.OTC
        # hedges = self.hedges

        r_dict = {'OTC Options': otcops,
                  'OTC Futures': otcft,
                  'Hedge Options': hedgeops,
                  'Hedge Futures': hedgeft,
                  'Net Greeks': nets}

        return str(pprint.pformat(r_dict))

    def update_sec_lots(self, sec, flag, lots):
        """Updates the lots of a given security, updates the dictionary it is contained in, and 
        updates net_greeks 

        Args:
            sec (list): list of securities whose lots are being updated
            flag (TYPE): indicates if this security is an OTC or hedge option
            lots (TYPE): list of lot values, where lots[i] corresponds to the new lot value of sec[i]
        """
        ops = self.OTC_options if flag == 'OTC' else self.hedge_futures
        fts = self.OTC_futures if flag == 'OTC' else self.hedge_futures
        for s in sec:
            # sanity checks: make sure the security is present in the relevant
            # list selected by flag
            if s.desc == 'Option' and s not in ops:
                raise ValueError('This option is not in the portfolio.')
            elif s.desc == 'Future' and s not in fts:
                raise ValueError('This future is not in the portfolio.')
            else:
                s.update_lots(lots[sec.index(s)])

        self.update_sec_by_month(False, flag, update=True)

    def empty(self):
        """Checks to see if portfolio is completely empty 

        Returns:
            bool: True if empty, false otherwise. 
        """
        return (len(self.OTC) == 0) and (len(self.hedges) == 0)

    def init_sec_by_month(self, iden):
        """Initializing method that creates the relevant futures list, options list, and dictionary depending on the flag passed in. 

        Args:
            iden (str): flag that indicates which set of data structures is to be initialized. 
            Valid Inputs: 'OTC', 'hedge' to initialize OTC and hedge positions respectively.

        Returns:
            None: Initializes the relevant data structures.
        """
        # initialize dictionaries based on whether securities are OTC or
        # hedge.
        if iden == 'OTC':
            op = self.OTC_options
            ft = self.OTC_futures
            dic = self.OTC
        elif iden == 'hedge':
            op = self.hedge_options
            ft = self.hedge_futures
            dic = self.hedges

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
        ''' Computes net greeks organized hierarchically according to product and
         month. Updates net_greeks by using OTC and hedges. '''

        final_dic = {}
        common_products = set(self.OTC.keys()) & set(
            self.hedges.keys())
        OTC_products_unique = set(self.OTC.keys()) - common_products
        hedge_products_unique = set(self.hedges.keys()) - common_products

        # dealing with common products
        for product in common_products:
            # checking existence.
            if product not in final_dic:
                final_dic[product] = {}
            # instantiating variables to make it neater.
            OTCdata = self.OTC[product]
            hedgedata = self.hedges[product]
            common_months = set(OTCdata.keys()) & set(
                hedgedata.keys())
            # finding unique months within this product.
            OTC_unique_mths = set(
                OTCdata.keys()) - common_months
            hedges_unique_mths = set(
                hedgedata.keys()) - common_months
            # dealing with common months
            for month in common_months:
                OTC_greeks = OTCdata[month][2:]
                # print('DEBUG: OTC greeks: ', OTC_greeks)
                hedge_greeks = hedgedata[month][2:]
                net = list(map(add, OTC_greeks, hedge_greeks))
                final_dic[product][month] = net
            # dealing with non overlapping months
            for month in OTC_unique_mths:
                # checking if month has options.
                if OTCdata[month][0]:
                    final_dic[product][month] = OTCdata[month][2:]
            for month in hedges_unique_mths:
                # checking if month has options.
                if hedgedata[month][0]:
                    final_dic[product][month] = hedgedata[month][2:]

        # dealing with non-overlapping products
        for product in OTC_products_unique:
            data = self.OTC[product]
            # checking existence
            if product not in final_dic:
                final_dic[product] = {}
            # iterating over all months corresponding to non-overlapping
            # product for which we have OTC positions
            for month in data:
                # checking if month has options.
                if data[month][0]:
                    final_dic[product][month] = data[month][2:]

        for product in hedge_products_unique:
            data = self.hedges[product]
            # checking existence
            if product not in final_dic:
                final_dic[product] = {}
            # iterating over all months corresponding to non-overlapping
            # product for which we have hedge positions.
            for month in data:
                # checking if month has options.
                if data[month][0]:
                    final_dic[product][month] = data[month][2:]

        # print('final_dic: ', final_dic)
        self.net_greeks = final_dic

    def add_security(self, security, flag):
        # adds a security into the portfolio, and updates relevant lists and
        # adjusts greeks of the portfolio.

        if flag == 'OTC':
            op = self.OTC_options
            ft = self.OTC_futures
        elif flag == 'hedge':
            op = self.hedge_options
            ft = self.hedge_futures

        for sec in security:
            if sec.get_desc() == 'option':
                try:
                    op.appendleft(sec)
                except UnboundLocalError:
                    print('flag: ', flag)
            elif sec.get_desc() == 'future':
                ft.append(sec)

        self.newly_added.extend(security)
        self.update_sec_by_month(True, flag)

    def remove_security(self, security, flag):
        # removes a security from the portfolio, updates relevant list and
        # adjusts greeks of the portfolio.
        if flag == 'OTC':
            op = self.OTC_options
            ft = self.OTC_futures
        elif flag == 'hedge':
            op = self.hedge_options
            ft = self.hedge_futures
        for sec in security:
            try:
                if sec.get_desc() == 'option':
                    op.remove(sec)
                elif sec.get_desc() == 'future':
                    ft.remove(sec)
            except ValueError:
                print(str(sec))
                print('specified security doesnt exist in this portfolio')

        self.toberemoved.extend(security)
        self.update_sec_by_month(False, flag)

    def remove_expired(self):
        '''Removes all expired options from the portfolio. '''
        explist = {'hedge': [], 'OTC': []}
        for sec in self.OTC_options:
            # handling barrier case
            if sec.barrier == 'amer':
                if sec.knockedout:
                    explist['OTC'].append(sec)
                    # self.remove_security(sec, 'OTC')
            # vanilla/knockin case
            if sec.check_expired():
                explist['OTC'].append(sec)
                # self.remove_security(sec, 'OTC')
        for sec in self.hedge_options:
            # handling barrier case.
            if sec.barrier == 'amer':
                if sec.knockedout:
                    explist['hedge'].append(sec)
                    # self.remove_security(sec, 'hedge')
                    # vanilla/knockin case
            elif sec.check_expired():
                explist['hedge'].append(sec)
                # self.remove_security(sec, 'hedge')
                # handle rollover futures.
        for ft in self.OTC_futures:
            if ft.check_expired():
                explist['OTC'].append(ft)
                # self.remove_security(ft, 'OTC')

        for ft in self.hedge_futures:
            if ft.check_expired():
                explist['hedge'].append(ft)
                # self.remove_security(ft, 'hedge')

        self.remove_security(explist['hedge'], 'hedge')
        self.remove_security(explist['OTC'], 'OTC')

    def update_sec_by_month(self, added, flag, update=None):
        '''
        Helper method that updates the sec_by_month dictionary.

        Inputs:
        1) added  : boolean flag that indicates if securities are being added or removed. Valid inputs: True, False.
        2) flag   : string flag that indicates if the securities to be manipulated are hedge or OTC. Valid inputs: 'hedge', 'OTC'
        3) update : use update=True if the portfolio is being updated with new prices and vols

        Outputs : Updates OTC, hedges and net_greeks dictionaries.

        Notes: this method does 90% of all the heavy lifting in the portfolio class. Don't mess with this unless you know EXACTLY what each part is doing.
        '''
        # print('update sec by month flag: ', flag)
        if flag == 'OTC':
            dic = self.OTC
            op = self.OTC_options
            ft = self.OTC_futures
        elif flag == 'hedge':
            dic = self.hedges
            op = self.hedge_options
            ft = self.hedge_futures
        # adding/removing security to portfolio

        if update is None:
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
                    self.compute_net_greeks()

            # removing
            else:
                target = self.toberemoved.copy()
                self.toberemoved.clear()
                # print('target: ', [str(sec) for sec in target])
                for sec in target:
                    product = sec.get_product()
                    month = sec.get_month()
                    # print('removing ', product, month)
                    data = dic[product][month]
                    try:
                        if sec.get_desc() == 'option':
                            # print('removing ' + str(product) +
                            #       str(month) + ' option')
                            data[0].remove(sec)
                            # print('removed ' + str(sec))
                        else:
                            # print('removing ' + str(product) +
                            #       str(month) + ' future')
                            data[1].remove(sec)
                            # print('removed ' + str(sec))
                    except KeyError:
                        print(
                            "The security specified does not exist in this Portfolio")
                    # check for degenerate case when removing sec results in no
                    # securities associated to this product-month
                    if not(data[0]) and not(data[1]):
                        # print('removing ' + str(month) +
                        #       ' from ' + str(product))
                        dic[product].pop(month)
                        if len(dic[product]) == 0:
                            dic.pop(product)
                        self.compute_net_greeks()
                    else:
                        self.update_greeks_by_month(
                            product, month, sec, added, flag)
                        self.compute_net_greeks()

        # updating greeks per month when feeding in new prices/vols
        else:
            # updating cumulative greeks on a month-by-month basis.
            d3 = self.net_greeks

            # reset all greeks
            for product in dic:
                for month in dic[product]:
                    dic[product][month][2:] = [0, 0, 0, 0]
            # for product in d2:
            #     for month in d2[product]:
            #         d2[product][month][2:] = [0, 0, 0, 0]
            for product in d3:
                for month in d3[product]:
                    d3[product][month] = [0, 0, 0, 0]

            # recompute greeks for all months and products.
            self.recompute(ft, flag)
            self.recompute(op, flag)

            # recompute net greeks
            self.compute_net_greeks()

    def recompute(self, lst, flag):
        for sec in lst:
            pdt = sec.get_product()
            month = sec.get_month()
            self.update_greeks_by_month(pdt, month, sec, True, flag)

    def update_greeks_by_month(self, product, month, sec, added, flag):
        """Updates the greeks for each month. This method is called every time update_sec_by_month
        is called, and does the work of actually computing changes to monthly greeks in the self.OTC and self.hedges
        dictionaries.

        Args:
            product (str): The product to be updated.
            month (str): The month to be updated.
            sec (Security): the security that has been added/removed.
            added (boolean): boolean indicating if the security was added or removed.
            flag (str): OTC or hedge. indicates which part of the portfolio the security was added/removed to/from.

        Returns:
            None: Update data structures in place and calls compute_net_greeks. 
        """
        dic = self.OTC if flag == 'OTC' else self.hedges

        if not dic:
            # print(flag + ' dic missing')
            return

        if (product in dic) and (month in dic[product]):
            # print(flag + ' dic exists')
            data = dic[product][month]
        else:
            return

        # print('DEBUG - data: ', data[2:])

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
            # print('DEBUG II - data: ', data[2:])

        elif sec.get_desc() == 'future':
            delta = sec.delta
            # print('FT DELTA: ', delta)
            if added:
                data[2] += delta
            else:
                data[2] -= delta

    def compute_value(self):
        """Computes the value of this portfolio by summing across all securities contained within. Current computation takes (value of OTC positions - value of hedge positions)

        Returns:
            double: The value of this portfolio.
        """
        val = 0
        # try:
        for sec in self.OTC_options:
            lm = multipliers[sec.get_product()][1]
            dm = multipliers[sec.get_product()][0]
            if sec.shorted:
                val += -sec.lots * sec.get_price() * lm * dm
            else:
                val += sec.lots * sec.get_price() * lm * dm

        for sec in self.hedge_options:
            lm = multipliers[sec.get_product()][1]
            dm = multipliers[sec.get_product()][0]
            if sec.shorted:
                val += -sec.lots * sec.get_price() * lm * dm
            else:
                val += sec.lots * sec.get_price() * lm * dm

        for sec in self.OTC_futures:
            lm = multipliers[sec.get_product()][1]
            dm = multipliers[sec.get_product()][0]
            if sec.shorted:
                val += -sec.lots * sec.price * lm * dm
            else:
                val += sec.lots * sec.price * lm * dm

        for sec in self.hedge_futures:
            lm = multipliers[sec.get_product()][1]
            dm = multipliers[sec.get_product()][0]
            if sec.shorted:
                val -= sec.lots * sec.price * lm * dm
            else:
                val += sec.lots * sec.price * lm * dm

        return val

    def exercise_option(self, sec, flag):
        """Exercises an option if it is in-the-money. This consist of removing an object object from the relevant dictionary (i.e. either OTC- or hedge-pos), and adding a future to the relevant dictionary. Documentation for 'moneyness' can be found in classes.py in the Options class.

        Args:
            sec (Option) : the options object to be exercised.
            flag (str)   : indicating which if the option is currently held as a OTC or hedge. 

        Returns:
            None         : Updates data structures in-place, returns nothing.
        """
        toberemoved = []
        tobeadded = []
        if flag == 'OTC':
            op = self.OTC_options
        else:
            op = self.hedge_options
        for option in op:
            if option.moneyness() == 1:
                # convert into a future.
                underlying = option.get_underlying()
                underlying.update_lots(option.lots)
                tobeadded.append(underlying)
                toberemoved.append(option)
        self.remove_security(toberemoved, flag)
        self.add_security(tobeadded, flag)

    def greeks_by_exp(self, buckets):
        """Returns a dictionary of net greeks, organized by product and expiry. 

        Returns:
            dictionary: dictionary 
        """
        net = {}
        if self.empty():
            return
        t = timer()
        # preallocating
        # otc_by_exp = {}
        # hedges_by_exp = {}

        for comm in self.get_unique_products():
            # preallocating net
            if comm not in net:
                net[comm] = {}
            for div in buckets:
                if div not in net[comm]:
                    net[comm][div] = []
                net[comm][div] = [set(), 0, 0, 0, 0]

        else:
            otcs = self.OTC
            hedges = self.hedges
            # handle OTCs first then hedges
            for comm in otcs:
                for month in otcs[comm]:
                    options = otcs[comm][month][0]
                    for op in options:
                        # bucket based on tau
                        optau = float(op.tau * 365)
                        bucket = min([x
                                      for x in buckets if x > optau])
                        # if bucket not in otc_by_exp:
                        # otc_by_exp[bucket] = [0, 0, 0, 0]
                        d, g, t, v = op.greeks()
                        net[comm][bucket][0].add(op)
                        net[comm][bucket][1] += d
                        net[comm][bucket][2] += g
                        net[comm][bucket][3] += t
                        net[comm][bucket][4] += v

            for comm in hedges:
                for month in hedges[comm]:
                    options = hedges[comm][month][0]
                    for op in options:
                        # bucket based on tau
                        optau = float(op.tau * 365)
                        bucket = min(
                            [x for x in buckets if x > optau])
                        # if bucket not in hedges_by_exp:
                        #     hedges_by_exp[bucket] = []
                        d, g, t, v = op.greeks()
                        net[comm][bucket][0].add(op)
                        net[comm][bucket][1] += d
                        net[comm][bucket][2] += g
                        net[comm][bucket][3] += t
                        net[comm][bucket][4] += v
        return net


############### getter/utility methods #################

    def get_securities_monthly(self, flag):
        """Returns the position dictionary based on the flag passed in.

        Args:
            flag (str): if flag == OTC, returns OTC. otherwise, returns hedges

        Returns:
            dictionary: returns the relevant dictionary.
        """
        if flag == 'OTC':
            dic = self.OTC
        else:
            dic = self.hedges
        dic = dic.copy()
        return dic

    def get_securities(self, flag):
        if flag == 'OTC':
            op = self.OTC_options
            ft = self.OTC_futures
        else:
            op = self.hedge_options
            ft = self.hedge_futures
        lst1 = op.copy()
        lst2 = ft.copy()
        return (lst1, lst2)

    def get_all_options(self):
        return self.OTC_options + self.hedge_options

    def get_underlying(self):
        u_set = set()
        all_options = self.OTC_options + self.hedge_options
        for sec in all_options:
            u_set.add(sec.get_underlying())
        return list(u_set)

    def get_underlying_names(self):
        underlying = self.get_underlying()
        namelist = [ud.get_product() for ud in underlying]
        return set(namelist)

    def get_all_futures(self):
        retr = self.get_underlying()
        port_futures = self.OTC_futures + self.hedge_futures
        retr.extend(port_futures)
        return retr

    def timestep(self, value):
        all_options = self.get_all_options()
        for option in all_options:
            option.update_tau(value)

    def get_net_greeks(self):
        return self.net_greeks

    def get_all_future_names(self):
        all_futures = self.get_all_futures()
        names = [ft.get_product() for ft in all_futures]
        return set(names)

    def decrement_ordering(self, product, i):
        options = self.get_all_options()
        futures = self.get_all_futures()
        # options = self.OTC_options + self.hedge_options
        # futures = self.OTC_futures + self.hedge_futures
        for op in options:
            if op.get_product() == product:
                op.decrement_ordering(i)
        for ft in futures:
            if ft.get_product() == product:
                ft.decrement_ordering(i)

    def compute_ordering(self, product, month):
        if product in self.OTC and month in self.OTC[product]:
            dic = self.OTC
        elif product in self.hedges and month in self.hedges[product]:
            dic = self.hedges
        data = dic[product][month]
        if data[0]:
            s = next(iter(data[0]))
            order = s.get_ordering()
        elif data[1]:
            s = next(iter(data[1]))
            order = s.get_ordering()
        return order

    def net_vega_pos(self, month, pdt=None):
        all_ops = [op for op in self.get_all_options()
                   if op.get_month() == month]
        if pdt is not None:
            all_ops = [op for op in all_ops if op.get_product() == pdt]
        if not all_ops:
            return 0, 0
        call_op_vega = sum([op.vega for op in all_ops if op.char == 'call'])
        put_op_vega = sum([op.vega for op in all_ops if op.char == 'put'])

        return call_op_vega, put_op_vega

    def net_gamma_pos(self, month):
        all_ops = [op for op in self.get_all_options()
                   if op.get_month() == month]
        net_gamma = sum([op.gamma for op in all_ops])
        return net_gamma

    def get_volid_mappings(self):
        """Returns a dictionary mapping vol_id to lists of options currently in this portfolio. 
        """
        dic = {}

        # handle OTCs first
        for product in self.OTC:
            for mth in self.OTC[product]:
                options = self.OTC[product][mth][0]
                for op in options:
                    volid = product + '  ' + op.get_op_month() + '.' + mth
                    if volid not in dic:
                        dic[volid] = []
                    dic[volid].append(op)
                    # dic[volid] = []

        # now handle hedges
        for product in self.hedges:
            for mth in self.hedges[product]:
                options = self.hedges[product][mth][0]
                for op in options:
                    volid = product + '  ' + op.get_op_month() + '.' + mth
                    if volid not in dic:
                        dic[volid] = []
                    dic[volid].append(op)
                    # dic[volid] = []

        return dic

    def get_unique_products(self):
        """returns a list of all unique products in this portfolio
        """

        otc_products = list(self.OTC.keys())
        hedge_products = list(self.hedges.keys())

        return set(otc_products + hedge_products)
