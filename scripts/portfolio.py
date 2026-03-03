"""
File Name      : portfolio.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 17/4/2017
Python version : 3.5
Description    : Script contains implementation of the Portfolio class,
                 as well as helper methods that set/store/manipulate instance
                 variables. This class is used in simulation.py.

"""

from operator import add
import pprint
import numpy as np
from collections import deque
import copy
from .constants import multipliers, RANDOM_SEED, BREAKEVEN_FACTOR

np.random.seed(RANDOM_SEED)

# TODO: Abstract away reliance on greeks by month; should be able to
# accept any other convention as well.


class Portfolio:

    """
    Class representing the overall portfolio.

    All objects (i.e. Options or Futures) within the Portfolio object are 
    placed in one of two categories: OTC or Hedge. OTC securities are the 'stuff we have positions in', while
    hedge securities are the 'stuff we used to manage the risks of OTC securities'.

    Attributes: 

        -- Position Representation --
        hedge_futures (list): list of all hedge futures
        hedge_options (TYPE): list of all hedge options
        OTC_futures (list): list of all OTC futures
        OTC_options (list): list of all OTC options
        OTC (dict): dictionary containing OTC securities, withthe following format: 
                    Product -> Month -> [set(options), set(futures), delta, gamma, theta, vega]
        hedges (dict): dictionary containing hedge securities, withthe following format: 
                    Product -> Month -> [set(options), set(futures), delta, gamma, theta, vega]
        net_greeks (dict): dictionary containing net greek positions, with following format:
                    Product -> Month -> [delta, gamma, theta, vega] where each greek is OTC - Hedge


        -- Hedging Specification --
        hedge_params (dict): dictionary specifying how and what is to be hedged.
        hedger (TYPE): Hedge object that takes in hedge_params as a constructor, responsible
                       for implementing these instructions. 

        -- Other Structures -- 
        families (list): optional list of 'related' Portfolios. Allows for contract rolling/delta rolling to be 
                         applied across all options in all families, even as each portfolio is subject to its own
                         specific hedging conditions. 

        name (str): Optional name given to this portfolio. 
        newly_added (list): freshly-added securities. 
        toberemoved (list): securities marked for removal

        -- Other Parameters -- 
        roll (bool): True if contract rolling is to be applied, False otherwise. 
        roll_product (str): used to specify a product on the basis of which contract rolling happens.
        ttm_tol (float): threshold that, when breached, triggers a contract roll.
        value (float): value of this portfolio. 

    """

    def __init__(self, hedge_params, name=None, roll=None, roll_product=None, ttm_tol=None):

        self.OTC_options = deque()
        self.hedge_options = deque()
        self.OTC_futures = []
        self.hedge_futures = []
        self.roll = roll
        self.roll_product = roll_product
        self.ttm_tol = ttm_tol

        # utility litst
        self.newly_added = []
        self.toberemoved = []

        self.OTC = {}
        self.hedges = {}

        self.net_greeks = {}

        self.families = []

        self.name = str(name) if name is not None else None

        if hedge_params is None:
            self.hedge_params = {}

        if isinstance(hedge_params, dict):
            if self.name in hedge_params:
                self.hedge_params = hedge_params[self.name]
            else:
                self.hedge_params = hedge_params

        # updating initialized variables
        self.init_sec_by_month('OTC')
        self.init_sec_by_month('hedge')
        self.compute_net_greeks()
        self.value = self.compute_value()

        # assigning the hedger for this portfolio
        self.hedger = None

    def __str__(self):
        # custom print representation for this class.
        otcops = [op.__str__() for op in self.OTC_options]
        otcft = [op.__str__() for op in self.OTC_futures]
        hedgeops = [op.__str__() for op in self.hedge_options]
        nets = self.net_greeks
        ft_dic = {}
        for product in self.net_greeks:
            if product not in ft_dic:
                ft_dic[product] = {}
            for mth in self.net_greeks[product]:
                if mth not in ft_dic[product]:
                    ft_dic[product][mth] = []
                longs = sum([x.lots for x in self.OTC_futures + self.hedge_futures
                             if x.get_product() == product and 
                             x.get_month() == mth and 
                             not x.shorted])
                shorts = sum([x.lots for x in self.OTC_futures + self.hedge_futures
                              if x.get_product() == product and 
                              x.get_month() == mth and x.shorted])
                ft_dic[product][mth] = longs - shorts

        r_dict = {'OTC Options': otcops,
                  'OTC Futures': otcft,
                  'Hedge Options': hedgeops,
                  'Hedge Futures': ft_dic,
                  'Net Greeks': nets}

        return str(pprint.pformat(r_dict))

    def _get_position_lists(self, flag):
        """Returns (options_list, futures_list, positions_dict) for the given flag.

        Args:
            flag (str): 'OTC' or 'hedge'

        Returns:
            tuple: (options deque, futures list, positions dict)
        """
        if flag == 'OTC':
            return self.OTC_options, self.OTC_futures, self.OTC
        return self.hedge_options, self.hedge_futures, self.hedges

    def set_families(self, lst):
        self.families = lst

    def get_families(self):
        return self.families

    def refresh(self):
        for op in self.get_all_options():
            op.update()
        if not self.families:
            self.update_sec_by_month(None, 'OTC', update=True)
            self.update_sec_by_month(None, 'hedge', update=True)
        else:
            self.update_by_family()
        self.value = self.compute_value()

    def update_by_family(self):
        """Helper method that passes on a call to self.refresh() to all
        constituent families, returning a fully updated composite portfolio. 
        """
        from .util import combine_portfolios
        tmp = None
        if self.families:
            for f in self.families:
                f.refresh()
            tmp = combine_portfolios(
                self.families, hedges=self.hedge_params, name=self.name)

        if tmp is not None:
            otc_fts = self.OTC_futures
            hedge_fts = self.hedge_futures
            tmp.add_security(otc_fts, 'OTC')
            tmp.add_security(hedge_fts, 'hedge')
            self.OTC_futures.clear()
            self.hedge_futures.clear()

            self.OTC = tmp.OTC
            self.hedges = tmp.hedges
            self.OTC_options = tmp.OTC_options
            self.hedge_options = tmp.hedge_options
            self.OTC_futures = tmp.OTC_futures
            self.hedge_futures = tmp.hedge_futures
            self.compute_net_greeks()

    def get_family_containing(self, sec):
        """Returns the family containing the specified security.

        Args:
            sec (object): option or future object being searched for. 

        Returns:
            object/None: returns the family containing this security if it exists, None otherwise. 
        """
        if not self.families:
            return None
        else:
            for f in self.families:
                if sec in f.get_all_options():
                    return f
            return None

    def update_sec_lots(self, sec, flag, lots):
        """Updates the lots of a given security, updates the dictionary it is contained in, and
        updates net_greeks

        Args:
            sec (list): list of securities whose lots are being updated
            flag (TYPE): indicates if this security is an OTC or hedge option
            lots (TYPE): list of lot values, where lots[i] corresponds
                        to the new lot value of sec[i]
        """
        ops = self.OTC_options if flag == 'OTC' else self.hedge_options
        fts = self.OTC_futures if flag == 'OTC' else self.hedge_futures
        for s in sec:
            # sanity checks: make sure the security is present in the relevant
            # list selected by flag
            if s.desc == 'option' and s not in ops:
                raise ValueError('This option is not in the portfolio.')
            elif s.desc == 'future' and s not in fts:
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

    def ops_empty(self):
        """Checks to see if there are any OTC options left; called in rebalance in simulation.
        """
        return len(self.OTC_options) == 0

    def init_sec_by_month(self, iden):
        """Initializing method that creates the relevant futures list,
            options list, and dictionary depending on the flag passed in.

        Args:
            iden (str): flag that indicates which set of data structures is to be initialized.
            Valid Inputs: 'OTC', 'hedge' to initialize OTC and hedge positions respectively.

        Returns:
            None: Initializes the relevant data structures.
        """
        op, ft, dic = self._get_position_lists(iden)

        # add in options
        for sec in op:
            month = sec.get_month()
            prod = sec.get_product()
            if prod not in dic:
                dic[prod] = {}
            if month not in dic[prod]:
                dic[prod][month] = [set([sec]), set(), 0, 0, 0, 0]
            else:
                dic[prod][month][0].add(sec)
            self.update_greeks_by_month(prod, month, sec, True, iden)
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
        """Computes net greeks organized by product and month.
        Net greeks = OTC greeks + hedge greeks for each product-month."""

        final_dic = {}
        all_products = set(self.OTC.keys()) | set(self.hedges.keys())

        for product in all_products:
            final_dic[product] = {}
            otc_data = self.OTC.get(product, {})
            hedge_data = self.hedges.get(product, {})
            all_months = set(otc_data.keys()) | set(hedge_data.keys())

            for month in all_months:
                otc_greeks = otc_data[month][2:] if month in otc_data else [0, 0, 0, 0]
                hedge_greeks = hedge_data[month][2:] if month in hedge_data else [0, 0, 0, 0]
                # only include month if it has options in at least one side
                has_otc_options = month in otc_data and otc_data[month][0]
                has_hedge_options = month in hedge_data and hedge_data[month][0]
                if has_otc_options or has_hedge_options:
                    final_dic[product][month] = list(map(add, otc_greeks, hedge_greeks))

            # remove empty products
            if not final_dic[product]:
                del final_dic[product]

        self.net_greeks = final_dic

    def add_security(self, security, flag):
        # adds a security into the portfolio, and updates relevant lists and
        # adjusts greeks of the portfolio.

        op, ft, _ = self._get_position_lists(flag)

        for sec in security:
            if sec.get_desc() == 'option':
                op.append(sec)
            elif sec.get_desc() == 'future':
                ft.append(sec)

        self.newly_added.extend(security)
        self.update_sec_by_month(True, flag)

    def remove_security(self, security, flag):
        # removes a security from the portfolio, updates relevant list and
        # adjusts greeks of the portfolio.
        op, ft, _ = self._get_position_lists(flag)
        self.toberemoved.extend(security)
        self.update_sec_by_month(False, flag)
        for sec in security:
            if sec.get_desc() == 'option':
                op.remove(sec)
            elif sec.get_desc() == 'future':
                ft.remove(sec)
        if self.families:
            for sec in security:
                if sec.get_desc() == 'option':
                    fpf = self.get_family_containing(sec)
                    if fpf is not None:
                        print('fpf.name: ', fpf.name)
                        print('fpf:', fpf.OTC)
                        print('fpf.OTC_options: ', fpf.OTC_options)
                        # case: option was removed from family.OTC during initial
                        # call to remove_security. happens because option set is passed
                        # by reference
                        if sec not in fpf.OTC[sec.get_product()][sec.get_month()][0]:
                            print('reg case hit')
                            fpf.OTC_options.remove(sec)
                        else:
                            # pretty sure this never gets triggered.
                            print('remove alt case hit')
                            fpf.remove_security([sec], flag)
                        fpf.refresh()

    def remove_expired(self):
        '''Removes all expired options from the portfolio. '''
        explist = {'hedge': [], 'OTC': []}
        for sec in self.OTC_options:
            # handling barrier case
            if sec.barrier == 'amer':
                if sec.knockedout:
                    explist['OTC'].append(sec)
            # vanilla/knockin case
            if sec.check_expired():
                explist['OTC'].append(sec)

            # case: daily options that have 0 in their ttm lists.
            if not sec.is_bullet():
                sec.remove_expired_dailies()
                
        for sec in self.hedge_options:
            # handling barrier case.
            if sec.barrier == 'amer':
                if sec.knockedout:
                    explist['hedge'].append(sec)

            elif sec.check_expired():
                explist['hedge'].append(sec)

        self.remove_security(explist['hedge'], 'hedge')
        self.remove_security(explist['OTC'], 'OTC')

    def update_sec_by_month(self, added, flag, update=None):
        '''
        Helper method that updates the sec_by_month dictionary.

        Inputs:
        1) added  : boolean flag that indicates if securities are being
                    added or removed. Valid inputs: True, False.
        2) flag   : string flag that indicates if the securities to be manipulated
                    are hedge or OTC. Valid inputs: 'hedge', 'OTC'
        3) update : use update=True if the portfolio is being updated with new prices and vols

        Outputs : Updates OTC, hedges and net_greeks dictionaries.

        Notes: this method does 90% of all the heavy lifting in the portfolio
            class. Don't mess with this unless you know EXACTLY what each part is doing.
        '''
        op, ft, dic = self._get_position_lists(flag)
        other = self.hedges if flag == 'OTC' else self.OTC

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
                for sec in target:
                    product = sec.get_product()
                    month = sec.get_month()
                    try:
                        data = dic[product][month]
                        if sec.get_desc() == 'option':
                            data[0].remove(sec)
                        else:
                            data[1].remove(sec)
                    except (KeyError, ValueError):
                        errorstr = str(sec) + " does not exist in Portfolio "
                        errorstr += self.name if self.name is not None else ''
                        raise ValueError(errorstr)
                        # return -1
                    # check for degenerate case when removing sec results in no
                    # securities associated to this product-month
                    ops = data[0]
                    fts = data[1]
                    other_op = other[product][month][0] if (
                        (product in other) and (month in other[product])) else None
                    other_ft = other[product][month][1] if (
                        (product in other) and (month in other[product])) else None

                    if (not ops) and (not fts) and (not other_op) and (not other_ft):
                        # print('degenerate case hit')
                        dic[product].pop(month)
                        # sanity check: pop product if no months left.
                        if len(dic[product]) == 0:
                            dic.pop(product)
                        # sanity check: other might be empty.
                        if product not in other:
                            pass
                        else:
                            if (other is not None) and \
                                    (product in other) and \
                                    (month in other[product]):
                                other[product].pop(month)
                            if len(other[product]) == 0:
                                other.pop(product)
                        self.compute_net_greeks()
                    else:
                        self.update_greeks_by_month(
                            product, month, sec, added, flag)
                        self.compute_net_greeks()

        # updating greeks per month when feeding in new prices/vols
        else:
            # updating cumulative greeks on a month-by-month basis.
            d3 = self.net_greeks

            # error-checking: remove pathological cases.
            toberemoved = {}
            # iterate once through, identify products/months that need to be
            # popped.
            for product in dic.copy():
                toberemoved[product] = []
                if dic[product] == {}:
                    toberemoved[product] = 'all'
                # iterate through the months in this product.
                for month in dic.copy()[product]:
                    if dic[product][month] == {} or dic[product][month][: 2] == [set(), set()]:
                        toberemoved[product].append(month)
                    # case: removing the last month in this product.
                    if toberemoved[product]:
                        if len(toberemoved[product]) == len(dic[product]):
                            toberemoved[product] = 'all'

            # iterate through once more and delete all entries marked for
            # removal.
            for product in toberemoved:
                if toberemoved[product] == 'all':
                    dic.pop(product)
                else:
                    for month in toberemoved[product]:
                        dic[product].pop(month)

            # reset all greeks
            for product in dic.copy():
                for month in dic[product]:
                    dic[product][month][2:] = [0, 0, 0, 0]

            for product in d3:
                for month in d3[product]:
                    d3[product][month] = [0, 0, 0, 0]

            # recompute greeks for all months and products.
            self.recompute(ft, flag)
            self.recompute(op, flag)

            # recompute net greeks
            self.compute_net_greeks()

    def recompute(self, lst, flag):
        """Helper method called in update_sec_by_month that recomputes the greeks of the 
        list of securities specified. 

        Args:
            lst (list): list of securities
            flag (str): OTC or hedge
        """
        for sec in lst:
            pdt = sec.get_product()
            month = sec.get_month()
            self.update_greeks_by_month(pdt, month, sec, True, flag)

    def update_greeks_by_month(self, product, month, sec, added, flag):
        """Updates the greeks for each month. This method is called every time update_sec_by_month
        is called, and does the work of actually computing changes to monthly greeks in the self.OTC
        and self.hedges dictionaries.

        Args:
            product (str): The product to be updated.
            month (str): The month to be updated.
            sec (Security): the security that has been added/removed.
            added (boolean): boolean indicating if the security was added or removed.
            flag (str): OTC or hedge. indicates which part of the portfolio
                        the security was added/removed to/from.

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
        """Computes the value of this portfolio by summing across all securities
        contained within. Current computation takes
        (value of OTC positions - value of hedge positions)

        Returns:
            double: The value of this portfolio.
        """
        val = 0
        # try:
        for sec in self.get_all_securities():
            pnl_mult = multipliers[sec.get_product()][-1]
            if sec.shorted:
                val += -sec.lots * sec.get_price() * pnl_mult
            else:
                val += sec.lots * sec.get_price() * pnl_mult

        return val

    def get_all_securities(self):
        return list(self.OTC_options) + list(self.hedge_options) + self.OTC_futures + self.hedge_futures

    def exercise_option(self, sec, flag):
        """Exercises an option if it is in-the-money. This consist of removing an object object
        from the relevant dictionary (i.e. either OTC- or hedge-pos), and adding a future to the
        relevant dictionary. Documentation for 'moneyness' can be found in classes.py in the Options class.

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

    def get_aggregated_greeks(self):
        dic = self.get_net_greeks().copy() 
        agg = {}
        for pdt in dic:
            if pdt not in agg:
                agg[pdt] = [0,0,0,0]
                for mth in dic[pdt]:
                    agg[pdt][0] += dic[pdt][mth][0]
                    agg[pdt][1] += dic[pdt][mth][1] 
                    agg[pdt][2] += dic[pdt][mth][2] 
                    agg[pdt][3] += dic[pdt][mth][3]
        return agg  
            

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
        op, ft, _ = self._get_position_lists(flag)
        return (op.copy(), ft.copy())

    def get_all_options(self, pdt=None, mth=None):
        """Gets all options, hedge and OTC

        Args:
            pdt (str, optional): Optional filter - get all options of this product. 
            mth (str, optional): Optional filter - get all options of this month

        Returns:
            list: list of options that satisfy the filter conditions. 
        """
        lst = self.OTC_options + self.hedge_options
        if pdt is not None:
            lst = [x for x in lst if x.get_product() == pdt]
        if mth is not None:
            lst = [x for x in lst if x.get_month() == mth]
        return lst

    def get_hedge_options(self, pdt=None, mth=None):
        """Returns hedge options. If pdt and month are specified, these
        are used to select and return the options.
        
        Args:
            pdt (None, optional): filters for this product. 
            mth (None, optional): filters for this month.
        """
        lst = self.hedge_options 
        if pdt is not None:
            lst = [x for x in lst if x.get_product() == pdt]
        if mth is not None:
            lst = [x for x in lst if x.get_month() == mth]
        return lst 

    def get_underlying(self):
        """Returns a list of all futures objects that are the underlying
        of some option in the portfolio. 

        Returns:
            list: list of future objects. 
        """
        u_set = set()
        all_options = self.OTC_options + self.hedge_options
        for sec in all_options:
            u_set.add(sec.get_underlying())
        return list(u_set)

    def get_underlying_names(self):
        """gets all products for which underlying futures exist.

        Returns:
            list: list of names. 
        """
        underlying = self.get_underlying()
        namelist = [ud.get_product() for ud in underlying]
        return set(namelist)

    def get_all_futures(self):
        """Returns a list of all future objects in the portfolio. 

        """
        retr = self.get_underlying()
        port_futures = self.OTC_futures + self.hedge_futures
        retr.extend(port_futures)
        return retr

    def get_pos_futures(self):
        """Gets all future objects that are NOT underlying objects
        of some option in the portfolio. 

        Returns:
            TYPE: Description
        """
        return self.OTC_futures + self.hedge_futures

    def timestep(self, value, allops=True, ops=False):
        """Timesteps the entire portfolio _value_ days into the future. 

        Args:
            value (int): number of days to timestep by.
            allops (bool, optional): True if all options are to be timestepped, False if only OTC options. 
            ops (list, optional): list of options we want to timestep.
        """
        if ops:
            all_options = ops
        else:
            if allops:
                all_options = self.get_all_options()
            else:
                all_options = self.OTC_options
        for option in all_options:
            option.update_tau(value)

        self.refresh()

    def get_net_greeks(self):
        return self.net_greeks

    def get_all_future_names(self):
        """Returns all products for which we have futures in the portfolio. 

        Returns:
            list: list of products. 
        """
        all_futures = self.get_all_futures()
        names = [ft.get_product() for ft in all_futures]
        return set(names)

    def decrement_ordering(self, product, i):
        """Decrements the ordering (C1, C2 etc) of 
        each option and future on the specified product. 

        Args:
            product (str): product we're decrementing
            i (int): the value to decrement it by. 
        """
        options = self.get_all_options()
        futures = self.get_all_futures()
        for op in options:
            if op.get_product() == product:
                op.decrement_ordering(i)
        for ft in futures:
            if ft.get_product() == product:
                ft.decrement_ordering(i)

    def compute_ordering(self, product, month):
        """Gets the order of a given product-month combination
        by looking up the computed order of a future in that
        category. 

        Args:
            product (str)
            month (str)

        Returns:
            int: the order.
        """
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
        """ Returns the net vega contribution from calls and puts individually of all
        options satisfying the filtering critera specified. 

        Args:
            month (str): month we're interested in. 
            pdt (str, optional): product we're interested in.

        Returns:
            tuple: call vega and put vega
        """
        all_ops = [op for op in self.get_all_options()
                   if op.get_month() == month]
        if pdt is not None:
            all_ops = [op for op in all_ops if op.get_product() == pdt]
        if not all_ops:
            return 0, 0
        call_op_vega = sum([op.vega for op in all_ops if op.option_type =='call'])
        put_op_vega = sum([op.vega for op in all_ops if op.option_type =='put'])

        return call_op_vega, put_op_vega

    def net_gamma_pos(self, month, pdt=None):
        """Same as net_vega_pos but for gamma. 

        Args:
            month (TYPE): Description
            pdt (None, optional): Description

        Returns:
            TYPE: Description
        """
        all_ops = [op for op in self.get_all_options()
                   if op.get_month() == month]
        if pdt is not None:
            all_ops = [x for x in all_ops if x.get_product() == pdt]
        if not all_ops:
            return 0
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

        # now handle hedges
        for product in self.hedges:
            for mth in self.hedges[product]:
                options = self.hedges[product][mth][0]
                for op in options:
                    volid = product + '  ' + op.get_op_month() + '.' + mth
                    if volid not in dic:
                        dic[volid] = []
                    dic[volid].append(op)

        return dic

    def get_unique_products(self):
        """returns a list of all unique products in this portfolio
        """

        otc_products = list(self.OTC.keys())
        hedge_products = list(self.hedges.keys())

        return set(otc_products + hedge_products)

    def get_unique_volids(self):
        """Returns a set of all unique vol_ids in the portfolio. 
        """
        allops = self.get_all_options()
        return set([x.get_vol_id() for x in allops])

    def breakeven(self, flag=None, conv=None):
        """Returns a dictionary of {pdt: {month: breakeven}} where breakeven is calculated by theta/gamma. 
        """
        bes = {}
        dic = self.net_greeks
        for pdt in dic:
            bes[pdt] = {}
            thetas = []
            gammas = []
            for mth in dic[pdt]:
                gamma, theta = abs(dic[pdt][mth][1]), abs(dic[pdt][mth][2])
                if np.isclose(gamma, 0) or np.isclose(theta, 0):
                    bes[pdt][mth] = 0 
                else:
                    thetas.append(theta)
                    gammas.append(gamma)
                    bes[pdt][mth] = (((BREAKEVEN_FACTOR*theta)/gamma) ** 0.5) / \
                        multipliers[pdt][0]

        return bes

    def assign_hedger_dataframes(self, vdf, pdf, settles=None):
        """Helper method that updates the dataframes
        present in this portfolio's hedger object. 

        Args:
            vdf (dataframe): Dataframe of volatilities
            pdf (dataframe): Dataframe of prices

        """
        if self.hedger is not None:
            self.hedger.update_dataframes(vdf, pdf, settles=settles)
        if self.families:
            for fa in self.families:
                fa.hedger.update_dataframes(vdf, pdf, settles=settles)

    def get_hedger(self):
        return self.hedger

    def get_hedgeparser(self, dup=False):
        return self.hedger.get_hedgeparser() if dup is False \
            else copy.deepcopy(self.hedger.get_hedgeparser())

    def get_unique_uids(self):
        """Helper method that returns a set of the unique underlyings currently in the portfolio. 
        Used primarily to maintain the dictionary of changes in the timestamp-loop in simulation.run_simulation. 
        """
        ret = set([x.get_uid() for x in self.get_all_options()])
        return ret

    def get_uid_price(self, uid):
        """Returns the underlying price of the specified underlying_id, e.g. C Z7.

        Args:
            uid (str): ID associated with a future, e.g. C Z7.

        Returns:
            float: price of said future. 
        """
        fts = [x for x in self.get_all_futures() if x.get_uid() == uid]
        return fts[0].get_price()

    def uid_price_dict(self):
        """Helper method that returns a dictionary of uid -> price. 
        """
        ret = {}
        for x in self.get_all_futures():
            if x.get_uid() in ret:
                continue
            ret[x.get_uid()] = x.get_price()

        return ret

    def update_hedger_breakeven(self):
        """Proxy method that calls this portfolio's hedger objects' set_breakeven method. 
        """
        if self.hedger is not None:
            self.hedger.set_breakeven(self.breakeven())
