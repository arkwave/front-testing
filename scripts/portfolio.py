"""
File Name      : portfolio.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 23/3/2017
Python version : 3.5
Description    : Script contains implementation of the Portfolio class, as well as helper methods that set/store/manipulate instance variables. This class is used in simulation.py.

"""


from operator import sub


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

# TODO: differentiating into hedges and target.

    def __init__(self):

        self.OTC_options = []
        self.hedge_options = []
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

    def set_pnl(self, pnl):
        self.pnl = pnl

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
            # print(OTC_unique_mths)
            hedges_unique_mths = set(
                hedgedata.keys()) - common_months
            # dealing with common months
            for month in common_months:
                OTC_greeks = OTCdata[month][2:]
                hedge_greeks = hedgedata[month][2:]
                net = list(map(sub, OTC_greeks, hedge_greeks))
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
        if security.get_desc() == 'option':
            try:
                op.append(security)
            except UnboundLocalError:
                print('flag: ', flag)
        elif security.get_desc() == 'future':
            ft.append(security)
        self.newly_added.append(security)
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
        try:
            if security.get_desc() == 'option':
                op.remove(security)
            elif security.get_desc() == 'future':
                ft.remove(security)
            self.toberemoved.append(security)
            self.update_sec_by_month(False, flag)
        except ValueError:
            return -1

    def remove_expired(self):
        '''Removes all expired options from the portfolio. '''
        for sec in self.OTC_options:
            # handling barrier case
            if sec.barrier == 'amer':
                if sec.knockedout:
                    self.remove_security(sec, 'OTC')
            # vanilla/knockin case
            if sec.tau == 0:
                self.remove_security(sec, 'OTC')
        for sec in self.hedge_options:
            # handling barrier case.
            if sec.barrier == 'amer':
                if sec.knockedout:
                    self.remove_security(sec, 'hedge')
            # vanilla/knockin case
            if sec.tau == 0:
                self.remove_security(sec, 'hedge')

    def update_sec_by_month(self, added, flag, price=None, vol=None):
        '''
        Helper method that updates the sec_by_month dictionary.

        Inputs:
        1) added  : boolean flag that indicates if securities are being added or removed. Valid inputs: True, False.
        2) flag   : string flag that indicates if the securities to be manipulated are hedge or OTC. Valid inputs: 'hedge', 'OTC'
        3) price  : new price of underlying. If explicitly passed in, the method assumes that new price/vol data is being fed into the portfolio, and acts accordingly. defaults to None.
        4) vol    : new vol. If explicitly passed in, the method assumes that new price/vol data is being fed into the portfolio, and acts accordingly. defaults to None.

        Outputs : Updates OTC, hedges and net_greeks dictionaries.

        Notes: this method does 90% of all the heavy lifting in the portfolio class. Don't mess with this unless you know EXACTLY what each part is doing.
        '''
        if flag == 'OTC':
            dic = self.OTC
            op = self.OTC_options
            ft = self.OTC_futures
        elif flag == 'hedge':
            dic = self.hedges
            op = self.hedge_options
            ft = self.hedge_futures
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
                    data = dic[product][month]
                    try:
                        if sec.get_desc() == 'option':
                            data[0].remove(sec)
                        else:
                            data[1].remove(sec)
                    except KeyError:
                        print(
                            "The security specified does not exist in this Portfolio")
                    # check for degenerate case when removing sec results in no
                    # securities associated to this product-month
                    if not(data[0]) and not(data[1]):
                        dic[product].pop(month)
                        return self.compute_net_greeks()
                    else:
                        self.update_greeks_by_month(
                            product, month, sec, added, flag)

        # updating greeks per month when feeding in new prices/vols
        else:
            # updating greeks based on new price and vol data
            for sec in op:
                sec.underlying.update_price(price)
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

        if flag == 'OTC':
            dic = self.OTC

        else:
            dic = self.hedges
        data = dic[product][month]

        if sec.get_desc() == 'option':
            delta, gamma, theta, vega = sec.greeks()

            if (added and not sec.shorted) or (not added and sec.shorted):
                # adding a long option or removing a short option.
                data[2] += delta
                data[3] += gamma
                data[4] += theta
                data[5] += vega

            elif (added and sec.shorted) or (not added and not sec.shorted):
                # adding a short option or removing a long option.
                data[2] -= delta
                data[3] -= gamma
                data[4] -= theta
                data[5] -= vega

            self.compute_net_greeks()

    def compute_value(self):
        """Computes the value of this portfolio by summing across all securities contained within. Current computation takes (value of OTC positions - value of hedge positions)

        Returns:
            double: The value of this portfolio.
        """
        val = 0
        # try:
        for sec in self.OTC_options:
            val += sec.get_price()
        for sec in self.hedge_options:
            val -= sec.get_price()
        for sec in self.OTC_futures:
            val += sec.get_price()
        for sec in self.hedge_futures:
            val -= sec.get_price()
        # except
        return val

    def exercise_option(self, sec, flag):
        """Exercises an option if it is in-the-money. This consist of removing an object object from the relevant dictionary (i.e. either OTC- or hedge-pos), and adding a future to the relevant dictionary. Documentation for 'moneyness' can be found in classes.py in the Options class.

        Args:
            sec (Option) : the options object to be exercised.
            flag (str)   : indicating which if the option is currently held as a OTC or hedge. 

        Returns:
            None         : Updates data structures in-place, returns nothing.
        """
        if flag == 'OTC':
            op = self.OTC_options
        else:
            op = self.hedge_options
        for option in op:
            if option.moneyness() == 1:
                # convert into a future.
                underlying = option.get_underlying()
                self.remove_security(option, flag)
                self.add_security(underlying, flag)

 ### getter/utility methods ###

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
        all_options = self.OTC_options + self.hedge_options
        for option in all_options:
            option.update_tau(value)

    def get_net_greeks(self):
        return self.net_greeks

    def get_all_future_names(self):
        all_futures = self.get_all_futures()
        names = [ft.get_product() for ft in all_futures]
        return set(names)
