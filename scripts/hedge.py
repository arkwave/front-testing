# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-07-20 18:26:26
# @Last Modified by:   arkwave
# @Last Modified time: 2017-07-25 15:43:11

from timeit import default_timer as timer
import numpy as np


class Hedge:
    """Class that defines a hedge object. Hedge object has two main functionalities:
    1) inferring the hedging parameters from the portfolio
    2) applying the relevant hedges to the portfolio.

    Inference of hedging parameters is done via the _calibrate method, while applying hedges is done via the _apply method.
    """

    def __init__(self, portfolio, hedges, pdf, desc, buckets=None, **kwargs):
        """Constructor. Initializes a hedge object subject to the following parameters. 

        Args:
            portfolio (TYPE): the portfolio being hedged
            vdf (TYPE): dataframe of volatilites 
            pdf (TYPE): Description
            view (TYPE): a string description of the greek representation. valid inputs are 'exp' for greeks-by-expiry and 'uid' for greeks by underlying.
            buckets (None, optional): Description
        """
        self.params = kwargs
        # self.flag = flag
        self.mappings = {}
        self.desc = desc
        self.greek_repr = {}
        self.pdf = pdf
        self.pf = portfolio
        self.buckets = buckets if buckets is not None else [0, 30, 60, 90, 120]
        self.hedges = hedges
        self.done = self.satisfied(self.pf)

        # self.params, self.greek_repr = None, None

    def _calibrate(self, flag, selection_criteria='median', buckets=None):
        """Helper method that constructs the hedging parameters based on the greek representation fed into the hedge object.

        Example (1): flag == 'exp' indicates that hedging is being done basis greeks bucketed according to time to maturity. 
        As such, the parameter dictionary generated is a dictionary mapping commodity and ttm to vol_id used to hedge that particular dictionary. 

        Example (2): flag == 'uid' indicates that hedging is being done basis greeks clubbed according to underlying contract, e.g. the greeks for a W Q6.U6 and W U6.U6 will be added together and
        hedged 


        Args:
            hedges (dictionary): dictionary of hedging logic passed into the simulation. 
            flag (TYPE): 


        Note: this does NOT check if greeks other than delta are being hedged, because delta by default is 
        hedged at the EOD by commodity/month, not on the basis of expiries. I.e. you will never use a W Q6 underlying 
        to hedge the deltas from a W Q6.U6 option. 

        """
        net = {}
        ttm = None

        # first case: greek by expiry.
        if self.desc == 'exp':
            calibration_dic = self.pf.greeks_by_exp(
                self.buckets) if buckets is None else self.pf.greeks_by_exp(buckets)

            net = calibration_dic

            for product in calibration_dic:
                df = self.pdf[self.pdf.pdt == product]
                for exp in calibration_dic[product]:
                    options = calibration_dic[product][exp][0]

                    # case: no options associated with this product/exp bucket
                    if not options:
                        continue

                    loc = (product, exp)

                    # select relevant hedge conditions based on flag passed in.
                    relevant_hedges = self.hedges[flag][0]

                    # case: ttm specification exists.
                    if len(relevant_hedges) == 4:
                        ttm_modifier = relevant_hedges[3]

                        # case: fixed value of ttm used to hedge this greek.
                        if ttm_modifier >= 1:
                            ttm = ttm_modifier

                        # case: proportion of current ttm passed in.
                        else:
                            print('selection_criteria: ', selection_criteria)
                            if selection_criteria == 'median':
                                ttm = np.median([op.tau for op in options])
                                print('ttm: ', ttm)
                            ttm = ttm * ttm_modifier

                    # check available vol_ids and pick the closest one.
                    closest_tau_val = min(df.tau, key=lambda x: abs(x - ttm))

                    vol_ids = df[df.tau == closest_tau_val].vol_id.values

                    print('vol_ids: ', vol_ids)
                    # select the closest opmth/ftmth combination
                    split = [x.split()[1] for x in vol_ids]
                    # sort by ft_year, ft_month and op_month
                    split = sorted(split, key=lambda y:
                                   (int(y[4]), y[3], y[0]))

                    volid = product + '  ' + split[0]

                    # assign to parameter dictionary.
                    if flag not in self.mappings:
                        self.mappings[flag] = {}

                    self.mappings[flag][loc] = volid

        # second case: greeks by underlying (regular thing we're used to)
        elif self.desc == 'uid':
            calibration_dic = self.pf.get_net_greeks()
            net = calibration_dic
            for product in net:
                df = self.pdf[self.pdf.pdt == product]
                for month in net[product]:
                    data = net[product][month]
                    if not data or (data == [0, 0, 0, 0]):
                        continue

                    loc = (product, month)

                    relevant_hedges = self.hedges[flag][0]

                    # grab max possible ttm (i.e. ttm of the same month option)

                    try:
                        volid = product + '  ' + month + '.' + month
                        max_ttm = df[(df.vol_id == volid) & (
                            df.call_put_id == 'C')].tau.values[0]
                    except IndexError:
                        print('hedge.uid_volid: cannot find max ttm')
                        print('debug 1: ', df[(df.vol_id == volid)])
                        print('debug 2: ', df[(df.call_put_id == 'C')])
                        print('debug 3: ', df[
                              (df.vol_id == volid) & (df.call_put_id == 'C')])

                    # case: ttm specification exists.
                    if len(relevant_hedges) == 4:
                        ttm_modifier = relevant_hedges[3]
                        # case: fixed value of ttm used to hedge this greek.
                        if ttm_modifier >= 1:
                            ttm = ttm_modifier / 365
                        else:
                            ttm = max_ttm * ttm_modifier

                    closest_tau_val = min(df.tau, key=lambda x: abs(x - ttm))

                    uid = product + '  ' + month

                    vol_ids = df[(df.tau == closest_tau_val) & (
                        df.underlying_id == uid)].vol_id.values

                    print('vol_ids: ', vol_ids)

                    # select the closest opmth/ftmth combination
                    split = [x.split()[1] for x in vol_ids]
                    # sort by ft_year, ft_month and op_month
                    split = sorted(split, key=lambda y:
                                   (int(y[4]), y[3], y[0]))

                    volid = product + '  ' + split[0]

                    # assign to parameter dictionary.
                    if flag not in self.mappings:
                        self.mappings[flag] = {}

                    self.mappings[flag][loc] = volid

        # self.mappings = params
        # if flag not in self.greek_repr:
        #     self.greek_repr[flag] = {}
        self.greek_repr = net
        # print('Hedge.calibrate - elapsed: ', timer() - t)

    def get_bucket(self, val, buckets=None):
        """Helper method that gets the bucket associated with a given value according to self.buckets. 

        Args:
            val (TYPE): Description
            buckets (TYPE, optional): Description
        """
        buckets = self.buckets if buckets is None else buckets
        return min([x for x in buckets if x > val])

    def satisfied(self, pf):
        """Helper method that delegates checks if the hedge conditions are satisfied 

        Args:
            pf (object): The portfolio object being hedged 
        """
        if self.desc == 'uid':
            return self.uid_hedges_satisfied(pf, self.hedges)

        elif self.desc == 'exp':
            return self.exp_hedges_satisfied(pf, self.hedges)

    def uid_hedges_satisfied(self, pf, hedges):
        """Helper method that ascertains if all entries in net_greeks satisfy the conditions laid out in hedges.

        Args:
            pf (portfolio object): portfolio being hedged
            hedges (ordered dictionary): contains hedge information/specifications

        Returns:
            Boolean: indicating if the hedges are all satisfied or not.
        """
        strs = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 3}
        tst = hedges.copy()
        if 'delta' in tst:
            tst.pop('delta')

        net_greeks = pf.get_net_greeks()
        # delta condition:
        conditions = []
        for greek in tst:
            conds = tst[greek]
            for cond in conds:
                # static bound case
                if cond[0] == 'static':
                    conditions.append((strs[greek], (-1, 1)))
                elif cond[0] == 'bound':
                    # print('to be literal eval-ed: ', hedges[greek][1])
                    c = cond[1]
                    tup = (strs[greek], c)
                    conditions.append(tup)
        # bound_and_static = True
        for pdt in net_greeks:
            for month in net_greeks[pdt]:
                greeks = net_greeks[pdt][month]
                for cond in conditions:
                    bound = cond[1]
                    if (greeks[cond[0]] > bound[1]) or (greeks[cond[0]] < bound[0]):
                        return False
        # rolls_satisfied = check_roll_hedges(pf, hedges)
        return True

    def exp_hedges_satisfied(self, pf, hedges):
        """Helper method that checks if greeks according to expiry
             representation are adequately hedged. 

        Args:
            pf (TYPE): Description
            hedges (TYPE): Description
        """
        strs = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 3}
        net_greeks = pf.greeks_by_exp(self.buckets)
        tst = hedges.copy()
        if 'delta' in tst:
            tst.pop('delta')

        # delta condition:
        conditions = []
        for greek in tst:
            conds = tst[greek]
            for cond in conds:
                # static bound case
                if cond[0] == 'static':
                    conditions.append((strs[greek], (-1, 1)))
                elif cond[0] == 'bound':
                    # print('to be literal eval-ed: ', hedges[greek][1])
                    c = cond[1]
                    tup = (strs[greek], c)
                    conditions.append(tup)
        print('conditions: ', conditions)

        for pdt in net_greeks:
            for exp in net_greeks[pdt]:
                greeks = net_greeks[pdt][exp][1:]
                for cond in conditions:
                    bound = cond[1]
                    if (greeks[cond[0]] > bound[1]) or (greeks[cond[0]] < bound[0]):
                        return False
        # rolls_satisfied = check_roll_hedges(pf, hedges)
        return True

    def refresh(self):
        """Helper method that re-inializes all values. To be used when updating portfolio to ascertain hedges have been satisfied. 
        """
        for flag in self.hedges:
            if flag != 'delta':
                print('refreshing ' + flag)
                self._calibrate(flag)

        self.done = self.satisfied(self.pf)
