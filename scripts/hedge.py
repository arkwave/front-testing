# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-07-20 18:26:26
# @Last Modified by:   Ananth
# @Last Modified time: 2017-10-04 22:10:39

import pandas as pd
import pprint
import numpy as np
from .util import create_straddle, create_underlying, create_strangle, create_vanilla_option
from .calc import _compute_value

multipliers = {
    'LH':  [22.046, 18.143881, 0.025, 0.05, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'QC': [1.2153, 10, 1, 25, 12.153],
    'SB':  [22.046, 50.802867, 0.01, 0.25, 1120],
    'CC':  [1, 10, 1, 50, 10],
    'CT':  [22.046, 22.679851, 0.01, 1, 500],
    'KC':  [22.046, 17.009888, 0.05, 2.5, 375],
    'W':   [0.3674333, 136.07911, 0.25, 10, 50],
    'S':   [0.3674333, 136.07911, 0.25, 20, 50],
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


class Hedge:
    """Class that defines a hedge object. Hedge object has two main functionalities:
    1) inferring the hedging parameters from the portfolio
    2) applying the relevant hedges to the portfolio.

    Inference of hedging parameters is done via the _calibrate method, while
    applying hedges is done via the _apply method.
    """

    def __init__(self, portfolio, hedges, vdf=None, pdf=None,
                 buckets=None, brokerage=None, slippage=None,
                 book=False, settlements=None):
        """Constructor. Initializes a hedge object subject to the following parameters.

        Args:
            portfolio (object): the portfolio being hedged
            hedges (dictionary): Description
            vdf (dataframe): dataframe of volatilites
            pdf (dataframe): dataframe of prices
            buckets (None, optional): buckets to be used in hedging by exp
            brokerage (None, optional): brokergae value
            slippage (None, optional): slippage value 
            book (bool, optional): True if hedging is done basis book vols, false otherwise.
            settlements (dataframe, optional): dataframe of settlement vols. 
            valid inputs are 'exp' for greeks-by-expiry and 'uid' for greeks by underlying.

        """
        self.settlements = settlements
        self.book = book
        self.b = brokerage
        self.s = slippage
        self.mappings = {}
        self.greek_repr = {}
        self.vdf = vdf
        self.pdf = pdf
        self.pf = portfolio
        self.buckets = buckets if buckets is not None else [0, 30, 60, 90, 120]
        self.hedges = hedges
        self.desc, self.params = self.process_hedges()
        self.date = None
        self.breakeven = self.pf.breakeven().copy()

        # check if vdf/pdf have been populated. if not, update.
        if (self.vdf is not None and self.pdf is not None):
            vdf.value_date = pd.to_datetime(vdf.value_date)
            pdf.value_date = pd.to_datetime(pdf.value_date)
            self.date = pd.to_datetime(pdf.value_date.unique()[0])
            self.calibrate_all()
            assert len([self.date]) == 1

    def __str__(self):
        # custom string representation for this class.
        r_dict = {'desc': self.desc,
                  'params': self.params,
                  'mappings': self.mappings,
                  'date': self.date,
                  'hedges': self.hedges}

        return str(pprint.pformat(r_dict))

    def set_book(self, val):
        self.book = val

    def set_breakeven(self, dic):
        """Setter method that sets self.breakeven = dic

        Args:
            dic (TYPE): dictionary of breakevens, organized by product/month
        """
        self.breakeven = dic

    def update_dataframes(self, vdf, pdf, hedges=None, settles=None):
        """ Helper method that updates the dataframes and (potentially) the hedges
        used for calibration/hedging in this hedger object. 

        Args:
            vdf (TYPE): dataframe of volatilities
            pdf (TYPE): dataframe of prices 
            hedges (None, optional): dictionary of hedges. 
        """
        self.vdf = vdf.copy()
        self.pdf = pdf.copy()
        self.settlements = settles
        self.date = pd.to_datetime(pdf.value_date.min())
        # # overriding the date and recomputing ttm if book is true
        # if self.book:
        #     self.vdf.value_date = self.date
        #     self.vdf.expdate = pd.to_datetime(self.vdf.expdate)
        #     self.vdf.tau = (
        #         (self.vdf.expdate - self.vdf.value_date).dt.days)/365
        #     # self.pdf.value_date = self.date
        if hedges is not None:
            self.hedges = hedges
            self.desc, self.params = self.process_hedges()

        self.calibrate_all()

    def process_hedges(self):
        """Helper function that sorts through the mess that is the hedge 
        dictionary. Returns the following:
        1)  representation used to hedge gamma/theta/vega
        2) additional parameters used to hedge the same
            (such as the type of structure and the specifications of that structure.)
        Assigns these to self.desc and self.params respectively.
        """
        # print('processing hedges')
        desc, params = None, {}
        for flag in self.hedges:
            # sanity check.
            if flag not in params:
                params[flag] = {}
            # moving onto bound-based hedging.
            all_conds = self.hedges[flag]
            r_conds = [x for x in all_conds if x[0] == 'bound']
            s_conds = [x for x in all_conds if x[0] == 'static']
            it_conds = [x for x in all_conds if x[0] == 'intraday']
            if s_conds:
                s_conds = s_conds[0]
                val = 0 if s_conds[1] == 'zero' else int(s_conds[1])
                params[flag]['eod'] = {'target': val}

            if r_conds:
                r_conds = r_conds[0]
                # print('r_conds: ', r_conds)
                # begin the process of assigning.
                desc = r_conds[-1]
                desc == 'uid' if desc is None else desc
                if desc == 'exp':
                    if r_conds[-2] is not None:
                        self.buckets = list(r_conds[-2])
                    params[flag]['kind'] = r_conds[-5]
                    params[flag]['spectype'] = r_conds[-4]
                    params[flag]['spec'] = r_conds[-3]
                    if len(r_conds) == 10:
                        if r_conds[4] == 'days':
                            params[flag]['tau_val'] = r_conds[3] / 365
                            params[flag]['tau_desc'] = 'days'
                        elif r_conds[4] == 'years':
                            params[flag]['tau_val'] = r_conds[3]
                            params[flag]['tau_desc'] = 'years'
                        elif r_conds[4] == 'ratio':
                            params[flag]['tau_val'] = r_conds[3]
                            params[flag]['tau_desc'] = 'ratio'

                elif desc == 'uid':
                    params[flag]['kind'] = r_conds[-4]
                    params[flag]['spectype'] = r_conds[-3]
                    params[flag]['spec'] = r_conds[-2]
                    # case: ttm is specified for UID hedging.
                    if len(r_conds) == 9:
                        if r_conds[4] == 'days':
                            params[flag]['tau_val'] = r_conds[3] / 365
                            params[flag]['tau_desc'] = 'days'
                        elif r_conds[4] == 'years':
                            params[flag]['tau_val'] = r_conds[3]
                            params[flag]['tau_desc'] = 'years'
                        elif r_conds[4] == 'ratio':
                            params[flag]['tau_val'] = r_conds[3]
                            params[flag]['tau_desc'] = 'ratio'
            if it_conds:
                # currently defaults to 0.
                it_conds = it_conds[0]
                print('it_conds: ', it_conds)
                params[flag]['intraday'] = {'kind': it_conds[1],
                                            'modifier': it_conds[2],
                                            'target': 0}

            if desc is None:
                desc = 'uid'

        return desc, params

    def calibrate_all(self):
        """Calibrates hedge object to all non-delta hedges. calibrate does one of the following things:
        1) if hedge_engine is initialized with desc='exp', calibrate generates a
            dictionary mapping product and expiry to a vol_id, which will be used to
            hedge that pdt/exp combination
        2) desc = 'uid' -> _calibrate generates a dictionary mapping product/underlying month
            to a vol_id, dependent on any ttm multipliers passed in.
        """
        for flag in self.hedges:
            if flag != 'delta':
                self._calibrate(flag)

    def _calibrate(self, flag, selection_criteria='median', buckets=None):
        """Helper method that constructs the hedging parameters based on
        the greek representation fed into the hedge object.

        Example (1): flag == 'exp' indicates that hedging is being done basis
        greeks bucketed according to time to maturity.

        As such, the parameter dictionary generated is a dictionary mapping
        commodity and ttm to vol_id used to hedge that particular dictionary.

        Example (2): flag == 'uid' indicates that hedging is being done basis
        greeks clubbed according to underlying contract, e.g. the greeks for
        a W Q6.U6 and W U6.U6 will be added together and
        hedged


        Args:
            hedges (dictionary): dictionary of hedging logic passed into the simulation.
            flag (TYPE):


        Note: this does NOT check if greeks other than delta are being hedged,
        because delta by default is hedged at the EOD by commodity/month, not
        on the basis of expiries. I.e. you will never use a W Q6 underlying
        to hedge the deltas from a W Q6.U6 option.

        """
        # print('-+-+-+- calibrating ' + flag + ' -+-+-+-')
        net = {}
        ttm = None

        data = self.params[flag]

        # first case: greek by expiry.
        if self.desc == 'exp':
            # self.pf.update_sec_by_month(None, 'OTC', update=True)
            # self.pf.update_sec_by_month(None, 'hedge', update=True)
            self.greek_repr = self.pf.greeks_by_exp(self.buckets) \
                if buckets is None else self.pf.greeks_by_exp(buckets)

            calibration_dic = self.pf.greeks_by_exp(self.buckets) \
                if buckets is None else self.pf.greeks_by_exp(buckets)

            for product in calibration_dic:
                df = self.vdf[(self.vdf.pdt == product) &
                              (self.vdf.call_put_id == 'C')]
                for exp in calibration_dic[product]:
                    loc = (product, exp)
                    if df.empty:
                        if flag not in self.mappings:
                            self.mappings[flag] = {}
                        self.mappings[flag][loc] = False
                        continue

                    options = calibration_dic[product][exp][0]

                    # case: no options associated with this product/exp bucket
                    if not options:
                        continue

                    # case: ttm specification exists.
                    if 'tau_val' in data:
                        ttm_modifier = data['tau_val']
                        # print('ttm modifier: ', ttm_modifier)
                        # case: ttm modifier is a ratio.
                        if data['tau_desc'] == 'ratio':
                            if selection_criteria == 'median':
                                ttm = np.median([op.tau for op in options])
                            ttm = ttm * ttm_modifier
                        # ttm modifier is an actual value. s
                        else:
                            ttm = ttm_modifier
                    else:
                        ttm = max([op.tau for op in options])

                    # check available vol_ids and pick the closest one.

                    closest_tau_val = sorted(
                        df.tau, key=lambda x: abs(x - ttm))

                    # sanity check: ensuring that the option being selected to
                    # hedge has at least 4 days to maturity (i.e. to account
                    # for a weekend)

                    valid = [x for x in closest_tau_val if x > 4/365]
                    closest_tau_val = valid[0]
                    vol_ids = df[df.tau == closest_tau_val].vol_id.values

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
            self.greek_repr = self.pf.get_net_greeks()
            net = self.pf.get_net_greeks()
            # assert not self.vdf.empty
            for product in net:
                product = product.strip()
                df = self.vdf[self.vdf.pdt == product]
                # case: holiday for this product.
                if df.empty:
                    print('data does not exist for this date and product. ',
                          self.date, product)
                for month in net[product]:
                    month = month.strip()
                    uid = product + '  ' + month
                    df = df[df.underlying_id == uid]
                    data = net[product][month]
                    loc = (product, month)
                    if df.empty:
                        if flag not in self.mappings:
                            self.mappings[flag] = {}
                        self.mappings[flag][loc] = False
                        continue
                    if not data or (data == [0, 0, 0, 0]):
                        continue
                    # grab max possible ttm (i.e. ttm of the same month option)
                    try:
                        volid = product + '  ' + month + '.' + month
                        ttm = df[(df.vol_id == volid) &
                                 (df.call_put_id == 'C')].tau.values[0]

                    except IndexError:
                        print('hedge.uid_volid: cannot find max ttm')
                        print('debug 1: ', df[(df.vol_id == volid)])
                        print('debug 2: ', df[(df.call_put_id == 'C')])
                        print('debug 3: ', df[
                              (df.vol_id == volid) & (df.call_put_id == 'C')])

                    # case: ttm specification exists.
                    if 'tau_val' in data:
                        ttm_modifier = data['tau_val']
                        # case: ttm modifier is a ratio.
                        if data['tau_desc'] == 'ratio':
                            ttm = ttm * ttm_modifier
                        # ttm modifier is an actual value.
                        else:
                            ttm = ttm_modifier

                    closest_tau_val = min(df.tau, key=lambda x: abs(x - ttm))
                    vol_ids = df[(df.underlying_id == uid) &
                                 (df.tau == closest_tau_val)].vol_id.values

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

    def get_bucket(self, val, buckets=None):
        """Helper method that gets the bucket associated with a given value according to self.buckets.

        Args:
            val (TYPE): Description
            buckets (TYPE, optional): Description
        """
        buckets = self.buckets if buckets is None else buckets
        return min([x for x in buckets if x > val])

    def satisfied(self):
        """Helper method that delegates checks if the hedge conditions are satisfied

        Args:
            pf (object): The portfolio object being hedged
        """
        if self.desc == 'uid':
            return self.uid_hedges_satisfied()

        elif self.desc == 'exp':
            return self.exp_hedges_satisfied()

    def uid_hedges_satisfied(self):
        """Helper method that ascertains if all entries in net_greeks satisfy
        the conditions laid out in hedges.

        Args:
            pf (portfolio object): portfolio being hedged
            hedges (ordered dictionary): contains hedge information/specifications

        Returns:
            Boolean: indicating if the hedges are all satisfied or not.
        """
        # print('--- checking uid hedges satisfied ---')
        self.pf.update_sec_by_month(None, 'OTC', update=True)
        self.pf.update_sec_by_month(None, 'hedge', update=True)
        strs = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 3}
        tst = self.hedges.copy()

        net_greeks = self.pf.get_net_greeks()
        self.greek_repr = net_greeks
        # delta condition:
        conditions = []
        for greek in tst:
            conds = tst[greek]
            # print('conds: ', conds)
            for cond in conds:
                # static bound case
                if cond[0] == 'static':
                    val = self.params[greek]['eod']['target']
                    ltol, utol = (val - 1, val + 1)
                    conditions.append((strs[greek], (ltol, utol)))
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
                    ltol, utol = cond[1]
                    index = cond[0]
                    # sanity check: if greek is negative, flip the sign of the bounds
                    # in the case of gamma/theta/vega
                    if greeks[index] < 0 and index != 0:
                        tmp = ltol
                        ltol = -utol
                        utol = -tmp
                    # print('scripts.hedge.check_uid_hedges: inputs - ',
                    #       greeks[index], ltol, utol)
                    if (greeks[index] > utol) or \
                            (greeks[index] < ltol):
                        # print(str(cond) + ' failed')
                        print(greeks[index], ltol, utol)
                        # print('--- done checking uid hedges satisfied ---')
                        return False

        # rolls_satisfied = check_roll_hedges(pf, hedges)
        # print('--- done checking uid hedges satisfied ---')
        return True

    def exp_hedges_satisfied(self):
        """Helper method that checks if greeks according to expiry
             representation are adequately hedged.

        Args:
            pf (TYPE): Description
            hedges (TYPE): Description
        """
        strs = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 3}
        self.pf.update_sec_by_month(None, 'OTC', update=True)
        self.pf.update_sec_by_month(None, 'hedge', update=True)
        self.greek_repr = self.pf.greeks_by_exp(self.buckets)
        net_greeks = self.greek_repr
        tst = self.hedges.copy()
        conditions = []
        for greek in tst:
            conds = tst[greek]
            for cond in conds:
                # static bound case
                if cond[0] == 'static':
                    val = self.params[greek]['target']
                    ltol, utol = val - 1, val + 1
                    conditions.append((strs[greek], (ltol, utol)))
                elif cond[0] == 'bound':
                    c = cond[1]
                    tup = (strs[greek], c)
                    conditions.append(tup)

        for pdt in net_greeks:
            for exp in net_greeks[pdt]:
                ops = net_greeks[pdt][exp][0]
                greeks = net_greeks[pdt][exp][1:]
                if ops:
                    for cond in conditions:
                        ltol, utol = cond[1]
                        index = cond[0]
                        # sanity check: if greek is negative, flip the sign of the bounds
                        # in the case of gamma/theta/vega
                        if greeks[index] < 0 and index != 0:
                            tmp = ltol
                            ltol = -utol
                            utol = -tmp
                        if (greeks[index] > utol) or \
                                (greeks[index] < ltol):

                            return False
        # rolls_satisfied = check_roll_hedges(pf, hedges)
        return True

    def refresh(self):
        """Helper method that re-inializes all values. To be used when updating portfolio
        to ascertain hedges have been satisfied.
        """
        self.pf.refresh()
        # print('scripts.hedge - pre-refresh greek repr: ', self.greek_repr)
        for flag in self.hedges:
            if flag != 'delta':
                # print('refreshing ' + flag)
                self._calibrate(flag)

        # sanity check: case where delta is the only hedge, and greek repr
        # hence doesn't get updated (in fact gets wiped on refresh.)
        if self.greek_repr == {}:
            self.greek_repr = self.pf.get_net_greeks()

        # self.done = self.satisfied()

    def apply(self, flag, price_changes=None):
        """Main method that actually applies the hedging logic specified.
        The kind of structure used to hedge is specified by self.params['kind']

        Args:
            pf (TYPE): The portfolio being hedged
            flag (string): the greek being hedged
        """
        # base case: flag not in hedges
        # print('======= applying ' + flag + ' hedge =========')
        if flag not in self.hedges:
            raise ValueError(
                flag + ' hedge is not specified in hedging logic for family ' + self.pf.name)

        cost = 0
        indices = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 3}
        ind = indices[flag]

        conds = self.hedges[flag]
        # print('conds: ', conds)
        isbounds = [conds[i] for i in range(len(conds))
                    if conds[i][0] == 'bound']
        isstatic = [conds[i] for i in range(len(conds))
                    if conds[i][0] == 'static']

        if isstatic:
            relevant_conds = isstatic[0]
            hedge_type = 'static'

        elif isbounds:
            relevant_conds = isbounds[0]
            hedge_type = 'bound'

        if flag == 'delta' and hedge_type == 'static':
            fee = self.hedge_delta(price_changes=price_changes)
            return fee

        for product in self.greek_repr:
            for loc in self.greek_repr[product]:
                fulldata = self.greek_repr[product][loc]
                # case: options included in greek repr
                if len(fulldata) == 5:
                    ops = fulldata[0]
                    data = fulldata[1:]
                else:
                    data = fulldata
                greekval = data[ind]

                # sanity check in the case of nonzero lower bound and exp
                # hedging
                if len(fulldata) == 5 and len(ops) == 0:
                    # print('no options present for this exp; skipping.')
                    continue

                if hedge_type == 'bound':
                    bounds = relevant_conds[1]
                    target = (bounds[1] + bounds[0]) / 2
                    # case: bounds are exceeded.
                    if abs(greekval) < bounds[0] or abs(greekval) > bounds[1]:
                        cost += self.hedge(flag, product,
                                           loc, greekval, target)
                    else:
                        continue

                elif hedge_type == 'static':
                    val = self.params[flag]['target']
                    cost += self.hedge(flag, product, loc, greekval, val)

        return cost

    def hedge(self, flag, product, loc, greekval, target):
        """Helper method that neutralizes the greek specified by flag.

        Args:
            pf (portfolio object): the portfolio being hedged
            flag (str): the greek being neutralized
            product (string): the product in question
            loc (TYPE): additional locating parameters

        """
        # sanity check: since greek signs will never arbitrarily flip, wanna make sure
        # they stay the same sign.
        if greekval < 0:
            target = -target

        cost = 0
        reqd_val = target - greekval
        # all hedging info is contained in self.params[flag]
        data = self.params[flag]

        # identifying what position in the specified hedge structure is
        # required to be taken
        shorted = self.pos_type(data['kind'], reqd_val, flag)

        reqd_val = abs(reqd_val)

        # grab the vol_id associated with this greek/product/localizer.
        hedge_id = self.mappings[flag][(product, loc)]

        # sanity check: if no vol_id, not due for hedging --> return .
        if not hedge_id:
            return 0

        # adding the hedge structures.
        ops = self.add_hedges(data, shorted, hedge_id, flag, reqd_val, loc)

        self.refresh()

        # computing slippage/brokerage if required.
        if ops:
            if self.s:
                pass
            if self.b:
                cost += self.b * sum([op.lots for op in ops])

            # case: hedging is being done basis book vols. In this case,
            # the difference in premium paid must be computed basis settlement
            # vols, and the added to the cost of transaction.
            if self.book:
                for op in ops:
                    try:
                        cpi = 'C' if op.char == 'call' else 'P'
                        df = self.settlements
                        settle_vol = df[(df.vol_id == op.get_vol_id()) &
                                        (df.call_put_id == cpi) &
                                        (df.strike == op.K)].vol.values[0]
                    except IndexError as e:
                        print('scripts.hedge - book vol case: cannot find vol: ',
                              op.get_vol_id(), cpi, op.K)
                        settle_vol = op.vol
                    print(op.get_vol_id() + ' settle_vol: ', settle_vol)
                    print('op.book vol: ', op.vol)
                    true_value = _compute_value(op.char, op.tau, settle_vol, op.K,
                                                op.underlying.get_price(), 0, 'amer', ki=op.ki,
                                                ko=op.ko, barrier=op.barrier, d=op.direc,
                                                product=op.get_product(), bvol=op.bvol)
                    print('op value basis settlements: ', true_value)
                    pnl_mult = multipliers[op.get_product()][-1]
                    diff = (true_value - op.get_price()) * op.lots * pnl_mult
                    print('diff: ', diff)
                    cost += diff

        return cost

    def hedge_delta(self, price_changes=None):
        """Helper method that hedges delta basis net greeks, irrespective of the
        greek representation the object was initialized with.

        Args:
            pf (TYPE): Description
        """
        cost = 0
        ft = None
        net_greeks = self.pf.get_net_greeks()
        tobehedged = {}
        # be_dict = None

        print('price_changes: ', price_changes)

        # case: intraday data
        if price_changes is not None:
            print('intraday hedging case')
            # static value.
            if self.params['delta']['intraday']['kind'] == 'static':
                comp_val = self.params['delta']['intraday']['modifier']
                for uid in price_changes:
                    pdt, mth = uid.split()
                    if abs(price_changes[uid]) > comp_val:
                        if pdt not in tobehedged:
                            tobehedged[pdt] = set()
                        tobehedged[pdt].add(mth)

            # breakeven-based hedging.
            else:
                be_dic = self.breakeven
                for pdt in be_dic:
                    for mth in be_dic[pdt]:
                        mults = self.params['delta']['intraday']['modifier']
                        uid = pdt + '  ' + mth
                        print('mults: ', mults)
                        if uid in mults:
                            be_mult = mults[uid]
                            print('pdt, mth, mult: ', pdt, mth, be_mult)
                        else:
                            be_mult = 1
                        be = be_dic[pdt][mth] * be_mult
                        print('be: ', be_dic[pdt][mth])
                        print('mult: ', be_mult)
                        print('target_breakeven: ', be)
                        print('actual move: ', price_changes[uid])
                        if price_changes[uid] >= be:
                            if pdt not in tobehedged:
                                tobehedged[pdt] = set()
                            tobehedged[pdt].add(mth)

        # case: settlement-to-settlement.
        else:
            tobehedged = net_greeks

        print('tobehedged: ', tobehedged)

        target_flag = 'eod' if price_changes is None else 'intraday'
        for product in tobehedged:
            for month in tobehedged[product]:
                # uid = product + '  ' + month
                print('delta hedging ' + product + '  ' + month)
                target = self.params['delta'][target_flag]['target']
                delta = net_greeks[product][month][0]
                delta_diff = delta - target
                shorted = True if delta_diff > 0 else False
                num_lots_needed = abs(round(delta_diff))
                if num_lots_needed == 0:
                    continue
                else:
                    try:
                        ft, _ = create_underlying(product, month, self.pdf,
                                                  self.date, shorted=shorted,
                                                  lots=num_lots_needed, flag=target_flag)
                        if ft is not None:
                            self.pf.add_security([ft], 'hedge')
                            print('adding ' + str(ft))
                    except IndexError:
                        print('price data for this day ' + '-- ' + self.date.strftime(
                            '%Y-%m-%d') + ' --' + ' does not exist. skipping...')
        if self.s:
            pass
        if self.b:
            cost += self.b * ft.lots

        return cost

    # TODO: shorted check for other structures, implement as and when necessary
    def pos_type(self, desc, val, flag):
        """Helper method that checks what exactly is required given the required value
        of the greek, the greek itself, and the structure being used to hedge it.

        Args:
            desc (TYPE): Description
            val (TYPE): Description
            flag (TYPE): Description
        """
        shorted = None
        if desc in ('straddle', 'strangle', 'call', 'put'):
            if flag in ('gamma', 'vega'):
                shorted = True if val < 0 else False
            elif flag == 'theta':
                shorted = True if val > 0 else False

        return shorted

    def add_hedges(self, data, shorted, hedge_id, flag, greekval, loc):
        """Helper method that checks the type of hedge structure specified, and creates/adds
        the requisite amount.

        Args:
            data (TYPE): Description
            shorted (TYPE): Description
            hedge_id (TYPE): Description
            flag (TYPE): Description
            greekval (TYPE): Description

        """
        ops = None
        try:
            # print(data['kind'], data['spectype'], data['spec'])
            if data['kind'] == 'straddle':
                if data['spectype'] == 'strike':
                    strike = data['spec']
                ops = create_straddle(hedge_id, self.vdf, self.pdf, self.date,
                                      shorted, strike=strike, greek=flag, greekval=greekval)
                gv = greekval if not shorted else -greekval
                # print('added straddle with ' + str(gv) + ' ' + str(flag))

            elif data['kind'] == 'strangle':
                strike1, strike2, delta1, delta2, c_delta = None, None, None, None, None

                if data['spectype'] == 'strike':
                    strike1, strike2 = data['spec']
                elif data['spectype'] == 'delta':
                    if isinstance(data['spec'], (float, int)):
                        c_delta = data['spec']
                    else:
                        delta1, delta2 = data['spec']

                if c_delta is not None:
                    delta1, delta2 = c_delta, c_delta

                ops = create_strangle(hedge_id, self.vdf, self.pdf, self.date,
                                      shorted, chars=['call', 'put'],
                                      strike=[strike1, strike2],
                                      delta=[delta1, delta2], greek=flag, greekval=greekval)
                # print('added strangle: ' + str([str(op) for op in ops]))

            elif data['kind'] == 'call':
                dval, strike = None, None
                if data['spectype'] == 'delta':
                    dval = data['spec']

                if data['spectype'] == 'strike':
                    strike = data['spec']
                print('inputs: ', hedge_id, shorted,
                      dval, strike, flag, greekval)
                op = create_vanilla_option(self.vdf, self.pdf, hedge_id, 'call',
                                           shorted, delta=dval, strike=strike,
                                           greek=flag, greekval=greekval)
                ops = [op]
                gv = greekval if not shorted else -greekval
                # print('added call with ' + str(gv) + ' ' + str(flag))

            elif data['kind'] == 'put':
                dval, strike = None, None
                if data['spectype'] == 'delta':
                    dval = data['spec']

                if data['spectype'] == 'strike':
                    strike = data['spectype']

                op = create_vanilla_option(self.vdf, self.pdf, hedge_id, 'put',
                                           shorted, delta=dval, strike=strike,
                                           greek=flag, greekval=greekval)
                ops = [op]
                gv = greekval if not shorted else -greekval
                # print('added call with ' + str(gv) + ' ' + str(flag))

            # sanity check
            for op in ops:
                if op.lots < 1:
                    # print('lots req < 1; ' + flag +
                    #       ' is within bounds. skipping hedging.')
                    return []

            self.pf.add_security(list(ops), 'hedge')

        except IndexError:
            product = hedge_id.split()[0]
            print(product + ' ' + str(loc) +
                  ' price and/or vol data does not exist. skipping...')

        return ops
