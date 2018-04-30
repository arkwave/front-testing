# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-07-20 18:26:26
# @Last Modified by:   arkwave
# @Last Modified time: 2018-04-05 12:50:20

import pandas as pd
import pprint
import numpy as np
from .util import create_straddle, create_underlying, create_strangle, create_vanilla_option
from .calc import _compute_value
from .hedge_mods import TrailingStop, HedgeParser

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
        self.last_hedgepoints = None
        self.update_hedgepoints()
        self.entry_levels = self.pf.uid_price_dict().copy()
        self.hedges = hedges
        self.intraday_conds = None
        self.desc, self.params = self.process_hedges()
        self.hedgeparser = None
        if ('delta' in self.params) and \
           ('intraday' in self.params['delta']):
            self.hedgeparser = HedgeParser(self, self.params['delta']['intraday'],
                                           self.intraday_conds, self.pf)
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
                  'date': self.date}

        return str(pprint.pformat(r_dict))

    def get_hedgeparser(self):
        return self.hedgeparser

    def get_intraday_conds(self):
        return self.intraday_conds

    def get_hedgepoints(self):
        return self.last_hedgepoints

    def set_book(self, val):
        self.book = val

    def get_breakeven(self):
        return self.breakeven

    def get_hedge_ratios(self, flag):
        """Helper method that queries the HedgeParser object for the run_ratio dictionary, and returns 
        1- each entry. 
        """
        if self.hedgeparser is None:
            return {uid: 1 for uid in self.pf.get_unique_uids()}

        dic = self.hedgeparser.parse_hedges(flag)
        dic = {uid: 1 - dic[uid] for uid in dic}

        return dic

    def set_breakeven(self, dic):
        """Setter method that sets self.breakeven = dic

        Args:
            dic (TYPE): dictionary of breakevens, organized by product/month
        """
        self.breakeven = dic
        if self.intraday_conds is not None:
            self.intraday_conds.update_thresholds(self.breakeven)

    def update_hedgepoints(self):
        """Helper method that constructs the self.hedge_points dictionary, which maps 
        vol_ids to the price point the last hedged was placed at. Used for intraday hedging
        to keep track of where on the breakeven/static value we are at. 
        """
        self.last_hedgepoints = self.pf.uid_price_dict()

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
            # static EOD delta hedging conditions.
            if s_conds:
                s_conds = s_conds[0]
                val = 0 if s_conds[1] == 'zero' else int(s_conds[1])
                params[flag]['eod'] = {'target': val}

            # bound conditions.
            if r_conds:
                r_conds = r_conds[0]
                # print('r_conds: ', r_conds)
                # begin the process of assigning.
                desc = r_conds[-1]
                desc == 'uid' if desc is None else desc

                # FIXME: replace this with the appropriate aggregation conditions
                if desc == 'agg':
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

            # intraday hedging conditions
            if it_conds:
                # currently defaults to 0.
                it_conds = it_conds[0]
                # case: no additional parameters are passed in.
                if len(it_conds) == 3:
                    ratio = 1

                # case: additional intraday parameters were passed in.
                else:
                    # delta running specifications passed in.
                    if isinstance(it_conds[-1], dict):
                        self.process_intraday_conds(it_conds[-1])

                    # ratio passed in.
                    ratio = it_conds[3] if isinstance(
                        it_conds[3], (int, float)) else 1

                params[flag]['intraday'] = {'kind': it_conds[1],
                                            'modifier': it_conds[2],
                                            'target': 0,
                                            'ratio': ratio}
            if desc is None:
                desc = 'uid'

        return desc, params

    # TODO: update this if more intraday conditions come into play later. for now, this just
    # handles trailing stops.
    def process_intraday_conds(self, dic):
        """Helper function that processes intraday hedging parameters. 

        Args:
            dic (TYPE): Dictionary containing additional constraints that intraday hedging is subject to. 
            Examples would include conditions for letting delta run (i.e. trailing stop losses, 
            returning to a fixed value, etc.)

        Returns:
            object: TrailingStop instance 
        """
        # print('intraday conds: dic - ', dic)
        if 'tstop' in dic:
            # print('-------- Creating TrailingStop Object ---------')
            tstop_obj = TrailingStop(self.entry_levels, dic[
                                     'tstop'], self.pf, self.last_hedgepoints, self)
            self.intraday_conds = tstop_obj
            assert self.intraday_conds is not None
            # print('------------ TrailingStop Created -------------')

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

        # first case: greeks by underlying (regular thing we're used to)
        if self.desc == 'uid':
            self.greek_repr = self.pf.get_net_greeks()
            net = self.pf.get_net_greeks()
            # assert not self.vdf.empty

        # second case: aggregated greeks. 
        # needs to do the following:
        # 1. find out the largest contributing gamma. 
        # 2. use the right TTM option to hedge - current method of specifying a ttm will work.
        elif self.desc == 'agg': 
            self.greek_repr = self.pf.get_aggregated_greeks() 
            net = self.greek_repr.copy() 

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

    def apply(self, flag, intraday=False, ohlc=False):
        """Main method that actually applies the hedging logic specified.
        The kind of structure used to hedge is specified by self.params['kind']

        Args:
            pf (TYPE): The portfolio being hedged
            flag (string): the greek being hedged
        """
        # base case: flag not in hedges
        # print('======= applying ' + flag + ' hedge =========')
        if flag not in self.hedges:
            print(
                flag + ' hedge is not specified in hedging logic for family ' + self.pf.name)
            return 0

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
            fee = self.hedge_delta(intraday=intraday)
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
        ops = self.add_hedges(data, shorted, hedge_id, flag,
                              reqd_val, loc, settlements=self.settlements)

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

    # TODO: need to handle trailing stops if they exist.
    def is_relevant_price_move(self, uid, val, comparison=None):
        """Helper method that checks to see if a price-uid combo fed in is a valid price move. 
        Three cases are handled: 
        1) price move > breakeven * mult
        2) price move > flat value
        3) price type is settlement. 
        """

        if uid in self.last_hedgepoints:
            last_val = self.last_hedgepoints[uid]

        # rawval overrwrites the last hedge point if passed in.
        if comparison is not None:
            last_val = comparison

        # case: intraday hedges not specified -> data is settlement.
        if not self.params or 'intraday' not in self.params['delta']:
            return True, 0
        else:
            delta_params = self.params['delta']['intraday']
            # case 1: trailing stop hit.
            if self.intraday_conds is not None:
                if self.intraday_conds.trailing_stop_hit(uid, val=val):
                    return True, 0

            # get price move.
            actual_value = abs(last_val - val)

            # case 2: flat value-based hedging.
            if delta_params['kind'] == 'static':
                comp_val = delta_params['modifier']
                if actual_value >= comp_val[uid] or np.isclose(actual_value, comp_val[uid]):
                    return True, np.floor(actual_value/comp_val[uid])

            # case 3: breakeven-based hedging.
            elif delta_params['kind'] == 'breakeven':
                mults = delta_params['modifier']
                pdt, mth = uid.split()
                if pdt in mults and mth in mults[pdt]:
                    be_mult = mults[pdt][mth]
                else:
                    be_mult = 1
                be = self.breakeven[pdt][mth] * be_mult
                # print('be: ', be)
                if actual_value >= be or np.isclose(actual_value, be):
                    return True, np.floor(actual_value/be)

            return False, 0

    def get_hedge_interval(self, uid=None):
        """Helper function that gets the hedging interval.

        Args:
            uid (str): the underlying id in question. 

        Returns:
            TYPE: the move level at which this uid should be hedged. 
        """
        if uid is not None:
            if self.params['delta']['intraday']['kind'] == 'static':
                # print('static value intraday case')
                comp_val = self.params['delta']['intraday']['modifier'][uid]

            elif self.params['delta']['intraday']['kind'] == 'breakeven':
                # print('breakeven intraday case')
                mults = self.params['delta']['intraday']['modifier']
                # print('mults: ', mults)
                pdt, mth = uid.split()
                if pdt in mults and mth in mults[pdt]:
                    be_mult = mults[pdt][mth]
                    print('pdt, mth, mult: ', pdt, mth, be_mult)
                else:
                    be_mult = 1
                comp_val = self.breakeven[pdt][mth] * be_mult

            return comp_val
        else:

            if self.params['delta']['intraday']['kind'] == 'static':
                return self.params['delta']['intraday']['modifier']
            elif self.params['delta']['intraday']['kind'] == 'breakeven':
                ret = {}
                mults = self.params['delta']['intraday']['modifier']
                for pdt in self.breakeven:
                    for mth in self.breakeven[pdt]:
                        uid = pdt + '  ' + mth
                        # get the mult.
                        mult = mults[pdt][mth]
                        fin = mult * self.breakeven[pdt][mth]
                        ret[uid] = fin
                return ret

    def hedge_delta(self, intraday=False, ohlc=False):
        """Helper method that hedges delta basis net greeks, irrespective of the
        greek representation the object was initialized with.

        Args:
            intraday (bool, optional): True if intraday hedge, False otherwise.
            ohlc (bool, optional): True if simulation uses OHLC data, False otherwise. 

        Returns:
            Portfolio: delta-hedged portfolio, based on inputs passed in. 
        """
        pnl = 0
        ft = None
        net_greeks = self.pf.get_net_greeks()
        tobehedged = {}
        print('last hedgepoints: ', self.last_hedgepoints)
        # case: intraday data
        if intraday:
            print('intraday hedging case')
            curr_prices = self.pf.uid_price_dict()
            for uid in self.pf.get_unique_uids():
                pdt, mth = uid.split()
                relevant_move, move_mult = self.is_relevant_price_move(
                    uid, curr_prices[uid])
                if relevant_move:
                    if pdt not in tobehedged:
                        tobehedged[pdt] = set()
                    tobehedged[pdt].add(mth)

        # case: settlement-to-settlement.
        else:
            tobehedged = net_greeks

        print('tobehedged: ', tobehedged)
        print('-------- Entering HedgeParser Logic ---------')
        target_flag = 'intraday' if intraday else 'eod'

        hedge_ratios = self.get_hedge_ratios(target_flag)

        print('hedge_ratios: ', hedge_ratios)
        print('--------- End HedgeParser Logic -------------')
        for product in tobehedged:
            for month in tobehedged[product]:
                uid = product + '  ' + month
                print('delta hedging ' + uid)
                target = self.params['delta'][target_flag]['target']
                delta = net_greeks[product][month][0] * hedge_ratios[uid]
                delta_diff = delta - target
                shorted = True if delta_diff > 0 else False
                num_lots_needed = abs(round(delta_diff))
                print('num_lots_needed: ', num_lots_needed)
                ft = None
                if num_lots_needed == 0:
                    print('no hedging required for ' + product +
                          '  ' + month + '; hedge point not updated.')
                    continue
                else:
                    try:
                        ft, _ = create_underlying(product, month, self.pdf,
                                                  self.date, shorted=shorted,
                                                  lots=num_lots_needed, flag=target_flag)
                    except IndexError:
                        print('price data for this day ' + '-- ' + self.date.strftime(
                            '%Y-%m-%d') + ' --' + ' does not exist. skipping...')

                    if ft is not None:
                        # update the last hedgepoint dictionary
                        self.last_hedgepoints[
                            ft.get_uid()] = ft.get_price()
                        self.pf.add_security([ft], 'hedge')

        if self.s:
            pass
        if self.b:
            pnl -= self.b * ft.lots

        return pnl

    def handle_ohlc_pnl(ft, diff):
        """Helper function that handles the PnL from futures when running the simulation
        on OHLC data. This function computes the PnL the future would have produced had it been
        added with price at the breakeven/static value, and ridden up/down to the current level. 

        Args:
            ft (TYPE): The future that was created as a result of an intraday hedge
            diff (TYPE): the difference between the actual magnitude of the move and the 
                         static value/breakeven the hedge was put in at. as such, the magnitude of the price
                         move we care about is precisely diff. 
        """
        print('handling OHLC PnL for ' + str(ft) + ' with diff of ' + str(diff))
        pnl_mult = multipliers[ft.get_product()][-1]
        shorted = -1 if ft.shorted else 1

        return diff * ft.lots * pnl_mult * shorted

    # TODO: shorted check for other structures, implement as and when necessary
    def pos_type(self, desc, val, flag):
        """Helper method that checks what exactly is required given the required value
        of the greek, the greek itself, and the structure being used to hedge it.

        Args:
            desc (TYPE): the kind of position being used to hedge
            val (TYPE): the required valued 
            flag (TYPE): the greek being hedged
        """
        shorted = None
        if desc in ('straddle', 'strangle', 'call', 'put'):
            if flag in ('gamma', 'vega'):
                shorted = True if val < 0 else False
            elif flag == 'theta':
                shorted = True if val > 0 else False

        return shorted

    def add_hedges(self, data, shorted, hedge_id, flag, greekval, loc, settlements=None):
        """Helper method that checks the type of hedge structure specified, and creates/adds
        the requisite amount.

        Args:
            data (dict): greek-specific dictionary containing hedging specifications. subset of self.params.
            shorted (bool): True if hedges are to be shorted, False otherwise.
            hedge_id (str): vol_id to be used in creating the hedge.
            flag (str): the greek being hedged
            greekval (float): the sizing of the hedge required. 
            loc (TYPE): loc = future month if uid hedging, else the ttm. 
            settlements (dataframe, optional): dataframe of settlement vols. used if self.book = True, i.e if hedging is
            being done basis book vols. 

        Returns:
            Option objects: the option objects used to hedge the greek. 

        """
        # designate the dataframes to be used for hedging.
        hedge_vols = settlements if self.book else self.vdf
        ops = None
        try:

            # print(data['kind'], data['spectype'], data['spec'])
            if data['kind'] == 'straddle':
                if data['spectype'] == 'strike':
                    strike = data['spec']
                ops = create_straddle(hedge_id, hedge_vols, self.pdf, self.date,
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

                ops = create_strangle(hedge_id, hedge_vols, self.pdf, self.date,
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
                op = create_vanilla_option(hedge_vols, self.pdf, hedge_id, 'call',
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

                op = create_vanilla_option(hedge_vols, self.pdf, hedge_id, 'put',
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
