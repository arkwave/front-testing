# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-07-20 18:26:26
# @Last Modified by:   Ananth
# @Last Modified time: 2017-08-21 15:03:03
import pandas as pd
from timeit import default_timer as timer
import numpy as np
from .util import create_straddle, create_underlying, create_strangle, create_vanilla_option


# TODO: accept specifications on hedge option construction from hedge
# dictionary.
class Hedge:
    """Class that defines a hedge object. Hedge object has two main functionalities:
    1) inferring the hedging parameters from the portfolio
    2) applying the relevant hedges to the portfolio.

    Inference of hedging parameters is done via the _calibrate method, while
    applying hedges is done via the _apply method.
    """

    def __init__(self, portfolio, hedges, vdf, pdf,
                 buckets=None, brokerage=None, slippage=None):
        """Constructor. Initializes a hedge object subject to the following parameters.

        Args:
            portfolio (TYPE): the portfolio being hedged
            vdf (TYPE): dataframe of volatilites
            pdf (TYPE): Description
            view (TYPE): a string description of the greek representation.
            valid inputs are 'exp' for greeks-by-expiry and 'uid' for greeks by underlying.
            buckets (None, optional): Description
        """
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
        # self.done = self.satisfied()
        self.date = pd.to_datetime(pdf.value_date.unique()[0])
        assert len([self.date]) == 1
        self.calibrate_all()

    def process_hedges(self):
        """Helper function that sorts through the mess that is the hedge dictionary. Returns the following:
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
            if s_conds:
                s_conds = s_conds[0]
                val = 0 if s_conds[1] == 'zero' else int(s_conds[1])
                params[flag]['target'] = val

            if r_conds:
                r_conds = r_conds[0]
                # print('r_conds: ', r_conds)
                # begin the process of assigning.
                desc = r_conds[-1]
                params[flag]['kind'] = r_conds[-4]
                params[flag]['spectype'] = r_conds[-3]
                params[flag]['spec'] = r_conds[-2]
                # case: r_conds contains ttm values to use for hedging.
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

                if desc == 'exp':
                    if r_conds[-2] is not None:
                        self.buckets = list(r_conds[-2])

        # print('processing hedges completed')

        # one last sanity check
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
        print('-+-+-+- calibrating ' + flag + ' -+-+-+-')
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
                df = self.pdf[self.pdf.pdt == product]
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
                        print('ttm modifier: ', ttm_modifier)
                        # case: ttm modifier is a ratio.
                        if data['tau_desc'] == 'ratio':
                            if selection_criteria == 'median':
                                ttm = np.median([op.tau for op in options])
                            ttm = ttm * ttm_modifier
                        # ttm modifier is an actual value. s
                        else:
                            ttm = ttm_modifier

                    # check available vol_ids and pick the closest one.

                    closest_tau_val = sorted(
                        df.tau, key=lambda x: abs(x - ttm))

                    # sanity check: ensuring that the option being selected to
                    # hedge has at least 4 days to maturity (i.e. to account
                    # for a weekend)
                    closest_tau_val = min(
                        [x for x in closest_tau_val if x > 4 / 365])

                    vol_ids = df[df.tau == closest_tau_val].vol_id.values

                    # print('vol_ids: ', vol_ids)
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
            # self.pf.update_sec_by_month(None, 'OTC', update=True)
            # self.pf.update_sec_by_month(None, 'hedge', update=True)
            self.greek_repr = self.pf.get_net_greeks()
            print('scripts.hedges.calibrate - greek repr ', self.greek_repr)
            net = self.pf.get_net_greeks()
            for product in net:
                df = self.pdf[self.pdf.pdt == product]
                for month in net[product]:
                    uid = product + '  ' + month
                    df = df[(df.underlying_id == uid)]
                    data = net[product][month]
                    loc = (product, month)
                    if df.empty:
                        print('df is empty, data missing. skipping..')
                        if flag not in self.mappings:
                            self.mappings[flag] = {}
                        self.mappings[flag][loc] = False
                        continue
                    if not data or (data == [0, 0, 0, 0]):
                        continue

                    # grab max possible ttm (i.e. ttm of the same month option)
                    try:
                        volid = product + '  ' + month + '.' + month
                        # print('max_ttm volid: ', volid)
                        ttm = df[(df.vol_id == volid) & (
                            df.call_put_id == 'C')].tau.values[0]
                    except IndexError:
                        print('hedge.uid_volid: cannot find max ttm')
                        print('debug 1: ', df[(df.vol_id == volid)])
                        print('debug 2: ', df[(df.call_put_id == 'C')])
                        print('debug 3: ', df[
                              (df.vol_id == volid) & (df.call_put_id == 'C')])

                    # case: ttm specification exists.
                    if 'tau_val' in data:
                        # print('ttm specification exists for hedging ' + flag)
                        ttm_modifier = data['tau_val']
                        # case: ttm modifier is a ratio.
                        if data['tau_desc'] == 'ratio':
                            ttm = ttm * ttm_modifier
                        # ttm modifier is an actual value. s
                        else:
                            ttm = ttm_modifier

                    closest_tau_val = min(df.tau, key=lambda x: abs(x - ttm))

                    # print('closest_tau_val: ', closest_tau_val)

                    vol_ids = df[(df.underlying_id == uid) &
                                 (df.tau == closest_tau_val)].vol_id.values

                    # print('product, month: ', product, month)
                    # print('vol_ids: ', vol_ids)

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

        print('-+-+-+- done calibrating ' + flag + ' -+-+-+-')

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
        print('--- checking uid hedges satisfied ---')
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
            print('conds: ', conds)
            for cond in conds:
                # static bound case
                if cond[0] == 'static':
                    val = self.params[greek]['target']
                    ltol, utol = val - 1, val + 1
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
                    bound = cond[1]
                    print('scripts.hedge.check_uid_hedges: inputs - ',
                          greeks[cond[0]], bound[0], bound[1])
                    if (greeks[cond[0]] > bound[1]) or \
                            (greeks[cond[0]] < bound[0]):
                        print(str(cond) + ' failed')
                        print(greeks[cond[0]], bound[0], bound[1])
                        print('--- done checking uid hedges satisfied ---')
                        return False

        # rolls_satisfied = check_roll_hedges(pf, hedges)
        print('--- done checking uid hedges satisfied ---')
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
        # if 'delta' in tst:
        #     tst.pop('delta')

        # delta condition:
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
                    # print('to be literal eval-ed: ', hedges[greek][1])
                    c = cond[1]
                    tup = (strs[greek], c)
                    conditions.append(tup)
        # print('conditions: ', conditions)

        for pdt in net_greeks:
            for exp in net_greeks[pdt]:
                ops = net_greeks[pdt][exp][0]
                greeks = net_greeks[pdt][exp][1:]
                # print('ops: ', ops)
                if ops:
                    print('ops is true for exp = ' + str(exp))
                    for cond in conditions:
                        bound = cond[1]
                        if (greeks[cond[0]] > bound[1]) or (greeks[cond[0]] < bound[0]):
                            print(str(cond) + ' failed')
                            return False
        # rolls_satisfied = check_roll_hedges(pf, hedges)
        return True

    def refresh(self):
        """Helper method that re-inializes all values. To be used when updating portfolio
        to ascertain hedges have been satisfied.
        """
        # print('scripts.hedge - pre-refresh greek repr: ', self.greek_repr)
        for flag in self.hedges:
            if flag != 'delta':
                print('refreshing ' + flag)
                self._calibrate(flag)

        # sanity check: case where delta is the only hedge, and greek repr
        # hence doesn't get updated (in fact gets wiped on refresh.)
        if self.greek_repr == {}:
            self.greek_repr = self.pf.get_net_greeks()

        # self.done = self.satisfied()

    def apply(self, flag):
        """Main method that actually applies the hedging logic specified.
        The kind of structure used to hedge is specified by self.params['kind']

        Args:
            pf (TYPE): The portfolio being hedged
            flag (string): the greek being hedged
        """
        # base case: flag not in hedges
        print('======= applying ' + flag + ' hedge =========')
        if flag not in self.hedges:
            raise ValueError(flag + ' hedge is not specified in hedging.csv')

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

        if flag == 'delta':
            fee = self.hedge_delta()
            # cost += self.hedge_delta(pf)
            print('======= done ' + flag + ' hedge =========')
            return fee

        # print('flag: ', flag)
        # print('hedge_type: ', hedge_type)
        print('greek repr: ', self.greek_repr)
        for product in self.greek_repr:
            for loc in self.greek_repr[product]:
                print('>> hedging ' + str(product) + ' ' + str(loc) + ' <<')
                fulldata = self.greek_repr[product][loc]
                # case: options included in greek repr
                if len(fulldata) == 5:
                    ops = fulldata[0]
                    data = fulldata[1:]
                else:
                    data = fulldata
                print('data: ', data)
                greekval = data[ind]

                # sanity check in the case of nonzero lower bound and exp
                # hedging
                if len(fulldata) == 5 and len(ops) == 0:
                    print('no options present for this exp; skipping.')
                    continue

                if hedge_type == 'bound':
                    bounds = relevant_conds[1]
                    target = (bounds[1] + bounds[0]) / 2
                    # case: bounds are exceeded.
                    if abs(greekval) < bounds[0] or abs(greekval) > bounds[1]:
                        cost += self.hedge(flag, product,
                                           loc, greekval, target)
                    else:
                        print(product + ' ' + str(loc) +
                              ' ' + flag + ' within bounds. skipping...')

                elif hedge_type == 'static':
                    val = self.params[flag]['target']
                    cost += self.hedge(flag, product, loc, greekval, val)
        print('======= done ' + flag + ' hedge =========')
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
        # if greekval < 0:
        #     target = -target

        cost = 0
        print('target ' + flag + ': ', target)
        print('current ' + flag + ': ', greekval)

        reqd_val = target - greekval

        print('required ' + flag + ': ', reqd_val)
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

        # computing slippage/brokerage if required.
        if ops:
            if self.s:
                pass
            if self.b:
                cost += self.b * sum([op.lots for op in ops])

        return cost

    def hedge_delta(self):
        """Helper method that hedges delta basis net greeks, irrespective of the
        greek representation the object was initialized with.

        Args:
            pf (TYPE): Description
        """
        # print('hedging delta')
        cost = 0
        ft = None
        net_greeks = self.pf.get_net_greeks()
        for product in net_greeks:
            for month in net_greeks[product]:
                # uid = product + '  ' + month
                target = self.params['delta']['target']
                delta = net_greeks[product][month][0]
                delta_diff = delta - target
                shorted = True if delta_diff > 0 else False
                num_lots_needed = abs(round(delta_diff))
                if num_lots_needed == 0:
                    print(product + ' ' + month +
                          ' delta is close enough to target. skipping hedging.')
                else:
                    try:
                        ft, _ = create_underlying(product, month, self.pdf,
                                                  self.date, shorted=shorted, lots=num_lots_needed)
                        if ft is not None:
                            self.pf.add_security([ft], 'hedge')
                            print('adding ' + str(ft))
                    except IndexError:
                        print('price data for this day does not exist. skipping...')
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
        if desc == 'straddle':
            if flag in ('gamma', 'vega'):
                shorted = True if val < 0 else False
            elif flag == 'theta':
                shorted = True if val > 0 else False

        elif desc in ('call', 'put'):
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
                print('added straddle with ' + str(gv) + ' ' + str(flag))

            elif data['kind'] == 'strangle':
                strike1, strike2, delta1, delta2 = None, None, None, None

                if data['spectype'] == 'strike':
                    strike1, strike2 = data['spec']
                elif data['spectype'] == 'delta':
                    delta1, delta2 = data['spec']

                ops = create_strangle(hedge_id, self.vdf, self.pdf, self.date,
                                      shorted, chars=['call', 'put'],
                                      strike=[strike1, strike2],
                                      delta=[delta1, delta2], greek=flag, greekval=greekval)
                print('added strangle: ' + str([str(op) for op in ops]))

            elif data['kind'] == 'call':
                dval, strike = None, None
                if data['spectype'] == 'delta':
                    dval = data['spec']

                if data['spectype'] == 'strike':
                    strike = data['spectype']

                op = create_vanilla_option(self.vdf, self.pdf, hedge_id, 'call',
                                           shorted, delta=dval, strike=strike,
                                           greek=flag, greekval=greekval)
                ops = [op]
                gv = greekval if not shorted else -greekval
                print('added call with ' + str(gv) + ' ' + str(flag))

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
                print('added call with ' + str(gv) + ' ' + str(flag))

            # sanity check
            for op in ops:
                if op.lots < 1:
                    print('lots req < 1; ' + flag +
                          ' is within bounds. skipping hedging.')
                    return []

            self.pf.add_security(list(ops), 'hedge')

        except IndexError:
            product = hedge_id.split()[0]
            print(product + ' ' + str(loc) +
                  ' price and/or vol data does not exist. skipping...')

        return ops
