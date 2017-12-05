# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-12-05 13:48:47
# @Last Modified by:   arkwave
# @Last Modified time: 2017-12-05 13:49:26

import numpy as np
from .hedge_mods import HedgeModifier, TrailingStop


class HedgeParser:

    """Simple class that determines the ratio of delta to be hedged during any
        particular call to Hedge.hedge_delta.

    Figures out the ratio by taking two parameters into account:
        1) the presence of a TrailingStop object
        2) the specification of a ratio under Hedge.params['delta']['intraday']
        3) relationship between trailingstop trigger bounds and breakeven

    Attributes:
        dic (dict): Dictionary of the hedging paramters passed into the parent hedge object.
        mod_obj (object): a hedge modification object. currently, the only implemented example
                         is a TrailingStop. Abstract as and when necessary.
        parent (TYPE): the Hedge object associated with this hedgeparser instance.
    """

    def __init__(self, parent, dic, mod_obj, pf):
        self.params = dic
        self.parent = parent
        self.mod_obj = mod_obj
        self.pf = pf
        if 'ratio' in self.params:
            self.hedger_ratio = self.params['ratio']
        else:
            # print('else case: ratio not specified')
            # print('params: ', self.params)
            self.hedger_ratio = 1

        if self.mod_obj is not None:
            assert isinstance(self.mod_obj, HedgeModifier)

    def get_hedger_ratio(self):
        return self.hedger_ratio

    def get_mod_obj(self):
        return self.mod_obj

    def get_parent(self):
        return self.parent

    def parse_hedges(self, flag):
        """
        3 cases handled:
            1) trigger bounds are wider than breakeven --> run deltas outside,
                                                           neutralize default ratio inside.
            2) trigger bounds are smaller than breakeven --> do nothing.
            3) trigger bounds are equal to breakeven --> return 1-hedge_ratio.

        Returns:
            TYPE: dictionary mapping UIDs to proportion (0 to 1) of deltas to run.
        """
        ret = {}
        uids = self.pf.get_unique_uids()
        hedger_ratio = self.hedger_ratio

        if flag == 'eod':
            # print('HedgeParser - EOD case.')
            return {uid: 0 for uid in uids}

        if self.mod_obj is None:
            # print('HedgeParser - No HedgeModifier detected.')
            ret = {uid: 1-hedger_ratio for uid in uids}

        elif isinstance(self.mod_obj, TrailingStop):
            # print('HedgeParser - TrailingStop HedgeModifier detected.')
            # get the parent Hedge object's breakeven dictionary.
            hedger_interval_dict = self.parent.get_hedge_interval()
            # get the trigger bounds.
            trigger_bounds = self.mod_obj.get_trigger_bounds_numeric(
                self.parent.get_breakeven())

            # sanity check the inputs.
            assert trigger_bounds.keys() == hedger_interval_dict.keys()
            assert set(trigger_bounds.keys()) == uids

            for uid in uids:

                run_deltas, type_str, stopval = \
                    self.mod_obj.run_deltas(uid, self.pf.uid_price_dict())

                if run_deltas:
                    # case: we want to run the deltas of this uid.
                    # print('Case (1): run %s deltas' % uid)
                    run_trigger, hedge_interval = trigger_bounds[
                        uid], hedger_interval_dict[uid]

                    # print('%s run_trigger: ' % uid, run_trigger)
                    # print('%s hedge interval: ' % uid, hedge_interval)

                    # case: run_delta + modification trigger != hedge
                    # interval stipulated.
                    if run_trigger > hedge_interval or run_trigger < hedge_interval:
                        # print('Case (1.1): hedge interval != run trigger
                        # bounds')
                        ret[uid] = 1

                    elif run_trigger == hedge_interval:
                        # print('Case (1.2): hedge interval == run trigger
                        # bounds')
                        ret[uid] = 1-hedger_ratio

                else:
                    # print('Case (2): Do not run deltas for %s' % uid)
                    # print('trailing stop %s' % ('hit' if type_str == 'hit' else 'not hit'))
                    ret[uid] = 0 if type_str == 'hit' else 1 - hedger_ratio

        return ret

    def relevant_price_move(self, uid, val, comparison=None):
        """ Helper function that takes in a uid, price value and an optional comparison
        price point returns a list of valid price moves between these two points, taking
        hedgemod constraints and hedging intervals into account. s

        Args:
            uid (str): the uid being handled, e.g. C  Z7
            val (float): the price point we're interested in.
            comparison (float, optional): the value to be compared against.
                                          used when the HedgeParser is called in granularize.


        Returns:
            list of valid price points between comparison and val.
            > empty list returned in the following scenarios:
                1) hedge mod is None and invalid price move.
                2) hedge mod is not None:
                    - no monitoring active and invalid price move.
                    - uid is active and val is further away from stop value than comparison.


        """
        hedger = self.get_parent()

        interval = hedger.get_hedge_interval(uid)
        int_mult = -1 if comparison > val else 1

        # generate a sequence of prices between comparison and val.
        # this is uni-directional by definition --> only need to care
        # about a single interval rather than caring about +- interval.
        prices = self.gen_prices(comparison, val, int_mult*interval, uid)

        return prices

    def gen_prices(self, start, end, interval, uid, hedgemod=None):
        """Generates a price series between start and end, subject to
        hedgemod and

        Args:
            start (float): start point.
            end (float): end point.
            interval (float): the hedging interval considered.
            hedgemod (HedgeModifier): HedgeModifier object.

        Returns:
            list -> list of valid prices between start and end, subject to hedgemod constraints.

        """
        print('-------------------------------------------------------------------')
        print('START: ', start)
        print('END: ', end)
        print('INTERVAL: ', interval)

        def f(start, end, interval):
            if start > end and interval > 0:
                return True
            elif start > end and interval < 0:
                return False
            elif start < end and interval > 0:
                return False
            elif start < end and interval < 0:
                return True

        hedgemod = self.get_mod_obj() if hedgemod is None else hedgemod

        # edge case 1: No hedgemod present.
        if hedgemod is None:
            mult = np.sign(interval)
            return list(np.arange(start+interval,
                                  end+(mult*1e-9), interval))

        # isolate the boolean condition to be tested.

        active = hedgemod.get_active(uid=uid)
        print('ACTIVE: ', active)

        lst = []

        prev = start
        curr = start + interval
        done = f(curr, end, interval)

        # edge case 2: uid is already active.
        if active:
            print('active case.')
            # # print('tstop initial: ', hedgemod)
            run, runtype, stopval = hedgemod.run_deltas(uid, {uid: end})
            # print('tstop final: ', hedgemod)
            # # print('run, str, stopval: ', run, runtype, stopval)
            if runtype == 'hit':
                # print('active - hit case')
                lst.append(stopval)
                prev = curr
                curr = stopval
            else:
                # if direction to stop value from curr != direction of interval,
                # we will never hit stop --> might as well return
                # print('active - fail case')
                if np.sign(stopval - end) != np.sign(interval):
                    return lst

        # edge case 3: one interval move surpasses end
        if (curr > end and interval > 0) or (curr < end and interval < 0):
            print('edge case 3')
            run, runtype, stopval = hedgemod.run_deltas(uid, {uid: curr})
            if runtype == 'hit':
                return [stopval]
            else:
                return []

        # main loop

        print('tstop before main loop: ', hedgemod)
        print('current: ', curr)

        count = 0
        while (not done and count < 10):
            print('main loop. curr, interval, active -  ', curr, interval, active)
            run, runtype, stopval = hedgemod.run_deltas(uid, {uid: curr})
            print('run, runtype, stopval: ', run, runtype, stopval)
            # case: price move activates monitoring.
            if run:
                print('HedgeParser.gen_prices: running deltas at ' + str(curr))
                # case: need to check for threshold between start and curr.
                lower, upper = hedgemod.get_thresholds(uid=uid)
                print('lower, upper: ', lower, upper)
                print('prev, curr: ', prev, curr)
                # if curr == upper or curr == lower:
                #     print('curr is equal to threshold')
                #     lst.append(curr)

                if curr > upper and prev < upper:
                    lst.append(upper)
                elif curr < lower and prev > lower:
                    lst.append(lower)

                # break
                prev = curr
                curr += interval

            else:
                # case: no breaches. price move is relevant.
                if (runtype == 'default') and not active:
                    lst.append(curr)
                    prev = curr
                    curr += interval

                # case: stops are hit.
                elif runtype == 'hit':
                    try:
                        assert active
                    except AssertionError as e:
                        raise AssertionError('tstop : ', hedgemod.__str__())
                    # case 1: stop val is in between start and curr.
                    if (stopval < curr and interval > 0) or (stopval > curr and interval < 0):
                        lst.append(stopval)
                        prev = curr
                        curr = stopval

                    # case 2: stop val == curr.
                    elif stopval == curr:
                        lst.append(curr)
                        prev = curr
                        curr += interval
                    active = False

            done = f(curr, end, interval)
            count += 1

        return lst
