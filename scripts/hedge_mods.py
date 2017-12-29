# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-11-29 19:56:16
# @Last Modified by:   arkwave
# @Last Modified time: 2017-12-29 17:39:57
import pprint
from abc import ABC, abstractmethod
import numpy as np


# TODO: Add on to this as necessary.
class HedgeModifier(ABC):

    """Abstract class that serves as a template for all Hedge modification objects. 
    """
    @abstractmethod
    def run_deltas(self):
        pass


class TrailingStop(HedgeModifier):
    """
    Class that handles all the details required for trailing stops. 

    Attributes:
        active (dict): UID -> true or false. Determines whether or not stop loss 
                              monitoring has been triggered. 
        anchor_points (TYPE): UID -> reference points from which trigger bounds are applied. 
        current_level (TYPE): UID -> current price level dictionary. 
        entry_level (TYPE): a dictionary mappng UID to the price level the tradewas entered at.
        locked (TYPE): Description
        maximals (TYPE): highest UID price values seen so far. 
        minimals (dict): the lowest UID price values seen so far. 
        parent (TYPE): the Hedge object that created this trailingstop. 
        pf (portfolio object): the portfolio this trailing stop object is monitoring.
        stop_levels: copy of params['value']. used to re-compute the stop values
        trigger_bounds (TYPE): dictionary mapping UID to trigger bounds around anchor point, 
                               exceeding which monitoring is turned on. 
        trigger_bounds_numeric (TYPE): uid -> anchor point + trigger bound for each UID. 


    """

    def __init__(self, entry_level, params, pf, last_hedged, parent):
        self.parent = parent
        self.entry_level = entry_level
        self.current_level = entry_level.copy()
        self.pf = pf
        self.trigger_bounds = None
        self.maximals = entry_level.copy()
        self.minimals = entry_level.copy()
        self.stop_values, self.thresholds, self.active = {}, {}, None
        self.stop_levels = None
        self.anchor_points = last_hedged
        self.locked = {uid: False for uid in self.current_level}
        self.trigger_bounds_numeric = None
        self.process_params(params)

    def __str__(self):
        r_dict = {'locked': self.locked,
                  'stop_values': self.stop_values,
                  'current levels': self.current_level,
                  'thresholds': self.thresholds,
                  'anchor_points': self.anchor_points,
                  'active': self.active}

        return pprint.pformat(r_dict)

    def process_params(self, dic):
        """Helper method that processes the 
           intraday params specified in dic. 

        Args:
            dic (TYPE): dictionary of intraday parameters, of the following form:
                {'trigger': {uid1: (30, price), uid2: (50, price)}, 'value': 
                {uid1: (-31.5, price), uid2: (-1, price}}
        Returns:
            tuple: trigger levels: price points at which to trigger a stop. 
                   active: dictionary mapping UIDS to True/False. Determines
                   whether stop loss monitoring has been triggered or not. 
        """
        # first: sanity check the inputs.
        try:
            # assert 'type' in dic
            assert 'trigger' in dic
            assert 'value' in dic
            assert isinstance(dic['value'], dict)
        except AssertionError as e:
            raise ValueError(
                "Something is wrong with the intraday input params passed in. ", dic)

        # divide up the input parameters into stop values, thresholds and
        # actives respectively.
        trigger_values = dic['trigger']
        breakevens = self.pf.breakeven()
        self.trigger_bounds = trigger_values
        self.update_thresholds(breakevens)
        self.update_active()
        stops = dic['value']
        self.stop_levels = stops
        self.update_stop_values()

    def get_thresholds(self):
        return self.thresholds

    def get_trigger_bounds(self):
        return self.trigger_bounds

    def get_locks(self):
        return self.locked

    # getter methods.
    def get_minimals(self):
        return self.minimals

    def get_maximals(self):
        return self.maximals

    def get_entry_level(self):
        return self.entry_level

    def get_current_level(self, uid=None):
        return self.current_level if uid is None else self.current_level[uid]

    def get_stop_values(self, uid=None):
        return self.stop_values if uid is None else self.stop_values[uid]

    def get_active(self, uid=None):
        if uid is None:
            return self.active
        else:
            if uid in self.active:
                return self.active[uid]
            else:
                raise ValueError(
                    "%s is not in the portfolio passed into the TrailingStop object" % uid)

    def get_anchor_points(self):
        return self.anchor_points

    # Setter/Update methods.
    def update_anchor_points(self, dic, uid=None):
        """Summary

        Args:
            dic (TYPE): Description
            uid (None, optional): Description
        """
        if uid is not None:
            self.anchor_points[uid] = dic[uid]
        else:
            self.anchor_points = dic
        self.update_thresholds(self.pf.breakeven())
        self.update_active()

    def get_trigger_bounds_numeric(self, breakevens):
        dic = self.trigger_bounds
        new = {}
        for uid in dic:
            bound = dic[uid][0]
            if dic[uid][1] == 'price':
                new[uid] = bound
            elif dic[uid][1] == 'breakeven':
                val = breakevens[uid] * bound
                new[uid] = val
        return new

    def update_thresholds(self, breakevens):
        """Summary

        Args:
            breakevens (TYPE): Description
        """
        trigger_values = self.trigger_bounds
        for uid in trigger_values:
            lastpt = self.anchor_points[uid]
            bound = trigger_values[uid][0]
            if trigger_values[uid][1] == 'price':
                # case: stop loss monitoring is active upon a certain flat
                # price move.
                self.thresholds[uid] = (lastpt - bound, lastpt + bound)

            elif trigger_values[uid][1] == 'breakeven':
                # case: stop loss monitoring is active upon a certain BE move.
                val = breakevens[uid] * bound
                self.thresholds[uid] = (lastpt - val, lastpt + val)

    def update_current_level(self, dic, uid=None):
        if uid is None:
            self.current_level = dic
        else:
            self.current_level[uid] = dic[uid]

        self.update_active()
        self.update_extrema()

    def reset_extrema(self, uid):
        self.maximals[uid] = self.current_level[uid]
        self.minimals[uid] = self.current_level[uid]
        self.update_stop_values()

    def update_extrema(self):
        for uid in self.current_level:
            if abs(self.current_level[uid]) > abs(self.maximals[uid]):
                self.maximals[uid] = self.current_level[uid]
            if abs(self.current_level[uid]) < abs(self.minimals[uid]):
                self.minimals[uid] = self.current_level[uid]
        self.update_stop_values()

    # need to figure out pathological cases.
    def update_stop_values(self, maximals=None, minimals=None):
        """
        Updates the values at which a given underlying will stop out by comparing current value to bounds 
        specified in self.threshold. Two cases handled:
            1) if current < lower bound: it's a buy stop. stop value = min + stop_level. 
            2) if current > upper bound: it's a sell stop. stop_value = max - stop_level 

        Args:
            maximals (None, optional): Description
            minimals (None, optional): Description
        """
        maximals = self.maximals if maximals is None else maximals
        minimals = self.minimals if minimals is None else minimals
        assert self.stop_levels is not None
        for uid in self.stop_levels:
            if not self.get_active(uid=uid):
                # print('%s trailing stop monitor is inactive.' % uid +
                #       ' setting stop_value to None.')
                self.stop_values[uid] = None
            else:
                lower, upper = self.thresholds[uid]
                data = self.stop_levels[uid]
                if data[1] == 'price':
                    # case 1: current < lower bound.
                    if self.current_level[uid] < lower:
                        self.stop_values[uid] = minimals[uid] + data[0]
                    elif self.current_level[uid] > upper:
                        self.stop_values[uid] = maximals[uid] - data[0]

                elif data[1] == 'breakeven':
                    breakevens = self.pf.breakeven()
                    val = breakevens[uid] * data[0]
                    if self.current_level[uid] < lower:
                        self.stop_values[uid] = minimals[uid] + val
                    elif self.current_level[uid] > upper:
                        self.stop_values[uid] = maximals[uid] - val

    def unlock(self, uid):
        if self.locked[uid]:
            self.locked[uid] = False

    def update_active(self):
        # true if price > upper threshold or < lower threshold.
        if self.active is not None:
            for uid in self.active:
                try:
                    new = ((abs(self.current_level[uid]) > abs(self.thresholds[uid][1])) or
                           (abs(self.current_level[uid]) < abs(self.thresholds[uid][0])))
                    newlock = True if new else False
                    if not self.locked[uid]:
                        self.active[uid] = new
                    # ensures that is set only if it is true.
                    if newlock:
                        self.locked[uid] = newlock

                except KeyError as e:
                    print('current_level: ', self.current_level)
                    print('thresholds: ', self.thresholds)
                    raise KeyError(
                        'Key %s not in current_level and/or threshold dictionaries' % uid)
        else:
            self.active = {uid: (abs(self.current_level[uid]) > abs(self.thresholds[uid][1])) or
                                (abs(self.current_level[uid]) < abs(
                                    self.thresholds[uid][0]))
                           for uid in self.current_level}

    def trailing_stop_hit(self, uid, val=None):
        """
        Helper function that checks to see if the trailing stop has been hit. if so, return true. 

        Args:
            uid (string): the underlying ID we are interested in 
            val (float, optional): an explicit value to compare. 

        Returns:
            TYPE: Description
        """
        # base case: stop monitoring not active.
        if not self.get_active(uid=uid):
            return False
        # get the current price and the direction of the stop.
        current_price = self.get_current_level(uid=uid) if val is None else val
        stop_direction = 1 if current_price < self.anchor_points[uid] else -1

        # case 1: sell-stop and current price <= stop value.
        if (current_price <= self.stop_values[uid]) and stop_direction == -1:
            return True

        # case 2: buy-stop and current price >= stop value.
        elif current_price >= self.stop_values[uid] and stop_direction == 1:
            return True

        return False

    def run_deltas(self, uid, price_dict):
        """The only function that should be called outside of the 

        Args:
            uid (string): The underlying ID we're interested in 
            price_dict (dic): dictionary of prices. 

        Returns:
            tuple: (bool, str). str argument is used to distinguish between the cases where
            run_deltas returns false because trailing stops are hit, where monitoring is inactive. 
        """
        # first: update the prices.
        print('price dict: ', price_dict)
        print('>>>> TrailingStop: old params pre-update <<<<')
        print(self.__str__())
        print('>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<')
        self.update_current_level(price_dict, uid=uid)

        print('>>>> TrailingStop: New Params post-update <<<<')
        print(self.__str__())
        print('>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<')

        if self.get_active(uid=uid):
            # case: trailingstop got hit. neutralize all.
            if self.trailing_stop_hit(uid):
                self.unlock(uid)
                self.update_anchor_points(price_dict, uid=uid)
                self.reset_extrema(uid)
                return False, 'hit'
            # case: active but TS not hit. run deltas.
            return True, ''
        else:
            # case: inactive. default to portfolio default.
            return False, 'default'

    def run_delta_iterative(self, uid, lst):
        fin = []
        for price in lst:
            print('price being run: ', price)
            run, typestr = self.run_deltas(uid, {uid: price})
            if not run:
                fin.append(price)

        return fin


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
            print('else case: ratio not specified')
            print('params: ', self.params)
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
            print('HedgeParser - EOD case.')
            return {uid: 0 for uid in uids}

        if self.mod_obj is None:
            print('HedgeParser - No HedgeModifier detected.')
            ret = {uid: 1-hedger_ratio for uid in uids}

        elif isinstance(self.mod_obj, TrailingStop):
            print('HedgeParser - TrailingStop HedgeModifier detected.')
            # get the parent Hedge object's breakeven dictionary.
            hedger_interval_dict = self.parent.get_hedge_interval()
            # get the trigger bounds.
            trigger_bounds = self.mod_obj.get_trigger_bounds_numeric(
                self.parent.get_breakeven())

            # sanity check the inputs.
            assert trigger_bounds.keys() == hedger_interval_dict.keys()
            assert set(trigger_bounds.keys()) == uids

            for uid in uids:

                run_deltas, type_str = \
                    self.mod_obj.run_deltas(uid, self.pf.uid_price_dict())

                if run_deltas:
                    # case: we want to run the deltas of this uid.
                    print('Case (1): run %s deltas' % uid)
                    run_trigger, hedge_interval = trigger_bounds[
                        uid], hedger_interval_dict[uid]

                    print('%s run_trigger: ' % uid, run_trigger)
                    print('%s hedge interval: ' % uid, hedge_interval)

                    # case: run_delta + modification trigger != hedge
                    # interval stipulated.
                    if run_trigger > hedge_interval or run_trigger < hedge_interval:
                        print('Case (1.1): hedge interval != run trigger bounds')
                        ret[uid] = 1

                    elif run_trigger == hedge_interval:
                        print('Case (1.2): hedge interval == run trigger bounds')
                        ret[uid] = 1-hedger_ratio

                else:
                    print('Case (2): Do not run deltas for %s' % uid)
                    print('trailing stop %s' %
                          ('hit' if type_str == 'hit' else 'not hit'))
                    ret[uid] = 0 if type_str == 'hit' else 1 - \
                        hedger_ratio

        return ret

    def relevant_price_move(self, uid, val, comparison=None):
        """Method that asserts whether or not a price move is relevant. A few cases are taken into account:
        1) price move > hedge interval, stop monitoring inactive -> True. 
        2) price move > hedge interval, stop monitoring active -> if stop hit, True else False
        3) price move < hedge interval, stop monitoring inactive -> False. 
        4) price move < hedge interval, stop monitoring active -> if stop hit, True else False. 


        Args:
            uid (str): the uid being handled, e.g. C  Z7
            val (float): the price point we're interested in. 
            comparison (float, optional): the value to be compared against. 
                                          used when the HedgeParser is called in granularize. 


        Returns:
            tuple: (bool, float, dict), representing the following:
            1) bool - indicates if the move should be considered valid or not. 
            2) float - the move multiple (e.g. how many times of hedge interval)
            3) list - intermediate price points. returns empty list in the following cases:
                      1_ price point is a relevant move but not hedged. 
                      2_ price point is irrelevant and:
                        > HedgeModifier monitoring is inactive (i.e. run_type == 'default')
                        > HedgeModifier monitoring is active (i.e. run_type == '')

            Aim: return a list of intermediate price points to be used in the create_intermediate_rows
            function in scripts/prep_data. 


        """

        # TODO: figure out how exactly the hegemod object is going to be
        # updated. Temporary solution: just use run_deltas?

        mod = self.get_mod_obj()
        hedger = self.get_parent()

        interval = hedger.get_hedge_interval(uid)
        int_mult = -1 if comparison > val else 1

        prices = list(np.arange(comparison, val, interval*int_mult))

        # base case: No HedgeModifier object is found.
        if self.get_mod_obj() is None:
            relevant, mult = self.parent.is_relevant_price_move(
                uid, val, comparison=comparison)
            return (relevant, mult, prices)

        else:
            for price in prices:
                hedger_relevant, mult = hedger.is_relevant_price_move(
                    uid, price, comparison=comparison)
                # cases to consider:
                if hedger_relevant:
                    run_deltas, run_type = mod.run_deltas(uid, {uid: val})
                    # case: mod returns that deltas should be run.
                    if run_deltas:
                        return False
                    # case: no monitoring is active.
                    elif run_type == 'default':
                        return True
                    # case: stop is hit.
                elif run_type == 'hit':
                    # need to granularize the actual hit value
                    # reset hedges there.
                    pass

                else:
                    # sanity checking case where it's an irrelevant move.
                    assert len(prices) == 1
                    assert prices[0] == val
                    run_deltas, run_type = mod.run_deltas(uid, {uid: val})
                    # case 4: run_deltas returns true.
                    if run_deltas:
                        return False
                    # case: no monitoring + irrelevant move.
                    elif run_type == 'default':
                        return False
                    elif run_type == 'hit':
                        return True
