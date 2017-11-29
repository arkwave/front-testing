# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-11-29 19:56:16
# @Last Modified by:   arkwave
# @Last Modified time: 2017-11-29 21:48:02
import pprint
from abc import ABC, abstractmethod


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
        current_level (TYPE): UID -> current price level dictionary. 
        last_hedgepoints (TYPE): last values these UIDS were hedged at. 
        entry_level (TYPE): a dictionary mappng UID to the price level the tradewas entered at.
        maximals (TYPE): highest UID price values seen so far. 
        minimals (dict): the lowest UID price values seen so far. 
        parent (TYPE): the Hedge object that created this trailingstop. 
        pf (portfolio object): the portfolio this trailing stop object is monitoring.
        stop_levels: copy of params['value']. used to re-compute the stop values
        trigger_bounds (TYPE): dictionary mapping UID to trigger bounds around last hedged point, exceeding which
                               monitoring is turned on. 
        thresholds (dict): dictionary mapping uid to the price level at which 
            trailing stop monitoring is 'turned on' 
        stop_values: uid -> price point upon which we stop out.  

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
        r_dict = {'maximals': self.maximals,
                  'stop_values': self.stop_values,
                  'stop_levels': self.stop_levels,
                  'current levels': self.current_level,
                  'thresholds': self.thresholds,
                  'last hedged': self.anchor_points,
                  'active': self.active,
                  'trigger bounds': self.trigger_bounds}

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

    def update_current_level(self, dic):
        self.current_level = dic
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
                print('%s trailing stop monitor is inactive.' % uid +
                      ' setting stop_value to None.')
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
            TYPE: Description
        """
        # first: update the prices.
        self.update_current_level(price_dict)

        if self.get_active(uid=uid):
            # second: check to see if a trailing stop got hit.
            if self.trailing_stop_hit(uid):
                self.unlock(uid)
                self.update_anchor_points(price_dict)
                self.reset_extrema(uid)
                return False
            return True
        else:
            return False


class HedgeParser:

    """Simple class that determines the ratio of delta to be hedged during any particular call to Hedge.hedge_delta. 

    Figures out the ratio by taking two parameters into account:
        1) the presence of a TrailingStop object 
        2) the specification of a ratio under Hedge.params['delta']['intraday']
        3) relationship between trailingstop trigger bounds and breakeven

    Attributes:
        dic (dict): Dictionary of the hedging paramters passed into the parent hedge object. 
        mod_obj (object): a hedge modification object. currently, the only implemented example is a TrailingStop. Abstract as and when necessary. 
        parent (TYPE): the Hedge object associated with this hedgeparser instance. 

    """

    def __init__(self, dic, parent, mod_obj):
        self.params = dic
        self.parent = parent
        self.mod_obj = mod_obj

    def get_hedge_ratio(self):
        """
        3 cases handled:
            1) trigger bounds are wider than breakeven --> run deltas outside, neutralize all inside. 
            2) trigger bounds are smaller than breakeven --> do nothing. 
            3) trigger bounds are equal to breakeven --> return 1-hedge_ratio.
        Returns:
            TYPE: Description
        """
        ret = {}
        # check to see if ratio parameter exists in the params dict.
        if 'ratio' in self.params:
            hedger_ratio = self.params['ratio']
            # now check to see which case the trailingstop object falls under.
            if isinstance(self.mod_obj, TrailingStop):
                # get the parent Hedge object's breakeven dictionary.
                hedger_interval_dict = self.parent.get_hedge_interval()
                # get the trigger bounds.
                trigger_bounds = self.mod_obj.get_trigger_bounds_numeric()
                assert trigger_bounds.keys() == hedger_interval_dict.keys()
                for uid in trigger_bounds:
                    run_trigger, hedge_interval = trigger_bounds[
                        uid], hedger_interval_dict[uid]
                    # case: run_delta + modification trigger is > hedge
                    # interval stipulated.
                    if run_trigger > hedge_interval or run_trigger < hedge_interval:
                        ret[uid] = 1
                    elif run_trigger == hedge_interval:
                        ret[uid] = 1-hedger_ratio
        else:
            return {uid: 1 for uid in self.parent.pf.uid_price_dict()}
