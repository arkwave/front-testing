# -*- coding: utf-8 -*-
# @Author: arkwave
# @Date:   2017-11-29 19:56:16
# @Last Modified by:   arkwave
# @Last Modified time: 2017-12-14 15:00:29
import pprint
from abc import ABC, abstractmethod
import numpy as np


# TODO: Add on to this as necessary.
class HedgeModifier(ABC):

    """Abstract class that serves as a template for all Hedge modification objects.
    """
    @abstractmethod
    def run_deltas(self):
        """Returns a tuple consisting of run, run_str and stopval
        run = True/False
        run_str = 'hit' if stops hit, '' if active, 'default' otherwise
        stopval = stop loss value.  
        """
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
        trigger_bounds (TYPE): dictionary mapping UID to trigger bounds around last hedged point,
                               exceeding which monitoring is turned on.
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
        r_dict = {'locked': self.locked,
                  'stop_values': self.stop_values,
                  'current levels': self.current_level,
                  'thresholds': self.thresholds,
                  'anchor_points': self.anchor_points,
                  'active': self.active,
                  'stop levels': self.stop_levels}

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
        breakevens = self.parent.get_breakeven()
        self.trigger_bounds = trigger_values
        self.update_thresholds(breakevens)
        self.update_active()
        stops = dic['value']
        self.stop_levels = stops
        self.update_stop_values()

    def get_thresholds(self, uid=None):
        if uid is None:
            return self.thresholds
        else:
            return self.thresholds[uid]

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

    def get_anchor_points(self, uid=None):
        if uid is None:
            return self.anchor_points
        else:
            try:
                return self.anchor_points[uid]
            except KeyError as e:
                raise KeyError(
                    'The provided uid %s is not in the trailingstop object.' % uid)

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
        self.update_thresholds(self.parent.get_breakeven())
        self.update_active()

    def get_trigger_bounds_numeric(self, breakevens):
        dic = self.trigger_bounds
        new = {}
        for uid in dic:
            bound = dic[uid][0]
            if dic[uid][1] == 'price':
                new[uid] = bound
            elif dic[uid][1] == 'breakeven':
                pdt, mth = uid.split()
                val = breakevens[pdt][mth] * bound
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
                pdt, mth = uid.split()
                val = breakevens[pdt][mth] * bound
                self.thresholds[uid] = (lastpt - val, lastpt + val)

    def update_current_level(self, dic, uid=None, update=True):
        if uid is None:
            self.current_level = dic
        else:
            self.current_level[uid] = dic[uid]

        if update:
            self.update_active()
            self.update_extrema()

    def reset_current_level(self, dic, uid=None):
        self.update_current_level(dic, uid=uid, update=False)

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
                # # print('%s trailing stop monitor is inactive.' % uid +
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
                    breakevens = self.parent.get_breakeven()
                    pdt, mth = uid.split()
                    val = breakevens[pdt][mth] * data[0]
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

        stop_direction = 1 if current_price <= self.anchor_points[uid] else -1
        stopval = self.stop_values[uid]

        print('val, current_price, stopval: ', val, current_price, stopval)
        print('stop direction: ', stop_direction)

        if np.isclose(current_price, stopval):
            return True, stopval

        # case 1: sell-stop and current price <= stop value.
        elif (current_price <= stopval) and stop_direction == -1:
            return True, stopval

        # case 2: buy-stop and current price >= stop value.
        elif (current_price >= stopval) and stop_direction == 1:
            return True, stopval

        return False, None

    def run_deltas(self, uid, price_dict, update=True):
        """The only function that should be called outside of the

        Args:
            uid (string): The underlying ID we're interested in
            price_dict (dic): dictionary of prices.
            update (bool, optional): Description

        Returns:
            tuple: (bool, str). str argument is used to distinguish between the cases where
            run_deltas returns false because trailing stops are hit, where monitoring is inactive.
        """
        # first: update the prices.
        init_active = self.get_active(uid=uid)

        initial_current_vals = self.get_current_level()

        self.update_current_level(price_dict, uid=uid, update=update)

        stopval = self.get_stop_values(uid=uid)

        curr_active = self.get_active(uid=uid)

        if curr_active:
            # case: trailingstop got hit. neutralize all.
            hit, val = self.trailing_stop_hit(uid)
            print('hit, val: ', hit, val)
            if hit:
                print('Update is %s' % update)
                if update:
                    self.unlock(uid)
                    self.update_anchor_points({uid: val}, uid=uid)
                    print('updated anchor points: ', self.get_anchor_points())
                    self.reset_extrema(uid)

                ret = (False, 'hit', stopval)

            # case: active but TS not hit. run deltas.
            else:
                # case: activated on this price move.
                if curr_active != init_active:
                    ret = (True, 'breached', stopval)
                # case: activated from some previous price move.
                else:
                    ret = (True, '', stopval)

        else:
            # case: inactive. default to portfolio default.
            ret = (False, 'default', stopval)

        if not update:
            self.reset_current_level(initial_current_vals)

        return ret
