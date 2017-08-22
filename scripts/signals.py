# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-08-02 18:00:14
# @Last Modified by:   Ananth
# @Last Modified time: 2017-08-03 20:22:14
import pandas as pd
from .util import create_vanilla_option, close_out_deltas
import numpy as np
import copy
from collections import deque


def apply_signal(pf, pdf, vdf, signals, date, sigvals,
                 strat='filo', brokerage=None, slippage=None, metric=None, maintain=True):
    """
    All-purpose function that applies the signal specified to the portfolio. 

    The signals csv has 8 columns:
                    1) Date       : the date that this signal is to be applied. 
                    2) vol_id     : the vol_id of the option in the case where a position is being put on. 
                    3) strike     : the strike of the optionf
                    4) call_put_id: 'call' or 'put'
                    5) lots       : numerical lot number. if empty, uses greek and greekval to specify lottage
                    6) greek      : greek used to specify lottage. 
                    7) greekval   : the value of the greek required. 
                    8) signal     : +i indicates GO LONG (i*lots) or (i*greekval)
                                  : 0/non-existent data indicates NO CHANGE
                                  : -1 indicates GO SHORT (-i*lots) or (-i*greekval)  

                    # Crucial note: whether or not the specified row's option is long or 
                                    short depends on the signal; shorted = True if signal < 0 else False 

    This function does the following:
        1_ grabs all relevant signals for that particular day
            > if empty/no signals --> returns portfolio. 
        2_ Checks the following 3 cases:
            > if sigval is 0, simple add. 
            > if sigval and current signal are the same sign, simple add. 
            > if sigval and current signal are opposite signs:
                - if abs(sigval) > abs(current_signal): liquidate. 
                - if abs(sigval) == abs(current_signal): close out. 
                - if abs(sigval) < abs(current_signal): close out, and add (sigval + signal). 

        # Note: signals are generated based on the start_of_day of the next day. 
        For example, settlements on 03/03 are used to place in a signal on 03/04 
        --> signal application happens after feed_data step corresponding to that given day. 
        --> signal in 03/03 can be conceptualized as having been put in start of the day on 03/04. 


    Args:
        pf (portfolio object): Portfolio being subjected to the signal
        pdf (pandas dataframe): dataframe of prices on that given day. 
        vdf (pandas dataframe): dataframe of vols on that given day. 
        signals (pandas dataframe): signals for that given day. 
        date (pandas Timestamp): current_date in the simulation. 
        sigvals (dict): dictionary mapping (product, call/put) -> current position.  
        strat (str, optional): how to select options during liquidation case; 
                               current inputs are 'filo' or 'dist'
        brokerage (int, optional): Brokerage value.
        slippage (int, optional): Slippage value. 
        metric (None, optional): if strat is set to dist, this is the method used to compare the options.
        maintain (bool, optional): if True, maintains the greek position of each 
                                   set of options rather than letting it decay over time. 

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description


    """

    # sanity check

    cols = ['value_date', 'vol_id', 'strike', 'call/put',
            'lots', 'greek', 'greekval', 'signal', 'pdt']

    if strat == 'dist' and metric is None:
        raise ValueError(
            'signal strat is set to dist but no metric is provided')

    cost = 0
    # actual loop
    for pdt, char in sigvals:
        pdt_signals = signals[(signals.pdt == pdt) & (
            signals['call/put'] == char)][cols]
        # print('pdt_signals: ', pdt_signals)
        # sanity check: ensure there are signals for this product on this day.
        if pdt_signals.empty:
            print(pdt, date)
            print('no signals for ' + pdt + ' on ' + date.strftime('%Y-%m-%d'))
            continue

        for _, row in pdt_signals.iterrows():
            date, vol_id, strike, char, lots, greek, greekval, signal, pdt = row.values

            curr_pos = sigvals[(pdt, char)]

            print('__________ APPLYING ' + vol_id + ' ' +
                  char + ' ' + 'signal: ', signal, '____________')
            print('inputs: ', date, vol_id, strike, char,
                  lots, greek, greekval, signal, pdt)

            # base case: signal is 0, do nothing.
            if signal == 0:
                print(pdt + ' signal on ' + date.strftime('%Y-%m-%d') +
                      ' is 0.')
                if maintain:
                    print('maintaining ' + vol_id + ' ' + char + ' position')
                    # def maintain_position(pf, vol_id, char, strike, pos,
                    # greek, greekval, tol=1000, slippage=None, brokerage=None)
                    pf, cost = maintain_position(
                        pf, pdf, vdf, vol_id, char, strike, curr_pos, lots, greek, greekval)
                print('continuing to next position')
                continue
            # case: Nonzero signal.
            else:
                newpos = curr_pos + signal
                # case: curr position and signal have the same sign, or
                # curr_pos == 0 -> simple adding to portfolio.
                if (curr_pos == 0) or (np.sign(curr_pos) == np.sign(signal)):
                    # shorted = True if signal < 0 else False
                    print('--- signals.apply_signal - add case ---')
                    print('curr_pos: ', curr_pos)
                    print('signal: ', signal)
                    print('vol_id, char: ', vol_id, char)

                    pf, cost = add_position(pf, signal, vdf, pdf, vol_id, char, strike,
                                            date, lots, greek, greekval,
                                            slippage=slippage, brokerage=brokerage, 
                                            maintain=maintain, newpos=newpos)

                    sigvals[(pdt, char)] = newpos
                    print('newpos: ', sigvals[(pdt, char)])

                # case: signal and curr_pos have opposite signs; liquidation
                # necessary.
                elif (np.sign(curr_pos) != np.sign(signal)):
                    # newpos = curr_pos + signal
                    # sub_case 1: net pos = 0.
                    if newpos == 0:
                        print(' --- signals.apply_signal - closing position ---')
                        print('curr_pos: ', curr_pos)
                        print('signal: ', signal)
                        print('vol_id, char: ', vol_id, char)
                        pf, cost = close_position(pf, vol_id, char)

                    # sub_case 2: close out a strict subset of current
                    # position, i.e. abs(curr_pos) > abs(signal).
                    # Example 1: curr_pos = 5, signal = -1, newpos = 4
                    # Example 2: curr_pos = -5, signal = 3, newpos = -2
                    elif abs(curr_pos) > abs(signal):
                        print('--- signals.apply_signal - liquidation case --- ')
                        print('curr_pos: ', curr_pos)
                        print('signal: ', signal)
                        print('volid, char: ', vol_id, char)
                        pf, cost = liquidate_position(pf, vol_id, char, greek,
                                                      greekval, signal, strat=strat,
                                                      maintain=maintain, newpos=newpos)

                    # sub case 3: need to liquidate more than what is owned.
                    # close out and add residual position.
                    # Example 1: curr_pos = -2, signal = 5. newpos = 3.
                    # Example 2: curr_pos = 3, signal = -4. newpos = -1
                    if abs(curr_pos) < abs(signal):
                        print('--- signals.apply_signal - flip case --- ')
                        print('curr_pos: ', curr_pos)
                        print('signal: ', signal)
                        print('volid, char: ', vol_id, char)
                        pf, cost_1 = close_position(pf, vol_id, char)
                        pf, cost_2 = add_position(pf, newpos, vdf, pdf, vol_id, char, strike,
                                                  date, lots, greek, greekval,
                                                  slippage=slippage, brokerage=brokerage, 
                                                  maintain=maintain, newpos=newpos)
                        cost += cost_1 + cost_2

                    # final: once all possible cases are handled, update
                    # position.
                    sigvals[(pdt, char)] = newpos
                    print('newpos: ', newpos)
            print('_____________________ SIGNAL APPLIED _____________________')

    return pf, cost, sigvals


def liquidate_position(pf, vol_id, char, greek, greekval, signal,
                       strat=None, brokerage=None, slippage=None,
                       maintain=None, newpos=None, **kwargs):
    """Helper method that deals with liquidation (i.e. reducing shorts or reducing longs)

    Args:
            pf (portfolio object): Description
            vol_id (string): Description
            char (string): Description
            greek (string): Description
            greekval (float): Description
            signal (int): Description
            strat (str, optional):
            **kwargs: Description
    """

    # initializing variables.
    cost = 0
    indices = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 3}
    pdt = vol_id.split()[0]
    opmth = vol_id.split()[1].split('.')[0]
    ftmth = vol_id.split()[1].split('.')[1]
    relevant_ops = deque([x for x in pf.OTC_options if x.char == char and
                          x.get_product() == pdt and x.get_month() == ftmth and
                          x.get_op_month() == opmth])
    index = indices[greek]
    residual = abs(signal) * greekval

    current_greekval = sum([op.greeks()[index] for op in relevant_ops])
    print('current ' + greek + ' value: ', current_greekval)

    # checks to see if the signal application defaults to maintaining a
    # position as well.

    # TODO: make sure this is necessary and sufficient; does not take into
    # account case where required is > stipulated.
    if maintain:
        print('signals.liquidate_position - maintain flag triggered')
        required_greekval = newpos * greekval

        diff = abs(abs(current_greekval) - abs(required_greekval))
        print('current_greekval: ', current_greekval)
        print('required_greekval: ', required_greekval)
        print('diff: ', diff)
        # print('residual: ', residual)
        residual = diff

    # preallocating toberemoved
    toberemoved = {}
    for product in pf.get_unique_products():
        toberemoved[product] = []

    # if strat == dist, sort according to the distance metric.
    # Note: currently the only distance metric is distance from specified
    # delta value for skews.
    if strat == 'dist':
        if 'metric' not in kwargs:
            raise ValueError(
                'liquidation strategy set to dist, but no metric provided.')
        metric = kwargs['metric']

        if metric[0] == 'delta':
            dval = metric[1]
            relevant_ops = sorted(
                relevant_ops, key=lambda x: abs(abs(x.delta / x.lots) - dval))

    while residual > 0:
        if strat in ['dist', 'filo']:
            op = relevant_ops[-1] if relevant_ops else None
        elif strat == 'fifo':
            op = relevant_ops[0] if relevant_ops else None

        if op is not None:
            print('op selected: ', op)
            greek_per_lot = abs(op.greeks()[index] / op.lots)
            print(greek + ' per lot: ', greek_per_lot)
            lots_required = round(abs(residual / greek_per_lot))
            print('lots required: ', lots_required)

            # case 1: lots required <= lots currently in option.
            if lots_required <= op.lots:
                print('signals.liquidate_position - lots available.')
                residual = 0
                newlots = op.lots - lots_required
                op.update_lots(newlots)
                # TODO: include slippage and brokerage.
                if brokerage is not None:
                    cost += brokerage * lots_required

            # case 2: current ops lots do not suffice.
            else:
                print('signals.liquidate_position: - lots unavailable. ')
                residual -= abs(op.greeks()[index])
                toberemoved[op.get_product()].append(op)
                relevant_ops.remove(op)
                # TODO: include slippage and brokerage.
                if brokerage is not None:
                    cost += brokerage * op.lots

            print(greek + ' remaining to liquidate: ', residual)

        else:
            print('liquidation requirement is beyond portfolio capacity. ending...')
            break

    # updating portfolio after all operations have been completed.
    pf.update_sec_by_month(False, 'OTC', update=True)

    # removing securities tagged for removal.
    for pdt in toberemoved:
        sec = toberemoved[pdt]
        if sec:
            print('removing ' + str([str(x) for x in sec]))
            pf.remove_security(sec, 'OTC')

    # debug statements
    pdt = vol_id.split()[0]
    opmth = vol_id.split()[1].split('.')[0]
    ftmth = vol_id.split()[1].split('.')[1]
    print('vol_id, pdt, opmth, ftmth, char: ', vol_id, pdt, opmth, ftmth, char)
    new_ops = deque([x for x in pf.OTC_options if
                     (x.char == char and x.get_product() == pdt and
                      x.get_month() == ftmth and x.get_op_month() == opmth)])

    print('ops after liquidation: ', str([str(x) for x in new_ops]))

    new_current_greekval = sum([op.greeks()[index] for op in new_ops])

    print('new current val: ', new_current_greekval)
    return pf, cost


# TODO: add in a flag that checks if all options of a given vol_id have
# been removed; if so, closes out the deltas too.
# TODO: once hedge futures become more of a thing, need to ensure that
# this closes them out too.
def close_position(pf, vol_id, char):
    """ Helper function that closes out the entire OTC position associated with a vol_id and a call/put.

    Args:
            pf (portfolio object): Description
            vol_id (string): Description
            char (string): Description
    """
    print('closing ' + vol_id + ' ' + char + 's')
    cost = 0
    pdt = vol_id.split()[0]
    opmth = vol_id.split()[1].split('.')[0]
    ftmth = vol_id.split()[1].split('.')[1]
    ops = [x for x in pf.OTC_options if x.get_product() == pdt and
           x.get_month() == ftmth and x.get_op_month() == opmth and x.char == char]

    for op in ops:
        cost += op.get_price() if op.shorted else -op.get_price()

    toberemoved = copy.copy(ops)
    pf.remove_security(toberemoved, 'OTC')

    # sanity check: ensure that there are still OTC options corresponding to
    # this ftmth. Else, close out deltas.
    volid_ops = pf.OTC[pdt][ftmth][0]
    volid_deltas = []
    if (pdt in pf.hedges) and (ftmth in pf.hedges[pdt]):
        volid_deltas += pf.hedges[pdt][ftmth][1]
    if (pdt in pf.OTC) and (ftmth in pf.OTC[pdt]):
        volid_deltas += pf.OTC[pdt][ftmth][1]
    if len(volid_ops) == 0 and len(volid_deltas) > 0:
        price = volid_deltas[0].get_price()
        iden = (pdt, ftmth, price)
        print('iden: ', iden)
        pf, cost_2 = close_out_deltas(pf, [iden])
        cost += cost_2

    print('finished closing out ' + vol_id + ' ' + char + 's')

    return pf, cost


def add_position(pf, signal, vdf, pdf, vol_id, char, strike, date,
                 lots, greek, greekval, slippage=None, brokerage=None,
                 maintain=None, newpos=None):
    """Helper method that creates and adds a position. 

    Args:
            pf (TYPE): Description
            signal (TYPE): Description
            vdf (TYPE): Description
            pdf (TYPE): Description
            vol_id (TYPE): Description
            char (TYPE): Description
            date (TYPE): Description
            lots (TYPE): Description
            greek (TYPE): Description
            greekval (TYPE): Description
    """
    cost = 0
    pdt = vol_id.split()[0]
    opmth = vol_id.split()[1].split('.')[0]
    ftmth = vol_id.split()[1].split('.')[1]
    indices = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 3}
    tobeadded = []
    shorted = True if signal < 0 else False
    lots = None if np.isnan(lots) else lots
    index = indices[greek]

    # maintain check
    if maintain:
        print('signals.add_position - maintain flag triggered.')
        desired_level = abs(newpos) * greekval
        relevant_ops = [x for x in pf.OTC_options if x.get_product() == pdt and 
                        x.get_month() == ftmth and x.get_op_month() == opmth and x.char == char]
        current_level = abs(sum([op.greeks()[index] for op in relevant_ops]))
        print('signals.add_position - current_level: ', current_level)
        print('signals.add_position - desired_level: ', desired_level)
        greekval = desired_level - current_level

    if not maintain:
        greekval = greekval * abs(signal)
    # adding option.
    # for i in range(abs(signal)):
    op = create_vanilla_option(vdf, pdf, vol_id, char, shorted,
                               date, lots=lots, greek=greek, greekval=greekval, strike=strike)
    print('signals.add_position - created ' + str(op))
    tobeadded.append(op)

    # computing cost.
    if slippage is not None:
        cost += slippage

    if brokerage is not None:
        cost += sum([op.lots for op in tobeadded]) * brokerage

    # debug statements:
    pf.add_security(tobeadded, 'OTC')
    relevant_ops = [x for x in pf.OTC_options if x.get_product() == pdt and 
                    x.get_month() == ftmth and x.get_op_month() == opmth and x.char == char]
    current_level = sum([op.greeks()[index] for op in relevant_ops])

    print('new current level: ', current_level)

    return pf, cost


def maintain_position(pf, pdf, vdf, vol_id, char, strike, pos, lots, 
                      greek, greekval, tol=1000, slippage=None, brokerage=None):
    """Helper method that maintains the current position, guarding against 
    greeks decaying over time. 

    Args:
        pf (prtfolio object): the portfolio being handled
        pdf (pandas dataframe): Description
        vdf (pandas dataframe): Description
        vol_id (string): vol_id of the position being managed.
        char (string): call/put
        strike (float): strike on which any new position is to be added. 
        pos (int): value of the position 
        greek (string): the greek on the basis of which this position is being computed
        greekval (float): value of the greek 
        tol (float, optional): Description
        slippage (None, optional): Description
        brokerage (None, optional): Description
    """
    cost = 0
    indices = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 3}
    pdt = vol_id.split()[0]
    opmth = vol_id.split()[1].split('.')[0]
    ftmth = vol_id.split()[1].split('.')[1]
    relevant_ops = [x for x in pf.OTC_options if x.char == char and
                    x.get_product() == pdt and x.get_month() == ftmth and 
                    x.get_op_month() == opmth]
    index = indices[greek]

    curr_level = sum([op.greeks()[index] for op in relevant_ops])

    desired_level = pos * greekval

    if abs(desired_level - curr_level) < tol:
        print('current: ', curr_level)
        print('desired: ', desired_level)
        print('signals.maintain_position - ' + vol_id + ' ' +
              char + ' ' + greek + ' position within tolerance.')
        return pf, 0

    else:
        # case: need to add greek; long option for gamma/vega, short for theta.
        if desired_level > curr_level:
            print('signals.maintain_position - desired > current')
            sig = 1 if greek in ['gamma', 'vega'] else -1

        # need to remove greek; short option for gamma/vega, long for theta.
        elif desired_level < curr_level:
            print('signals.maintain_position - desired < current')
            sig = -1 if greek in ['gamma', 'vega'] else 1

        print('desired ' + greek + ': ', desired_level)
        print('current ' + greek + ': ', curr_level)
        diff = abs(desired_level - curr_level)
        # add_position(pf, signal, vdf, pdf, vol_id, char, strike, date,
        #      lots, greek, greekval, slippage=None, brokerage=None)
        date = pd.to_datetime(pdf.value_date.unique()[0])
        lots = None if np.isnan(lots) else lots
        pf, cost = add_position(pf, sig, vdf, pdf, vol_id, char, strike, date,
                                np.nan, greek, diff, slippage=slippage, brokerage=brokerage)
        # sanity checking.
        new_ops = [x for x in pf.OTC_options if x.char == char and
                   x.get_product() == pdt and x.get_month() == ftmth and 
                   x.get_op_month() == opmth]

        new_curr_level = sum([op.greeks()[index] for op in new_ops])
        print('new current: ', new_curr_level)

        return pf, cost
