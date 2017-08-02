import pandas as pd
from .portfolio import Portfolio
from .util import create_vanilla_option, close_out_deltas
import numpy as np
import copy


# TODO: make sure that adding/removing is okay and won't mess with PnL
# calculations.
def apply_signal(pf, pdf, vdf, signals, date, sigvals,
                 strat='dist', brokerage=None, slippage=None, metric=None):
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
            8) signal 	  : +i indicates GO LONG (i*lots) or (i*greekval)
                                      : 0/non-existent data indicates NO CHANGE
                                      : -1 indicates GO SHORT (-i*lots) or (-i*greekval)  

            # Crucial note: whether or not the specified row's option is long or short depends on the signal; shorted = True if signal < 0 else False 

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
            strat (str, optional): how to select options during liquidation case; current inputs are 'filo' or 'dist'
            brokerage (int, optional): Brokerage value.
            slippage (int, optional): Slippage value. 


    """

    # sanity check
    if strat == 'dist' and metric is None:
        raise ValueError(
            'signal strat is set to dist but no metric is provided')

    cost = 0
    # actual loop
    for pdt in sigvals:
        pdt_signals = signals[signals.product == pdt]
        # sanity check: ensure there are signals for this product on this day.
        if pdt_signals.empty:
            print('no signals for ' + pdt + ' on ' + date.strftime('%Y-%m-%d'))
            continue

        for _, row in pdt_signals.iterrows():
            date, vol_id, char, lots, greek, greekval, signal, pdt = row.values
            curr_pos = sigvals[(pdt, char)]

            # base case: signal is 0, do nothing.
            if signal == 0:
                print(pdt + ' signal on ' + date.strftime('%Y-%m-%d') +
                      ' is 0; continuing to next product')
                continue
            # case: Nonzero signal.
            else:
                # case: curr position and signal have the same sign, or
                # curr_pos == 0 -> simple adding to portfolio.
                if (curr_pos == 0) or (np.sign(curr_pos) == np.sign(signal)):
                    # shorted = True if signal < 0 else False
                    print('--- signals.apply_signal - add case ---')
                    print('curr_pos: ', curr_pos)
                    print('signal: ', signal)
                    print('vol_id, char: ', vol_id, char)
                    pf, cost = add_position(pf, signal, vdf, pdf, vol_id, char,
                                            date, lots, greek, greekval,
                                            slippage=slippage, brokerage=brokerage)
                    sigvals[(pdt, char)] += signal
                    print(' --- add complete ---')

                # case: signal and curr_pos have opposite signs; liquidation
                # necessary.
                elif (np.sign(curr_pos) != np.sign(signal)):
                    newpos = curr_pos + signal
                    # sub_case 1: net pos = 0.
                    if newpos == 0:
                        print(' --- signals.apply_signal - zero case ---')
                        print('curr_pos: ', curr_pos)
                        print('signal: ', signal)
                        print('vol_id, char: ', vol_id, char)
                        pf, cost = close_position(pf, vol_id, char)
                        print('--- position closed ---')

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
                                                      greekval, signal)
                        print(' --- liquidation complete --- ')

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
                        pf, cost_2 = add_position(pf, newpos, vdf, pdf, vol_id, char,
                                                  date, lots, greek, greekval,
                                                  slippage=slippage, brokerage=brokerage)
                        cost += cost_1 + cost_2
                        print('--- position flipped ---')
                    # final: once all possible cases are handled, update
                    # position.
                    sigvals[(pdt, char)] = newpos

    return pf, cost, sigvals


def liquidate_position(pf, vol_id, char, greek, greekval, signal, strat=None, **kwargs):
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
    indices = {'delta': 0, 'gamma': 1, 'theta': 2, 'vega': 4}
    pdt = vol_id.split()[0]
    opmth = vol_id.split()[1].split('.')[0]
    ftmth = vol_id.split()[1].split('.')[1]
    relevant_ops = [x for x in pf.OTC_options if x.char == char and
                    x.get_product() == pdt and x.get_month() == ftmth
                    and x.get_op_month() == opmth]
    index = indices[greek]

    curr_greek_level = sum([op.greeks()[index] for op in relevant_ops])

    desired_greek_level = curr_greek_level + (signal * greekval)

    # if strat == dist, sort according to the distance metric.
    # figure out how to insitute a general comparator for the dist metric.
    # filo is easy, just pop from the list.
    if strat == 'dist':
        if 'metric' not in kwargs:
            raise ValueError(
                'liquidation strategy set to dist, but no metric provided.')

    # TODO: finish implementing the liquidation logic.
    while curr_greek_level >= desired_greek_level:
        op = relevant_ops.pop()

    pass


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
    cost = 0
    pdt = vol_id.split()[0]
    opmth = vol_id.split()[1].split('.')[0]
    ftmth = vol_id.split()[1].split('.')[1]
    ops = [x for x in pf.OTC_options if x.get_product() == pdt
           and x.get_month() == ftmth and x.get_op_month() == opmth and x.char == char]

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
        pf, cost_2 = close_out_deltas(pf, iden)
        cost += cost_2

    return pf, cost


def add_position(pf, signal, vdf, pdf, vol_id, char, date, lots, greek, greekval, slippage=None, brokerage=None):
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
    tobeadded = []
    shorted = True if signal < 0 else False

    # adding option.
    for i in range(abs(signal)):
        op = create_vanilla_option(vdf, pdf, vol_id, char, shorted,
                                   date, lots=lots, greek=greek, greekval=greekval)
        tobeadded.append(op)

    # computing cost.
    if slippage is not None:
        cost += slippage

    if brokerage is not None:
        cost += sum([op.lots for op in tobeadded]) * brokerage

    pf.add_security(tobeadded, 'OTC')
    return pf, cost
