import numpy as np
from scripts.util import create_composites
from scripts.calc import compute_strike_from_delta
from scripts.classes import Option


def hedge_delta_roll(fpf, pdf, brokerage=None, slippage=None):
    """Rolls delta of the option back to a value specified in hedge dictionary if op.delta 
    exceeds certain bounds.

    Args:
        pf (object): Portfolio being hedged
        roll_cond (list): list of the form ['roll', value, frequency, bound]
        vdf (pandas df): volatility data frame containing information for current day
        pdf (pandas df): price dataframe containing information for current day
        brokerage (int, optional): brokerage fees per lot
        slippage (int, optional): slippage loss

    Returns:
        tuple: updated portfolio and cost of purchasing/selling options.
    """
    # print('hedge_delta_roll conds: ', roll_cond)

    cost = 0

    # initializing dictionary mapping pf -> processed options in that pf.
    processed = {}

    for pf in fpf.get_families():
        # initial sanity checks.
        if pf not in processed:
            processed[pf] = []
        else:
            # TODO: figure out what happens with hedge options.
            # case: number of options processed = number of relevant options,
            # meaning all have been rolled.
            if len(processed[pf]) == len(pf.OTC_options):
                continue

        print(' --- handling rolling for family ' + str(pf.name) + ' ---')

        # grab list of processed options for this portfolio.
        processed_ops = processed[pf]
        hedges = pf.hedge_params

        # isolating roll conditions.
        roll_cond = [hedges['delta'][i] for i in range(len(hedges['delta'])) if hedges[
            'delta'][i][0] == 'roll']
        if roll_cond:
            # case: roll conditions found.
            roll_cond = roll_cond[0]
            roll_val, bounds = roll_cond[1], np.array(roll_cond[3]) / 100
        else:
            # case: no roll conditions found.
            print('no roll conditions found for family ' + str(pf.name))
            continue

        # starting of per-option rolling logic.
        for op in pf.OTC_options.copy():
            # case: option has already been processed due to its partner being
            # processed.
            if op in processed_ops:
                continue
            composites = []
            delta = abs(op.delta / op.lots)
            # case: delta not in bounds.
            diff = (delta - roll_val/100)
            # print('diff, bounds: ', diff, bounds)
            if (diff < bounds[0]) or (diff > bounds[1]):
                # if delta > bounds[1] or delta < bounds[0]:
                print('rolling delta: ', op.get_product(),
                      op.char, round(abs(op.delta / op.lots), 2))
                newop, old_op, rcost = delta_roll(pf, op, roll_val, pdf,
                                                  slippage=slippage, brokerage=brokerage)
                processed_ops.append(old_op)
                composites.append(newop)
                cost += rcost
                # if rolling option, roll all partners as well.
                for opx in op.partners:
                    print('rolling delta: ', opx.get_product(),
                          opx.char, round(abs(opx.delta / opx.lots), 2))
                    pf2 = fpf.get_family_containing(opx)
                    if pf2 is None:
                        raise ValueError(str(opx) + ' belongs to a \
                                            non-existent family.')

                    new_opx, old_opx, rcost = delta_roll(pf2, opx, roll_val, pdf,
                                                         slippage=slippage, brokerage=brokerage)
                    composites.append(new_opx)
                    if pf2 not in processed:
                        processed[pf2] = []
                    processed[pf2].append(old_opx)
                    cost += rcost
                    composites = create_composites(composites)
            else:
                processed_ops.append(op)
                print('op' + str(op) + ' is within bounds. skipping...')

    print(' --- finished rolling for family ' + str(pf.name) + ' ---')
    for x in processed:
        print('ops processed belonging to ' + x.name + ':')
        print([str(i) for i in processed[x]])

    fpf.refresh()
    return fpf, cost


def delta_roll(pf, op, roll_val, pdf, slippage=None, brokerage=None):
    """Helper function that deals with delta-rolling options.

    Args:
        op (TYPE): Description
        roll_val (TYPE): Description
        slippage (None, optional): Description
        brokerage (None, optional): Description

    Returns:
        TYPE: Description
    """
    # print('delta not in bounds: ', op, abs(op.delta) / op.lots)

    print('handling ' + str(op) + ' from family ' + pf.name)
    print('family rolling conds: ', pf.hedge_params['delta'])

    cost = 0
    cpi = 'C' if op.char == 'call' else 'P'
    # get the vol from the vol_by_delta part of pdf
    col = str(int(roll_val)) + 'd'
    try:
        vid = op.get_product() + '  ' + op.get_op_month() + '.' + op.get_month()
        vol = pdf.loc[(pdf.call_put_id == cpi) &
                      (pdf.vol_id == vid), col].values[0]
        # print('vol found')
    except IndexError:
        print('[ERROR] hedge_delta_roll: vol not found. Inputs: ', cpi, vid)
        vol = op.vol

    strike = compute_strike_from_delta(
        op, delta1=roll_val / 100, vol=vol)
    # print('roll_hedging - newop tau: ', op.tau)
    newop = Option(strike, op.tau, op.char, vol, op.underlying,
                   op.payoff, op.shorted, op.month, direc=op.direc,
                   barrier=op.barrier, lots=op.lots, bullet=op.bullet,
                   ki=op.ki, ko=op.ko, rebate=op.rebate,
                   ordering=op.ordering, settlement=op.settlement)

    # handle expenses: brokerage and old op price - new op price

    val = -(op.compute_price() - newop.compute_price())
    cost += val

    if brokerage:
        cost += (brokerage * (op.lots + newop.lots))

    if slippage:
        ttm = newop.tau * 365
        if ttm < 60:
            s_val = slippage[0]
        elif ttm >= 60 and ttm < 120:
            s_val = slippage[1]
        else:
            s_val = slippage[-1]
        cost += (s_val * (newop.lots + op.lots))

    # handle adds and removes.
    # print('simulation.hedge_delta_roll - removing ' +
    #       str(op) + ' from pf ' + str(pf.name))
    # print('simulation.hedge_delta_roll - adding ' +
    #       str(newop) + ' to pf ' + str(pf.name))

    pf.remove_security([op], 'OTC')

    pf.add_security([newop], 'OTC')

    return newop, op, cost
