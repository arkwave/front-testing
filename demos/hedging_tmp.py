import copy
from simulation import gen_hedge_inputs, hedge, hedges_satisfied, check_roll_status, hedge_delta_roll, hedge_delta
from scripts.hedge import Hedge


def rebalance(vdf, pdf, pf, hedges, counters, desc, buckets=None, brokerage=None, slippage=None, hedge_type='straddle'):
    """Function that handles EOD greek hedging. Calls hedge_delta and hedge_gamma_vega.

    Notes:
        1) hedging gamma and vega done by buying/selling ATM straddles. No liquidity constraints assumed.
        2) hedging delta done by shorting/buying -delta * lots futures.
        3)

    Args:
        vdf (pandas dataframe): Dataframe of volatilities
        pdf (pandas dataframe): Dataframe of prices
        pf (object): portfolio object
        hedges (dict): Dictionary of hedging conditions
        counters (TYPE): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description

    Returns:
        tuple: portfolio, counters
    """
    # compute the gamma and vega of atm straddles; one call + one put.
    # compute how many such deals are required. add to appropriate pos.
    # return both the portfolio, as well as the gain/loss from short/long pos
    # hedging delta, gamma, vega.
    delta_freq, gamma_freq, theta_freq, vega_freq = counters
    hedgearr = [False, False, False, False]
    droll = None
    # updating counters
    if pf.empty():
        return pf, counters, 0, False

    for greek in hedges:
        if greek == 'delta':
            if delta_freq == hedges[greek][0][2]:
                counters[0] = 1
                hedgearr[0] = True
            else:
                print('delta freq not met.')
                counters[0] += 1
        elif greek == 'gamma':
            if gamma_freq == hedges[greek][0][2]:
                counters[1] = 1
                hedgearr[1] = True
            else:
                print('gamma freq not met')
                counters[1] += 1
        elif greek == 'vega':
            if vega_freq == hedges[greek][0][2]:
                counters[3] = 1
                hedgearr[3] = True
            else:
                print('vega freq not met')
                counters[3] += 1
        elif greek == 'theta':
            if theta_freq == hedges[greek][0][2]:
                counters[2] = 1
                hedgearr[2] = True
            else:
                print('gamma freq not met')
                counters[2] += 1

    roll_hedged = check_roll_status(pf, hedges)
    droll = not roll_hedged

    cost = 0

    # first: handle roll-hedging.
    if roll_hedged:
        print('deltas within bounds. skipping roll_hedging')

    if not roll_hedged:
        print(' ++ ROLL HEDGING REQUIRED ++ ')
        for op in pf.OTC_options:
            print('delta: ', abs(op.delta / op.lots))
        roll_cond = [hedges['delta'][i] for i in range(len(hedges['delta'])) if hedges[
            'delta'][i][0] == 'roll'][0]
        pf, exp = hedge_delta_roll(
            pf, roll_cond, pdf, brokerage=brokerage, slippage=slippage)
        cost += exp

    hedge_count = 0

    # initialize hedge engine.
    hedge_engine = Hedge(pf, hedges, pdf, desc,
                         buckets=buckets, kind=hedge_type)

    # calibrate hedge object to all non-delta hedges.
    for flag in hedges:
        if flag != 'delta':
            hedge_engine._calibrate(flag)

    done_hedging = hedge_engine.satisfied(pf)

    # hedging non-delta greeks.
    while (not done_hedging and hedge_count < 10):
        # insert the actual business of hedging here.
        for flag in hedges:
            if flag == 'gamma' and hedgearr[1]:
                cost += hedge_engine.apply(pf, 'gamma', 'bound')
            elif flag == 'vega' and hedgearr[3]:
                cost += hedge_engine.apply(pf, 'vega', 'bound')
            elif flag == 'theta' and hedgearr[2]:
                cost += hedge_engine.apply(pf, 'theta', 'bound')
            hedge.refresh()

        hedge_count += 1
        done_hedging = hedge_engine.satisfied(pf)

    # check if delta hedging is required. if so, perform. else, skip.
    if hedgearr[0] and 'delta' in hedges:
        # grabbing condition that indicates zeroing condition on
        # delta
        hedge_type = hedges['delta']
        hedge_type = [hedge_type[i] for i in range(len(hedge_type)) if hedge_type[
            i][0] == 'static'][0][1]

        cost += hedge_engine.apply(pf, 'delta', hedge_type)

    else:
        print('no delta hedging specifications found')

    return (pf, counters, cost, droll)
