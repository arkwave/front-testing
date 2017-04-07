from scripts.classes import Option, Future
import copy
import pandas as pd
from scripts.portfolio import Portfolio
from math import ceil
brokerage = 1


def rebalance(vdf, pdf, pf, hedges):
    """ Function that handles EOD greek hedging. Calls hedge_delta and hedge_gamma_vega.
    Notes:
    1) hedging gamma and vega done by buying/selling ATM straddles. No liquidity constraints assumed.
    2) hedging delta done by shorting/buying -delta * lots futures.
    3)

    Args:
        vdf (TYPE): Description
        pdf (TYPE): Description
        pf (TYPE): Description
        hedges (TYPE): Description

    Returns:
        TYPE: Description
    """
    # compute the gamma and vega of atm straddles; one call + one put.
    # compute how many such deals are required. add to appropriate pos.
    # return both the portfolio, as well as the gain/loss from short/long pos
    expenditure = 0
    # hedging delta, gamma, vega.
    dic = copy.deepcopy(pf.get_net_greeks())
    for product in dic:
        for month in dic[product]:
            ordering = pf.compute_ordering(product, month)
            ginputs = gen_hedge_inputs(
                hedges, vdf, pdf, month, pf, product, ordering, 'gamma')
            vinputs = gen_hedge_inputs(
                hedges, vdf, pdf, month, pf, product, ordering, 'vega')
            cost, pf = hedge(pf, ginputs, product, month)
            expenditure += cost
            cost, pf = hedge(pf, vinputs, product, month)
            expenditure += cost
            cost, pf = hedge_delta(hedges['delta'], vdf, pdf,
                                   month, pf, product, ordering)
            expenditure += cost
    return expenditure, pf


# TODO: update this with new objects in mind.
def gen_hedge_inputs(hedges, vdf, pdf, month, pf, product, ordering, flag):
    """Helper function that generates the inputs required to construct atm 
    straddles for hedging, based on the flag. 

    Args:
        hedges (TYPE): hedging rules.
        vdf (TYPE): volatility dataframe
        pdf (TYPE): price dataframe
        month (TYPE): month being hedged
        pf (TYPE): portfolio being hedged
        product (TYPE): product being hedged
        ordering (TYPE): ordering corresponding to month being hedged
        flag (TYPE): gamma or vega

    Returns:
        list : inputs required to construct atm straddles. 
    """
    net_greeks = pf.get_net_greeks()
    greeks = net_greeks[product][month]
    # naming variables for clarity.
    gamma = greeks[1]
    vega = greeks[3]
    greek = gamma if flag == 'gamma' else vega
    gamma_bound = hedges['gamma']
    vega_bound = hedges['vega']
    bound = gamma_bound if flag == 'gamma' else vega_bound

    # relevant data for constructing Option and Future objects.
    price = pdf[(pdf.pdt == product) & (
        pdf.order == ordering)].settle_value.values[0]
    k = round(price/10) * 10
    cvol = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'C') & (vdf.order == ordering) & (vdf.strike == k)].settle_vol.values[0]
    pvol = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'P') & (vdf.order == ordering) & (vdf.strike == k)].settle_vol.values[0]
    tau = vdf[(vdf.pdt == product) & (
        vdf.call_put_id == 'P') & (vdf.order == ordering) & (vdf.strike == k)].tau.values[0]
    underlying = Future(month, price, product)

    return [price, k, cvol, pvol, tau, underlying, greek, bound, ordering]


def hedge(pf, inputs, product, month, flag):
    """This function does the following:
    1) constructs atm straddles with the inputs from _inputs_
    2) hedges the greek in question (specified by flag) with the straddles.

    Args:
        pf (TYPE): portfolio object
        inputs (TYPE): list of inputs reqd to construct straddle objects
        product (TYPE): the product being hedged
        month (TYPE): month being hedged

    Returns:
        tuple: cost of the hedge, and the updated portfolio 
    """

    expenditure = 0
    price, k, cvol, pvol, tau, underlying, greek, bound, ordering = inputs

    # creating straddle components.
    callop = Option(price, tau, 'call', cvol, underlying,
                    'euro', ordering=ordering, shorted=None)
    putop = Option(price, tau, 'put', pvol, underlying,
                   'euro', ordering=ordering, shorted=None)
    straddle_val = callop.compute_price() + putop.compute_price()

    # gamma and vega hedging.
    cdelta, cgamma, ctheta, cvega = callop.greeks()
    pdelta, pgamma, ptheta, pvega = putop.greeks()

    if flag == 'gamma':
        pgreek, cgreek = pgamma, cgamma
    else:
        pgreek, cgreek = pvega, cvega

    # checking if gamma exceeds bounds
    if greek not in range(*bound):
        lower = bound[0]
        upper = bound[1]
        # gamma hedging logic.
        if greek < lower:
            # need to buy straddles. expenditure is positive.
            callop.shorted = False
            putop.shorted = False
            num_required = ceil((lower-greek)/(pgreek + cgreek))
        elif greek > upper:
            # need to short straddles. expenditure is negative.
            callop.shorted = True
            putop.shorted = True
            num_required = ceil((upper-greek)/(pgreek + cgreek))
        expenditure += num_required * (straddle_val + brokerage)
        for i in range(num_required):
            pf.add_security(callop, 'hedge')
            pf.add_security(putop, 'hedge')

    return expenditure, pf


# TODO: Note: assuming that futures are 10 lots each.
def hedge_delta(cond, vdf, pdf, month, pf, product, ordering):
    """Helper function that implements delta hedging. General idea is to zero out delta at the end of the day by buying/selling -delta * lots futures. Returns expenditure (which is negative if shorting and postive if purchasing delta) and the updated portfolio object.

    Args:
        cond (string): condition for delta hedging
        vdf (dataframe): Dataframe of volatilities
        pdf (dataframe): Dataframe of prices
        net (list): greeks associated with net_greeks[product][month]
        month (str): month of underlying future.
        pf (portfolio): portfolio object specified by portfolio_specs.txt

    Returns:
        tuple: hedging costs and final portfolio with hedges added.

    """
    future_price = pdf[(pdf.pdt == product) & (
        pdf.order == ordering)].settle_value.values[0]
    expenditure = 0
    net_greeks = pf.get_net_greeks()
    if cond == 'zero':
        # flag that indicates delta hedging.
        for product in net_greeks:
            for month in net_greeks[product]:
                vals = net_greeks[product][month]
                delta = vals[0]
                num_lots_needed = delta * 100
                num_futures = ceil(num_lots_needed / 10)
                shorted = True if delta > 0 else False
                ft = Future(month, future_price, product,
                            shorted=shorted, ordering=ordering)
                for i in range(num_futures):
                    pf.add_security(ft, 'hedge')
                cost = num_futures * (future_price + brokerage)
                expenditure = (expenditure - cost) if shorted else (
                    expenditure + cost)
    return expenditure, pf


def generate_portfolio(flag):
    """Generate portfolio for testing purposes. """
    # Underlying Futures
    ft1 = Future('K7', 300, 'C')
    ft2 = Future('K7', 250, 'C')
    ft3 = Future('N7', 320, 'C')
    ft4 = Future('N7', 330, 'C')
    ft5 = Future('N7', 240, 'C')

    short = False if flag == 'long' else True
    # options

    op1 = Option(
        350, 0.301369863013698, 'call', 0.4245569263291844, ft1, 'amer', short, 'K7', ordering=1)

    op2 = Option(
        290, 0.301369863013698, 'call', 0.45176132048500206, ft2, 'amer', short, 'K7', ordering=1)

    op3 = Option(300, 0.473972602739726, 'call', 0.14464169782291536,
                 ft3, 'amer', short, 'N7',  direc='up', barrier='amer', bullet=False,
                 ko=350, ordering=2)

    op4 = Option(330, 0.473972602739726, 'put', 0.18282926924909026,
                 ft4, 'amer', short, 'N7', direc='down', barrier='amer', bullet=False,
                 ki=280, ordering=2)
    op5 = Option(
        320, 0.473972602739726, 'put', 0.8281728247909962, ft5, 'amer', short, 'N7', ordering=2)

    # Portfolio Futures
    # ft6 = Future('K7', 370, 'C', shorted=False, ordering=1)
    # ft7 = Future('N7', 290, 'C', shorted=False, ordering=2)
    # ft8 = Future('Z7', 320, 'C', shorted=True, ordering=4)
    # ft9 = Future('Z7', 320, 'C', shorted=True, ordering=4)

    OTCs, hedges = [op1, op2, op3], [op4, op5]

    # creating portfolio
    pf = Portfolio()
    for sec in hedges:
        pf.add_security(sec, 'OTC')

    for sec in OTCs:
        pf.add_security(sec, 'OTC')

    return pf


pf1 = generate_portfolio('long')
g1 = pf1.net_greeks['C']['K7']

pf2 = generate_portfolio('short')
g2 = pf2.net_greeks['C']['K7']
