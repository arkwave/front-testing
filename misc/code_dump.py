

def apply_signal(pf, vdf, pdf, signals, date, next_date, roll_cond, strat='dist', tol=3000):
    """Applies the signal generated by the recommendation program to the portfolio.
    Args:
        pf (object): portfolio
        vdf (pandas dataframe): dataframe of volatilities
        pdf (pandas dataframe): dataframe of prices
        signals (pandas dataframe): signals
        date (pandas Timestamp): current date
        next_date (pandas Timestamp): next date in simulation
        roll_cond (list): list of delta_roll conditions
        strat (str, optional): Description

    Returns:
        portfolio object: the portfolio with the requisite changes applied.
    """
    # identifying relevant columns; hardcoded due to formatting, change if
    # necessary.
    cols = ['delta_call_25_a', 'delta_put_25_b',
            'signal', 'opmth', 'ftmth', 'pdt', 'lots', 'vega']
    # getting inputs from signals dataframe
    print('next_date: ', next_date)

    if next_date is None:
        print('reached end of signal period')
        return pf

    cvol, pvol, sig, opmth, ftmth, pdt, lots, vega_req = signals.loc[
        signals.value_date == next_date, cols].values[0]

    print('________APPLYING SIGNAL_______: ', sig)

    ret = None
    next_date = pd.to_datetime(next_date)

    # Case 1: flatten signal
    if sig == 0:
        ret = Portfolio()

    else:
        dval = roll_cond[1]/100

        # determining if options are to be shorted or not
        shorted = True if sig < 0 else False
        cvol, pvol = cvol/100, pvol/100
        # computing ordering
        curr_mth = date.month
        curr_mth_sym = month_to_sym[curr_mth]
        curr_yr = date.year % (2000 + decade)
        curr_sym = curr_mth_sym + str(curr_yr)
        order = find_cdist(curr_sym, ftmth, contract_mths[pdt])

        # create the underlying future
        uid = pdt + '  ' + ftmth
        try:
            ftprice = pdf[(pdf.value_date == date) &
                          (pdf.underlying_id == uid)].settle_value.values[0]
        except IndexError:
            print('inputs: ', date, uid)

        ft = Future(ftmth, ftprice, pdt, shorted=False,
                    lots=lots, ordering=order)

        # create the options; long one dval call, short on dval put
        vol_id = pdt + '  ' + opmth + '.' + ftmth

        # computing tau
        tau = vdf[(vdf.value_date == date) &
                  (vdf.vol_id == vol_id)].tau.values[0]

        # computing strikes
        c_strike = compute_strike_from_delta(
            None, delta1=dval, vol=cvol, s=ftprice, tau=tau, char='call', pdt=pdt)
        p_strike = compute_strike_from_delta(
            None, delta1=dval, vol=pvol, s=ftprice, tau=tau, char='put', pdt=pdt)

        # creating placeholder options objects for computation purposes
        callop = Option(c_strike, tau, 'call', cvol, ft, 'amer',
                        shorted, opmth, lots=lots, ordering=order)
        putop = Option(p_strike, tau, 'put', pvol, ft, 'amer',
                       not shorted, opmth, lots=lots, ordering=order)

        # calculate lots required for requisite vega specified; done according
        # to callop.
        pnl_mult = multipliers[pdt][-1]

        call_vega = (callop.vega * 100) / (callop.lots * pnl_mult)
        print('call vega: ', call_vega)

        # lots_req = round(
        #     abs(((vega_req * num_skews * 100) / call_vega * pnl_mult)))
        lots_req = round((abs(vega_req * num_skews) * 100) /
                         abs(call_vega * pnl_mult))

        # print('lots req: ', lots_req)
        # initializing useful variables
        long_calls = [op for op in pf.OTC_options if op.char ==
                      'call' and op.shorted == False]
        short_puts = [op for op in pf.OTC_options if op.char ==
                      'put' and op.shorted == True]
        short_calls = [op for op in pf.OTC_options if op.char ==
                       'call' and op.shorted == True]
        long_puts = [op for op in pf.OTC_options if op.char ==
                     'put' and op.shorted == False]

        if (not pf.empty() and (sig < 0) and (long_calls) and (short_puts)):
            # Case 2-1: close off existing long skew pos selected according to strat
            # when getting sell sig
            print('sig < 0; liquidating long skew pos')
            calls = long_calls
            puts = short_puts
            ret = liquidate_skew_pos(vega_req, num_skews, dval,
                                     calls, puts, pf, strat)

        elif (not pf.empty() and (sig > 0) and (short_calls) and (long_puts)):
            # Case 2-2: close off existing short skew pos selected according to strat
            # when getting buy sig.
            print('sig > 0; liquidating short skew pos')
            calls = short_calls
            puts = long_puts
            ret = liquidate_skew_pos(vega_req, num_skews, dval,
                                     calls, puts, pf, strat)

        # Other 4 cases: buy/sell with empty portfolio, or buy with no
        # shorts/sell with no longs
        else:
            print('non-liquidation signal')
            print('lots: ', lots_req)
            tobeadded = []
            for i in range(num_skews):
                callop = Option(c_strike, tau, 'call', cvol, ft, 'amer',
                                shorted, opmth, lots=lots_req, ordering=order)
                putop = Option(p_strike, tau, 'put', pvol, ft, 'amer',
                               not shorted, opmth, lots=lots_req, ordering=order)
                print('vegas: ', callop.vega, putop.vega)
                pf.add_security([callop, putop], 'OTC')
                tobeadded.extend([callop, putop])

            for op in tobeadded:
                print('added op deltas: ', op, abs(op.delta/op.lots))
            tobeadded.clear()

        # one final sanity check: if option lists are empty and there are hedge futures, close them
        # all out.

        ret = close_out_deltas(pf, ftprice)

    print('_________ SIGNAL APPLIED __________')

    return ret


# FIXME; figure out if this needs to handle multiple products.
def liquidate_pos(char, resid_vega, ops, pf, strat, dval):
    """Buys/Sells (vega_req * num_req) worth of dval skew composites. For example if vega = 10000, num_skews =1 and dval = 25, then the function figures out how to liquidate 10,000 vega worth of that day's 25 Delta skew positions ON EACH LEG (i.e. a Long 25 Delta Call and a Short 25 Delta Put) from the currently held position.

Does so by selecting  a skew position to liquidate based on strat;
currently implemented strategies for liquidation are 'dist' (the options
selected are furthest away from dval) and 'filo' (first in last out),
computing the vega per lot of that skew position, and liquidating the
required number of lots. If the number of lots contained in the selected
skew position is smaller than the number of lots required, the skew
position is completely closed out (i.e. shorts are bought back, longs
are sold) and the leftover vega is liquidated using the next highest
skew pos, selected according to strat.

If all options of a given type have been spent in this manner and
liquidation is still not complete, the algorithm simply skips to handle
options of the next type. In the pathological case where the entire
portfolio's contents are insufficient to meet the liquidation condition,
the entire portfolio is closed, and remaining deltas from hedging are
closed off.

In practice, since we're interested in the call-side (long skew), lots
required for the requisite vega is determined by the call option. Put
option is instantiated with exactly the same number of lots, which could
lead to a slight vega differential (currently being ignored, but might
need to be addressed in the future).


    Notes:
        - for readability: vpl = vega per lot.
    Args:
        vega_req (TYPE): total amount of vega to be liquidated.
        num_skews (TYPE): number of buy/sell signals on this day. total vega to be liquidated would be vega_req * num_skews
        dval (TYPE): delta value of the skew pos we want to run.
        calls (TYPE): list of call options selected according to which liquidation scenario is in effect (i.e. liquidating shorts with a buy signal, or liquidating longs with a sell signal)
        puts (TYPE): list of put options selected according to which liquidation scenario is in effect.
        pf (TYPE): portfolio object being subjected to the signal.
strat (TYPE): the regime used to determine which skew positions are
liquidated first.

    Returns:
        portfolio: the updated portfolio with the appropriate equivalent position liquidated.
    """

    toberemoved = []

    # handling puts
    print('HANDLING ' + char.upper())
    while resid_vega > 0:
        print('residual vega: ', resid_vega)
        if strat == 'dist':
            print('selecting skew acc to dist')
            print('portfolio at loop start: ', pf)
            put_ops = sorted(ops, key=lambda x: abs(
                abs(x.delta/x.lots) - dval))

            max_put_op = put_ops[-1] if put_ops else None

            # print('putops: ', [str(p) for p in put_ops])
        elif strat == 'filo':
            print('selecting skew acc to filo')
            print('portfolio at loop start: ', pf)
            max_op = ops[0] if ops else None

        if max_op is not None:

            print('op selected: ', max_op)
            # handle puts
            vpl = abs(max_op.vega/max_op.lots)
            print('puts vega per lot: ', vpl)
            put_lots_req = round(abs(resid_vega / vpl))
            print('put lots req: ', put_lots_req)

            # Case 1: lots required < lots available.
            if put_lots_req < max_put_op.lots:
                print('l_puts: lots available.')
                resid_vega = 0
                newlots = max_put_op.lots - put_lots_req
                max_put_op.update_lots(newlots)
                print('puts - new lots: ', newlots)
                # break

            # Case 2: lots required > lots available.
            else:
                print('l_puts: lots unavailable.')
                resid_vega -= max_op.lots * vpl
                toberemoved.append(max_op)
                ops.remove(max_op)
        else:
            print(
                'cannot liquidate existing positions any further. continuing to handle calls..')
            break

    # update any lot size changes
    pf.update_sec_by_month(False, 'OTC', update=True)

    # remove securities flagged for removal
    pf.remove_security(toberemoved, 'OTC')

    # debug statement
    for op in toberemoved:
        print('op removed deltas: ', op, abs(op.delta/op.lots))
    toberemoved.clear()
    print('pf after liquidation: ', pf)
    return pf
