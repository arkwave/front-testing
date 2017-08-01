# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-08-01 18:18:23
# @Last Modified by:   Ananth
# @Last Modified time: 2017-08-01 22:08:26


def apply_signal(pf, vdf, pdf, signals, date, next_date, roll_cond, strat='dist', tol=1000, brokerage=None, slippage=None):
    """Wrapper method that delegates application of the signal to the relevant sub-function.

    Args:
        pf (TYPE): Description
        vdf (TYPE): Description
        pdf (TYPE): Description
        signals (TYPE): Description
        date (TYPE): Description
        next_date (TYPE): Description
        roll_cond (TYPE): Description
        strat (str, optional): Description
        tol (int, optional): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description
    """

    # check sigtype
    sigtype = signals.sigtype.unique()[0]
    if sigtype == 'skew':
        return apply_skew_signal(pf, vdf, pdf, signals, date, next_date, roll_cond, strat=strat, , tol=tol, brokerage=brokerage, slippage=slippage)

    elif sigtype == 'straddle':
        return apply_straddle_signal(pf, vdf, pdf, signals, date, next_date, roll_cond, strat=strat, tol=tol, brokerage=brokerage, slippage=slippage)


def apply_straddle_signal(pf, vdf, pdf, signals, date, next_date, roll_cond, strat='dist', tol=1000, brokerage=None, slippage=None):
    """Helper method that deals with straddle signals. 

    Args:
        pf (TYPE): Description
        vdf (TYPE): Description
        pdf (TYPE): Description
        signals (TYPE): Description
        date (TYPE): Description
        next_date (TYPE): Description
        roll_cond (TYPE): Description
        strat (str, optional): Description
        tol (int, optional): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description
    """
    cost = 0
    cols = ['call_vol', 'put_vol',
            'signal', 'opmth', 'ftmth', 'pdt', 'vega', 'strike']
    # getting inputs from signals dataframe
    print('next_date: ', next_date)

    if next_date is None:
        print('reached end of signal period')
        return pf, 0

    relevant_signals = signals[signals.value_date == next_date][cols]
    for _, row in relevant_signals.iterrows():
        cvol, pvol, sig, opmth, ftmth, pdt, vega_req, strike = row.values
        inputs = [cvol, pvol, sig, opmth, ftmth, pdt, vega_req, strike]
        print('________APPLYING SIGNAL_______: ', sig)
        print('Inputs: ', inputs)
        ret = None
        next_date = pd.to_datetime(next_date)

        # Case 1: flatten signal
        if sig == 0:
            ret, cost = Portfolio(), 0

        # Case 2: Nonzero signal.
        else:
            handle_longs, handle_shorts = True, True
            # Case 2-1: Adding to empty portfolio.
            if len(pf.OTC_options) == 0:
                vol_id = pdt + '  ' + opmth + '.' + ftmth
                greekval = vega_req * abs(signal)
                shorted = True if signal < 0 else False
                ops = create_straddle(
                    vol_id, vdf, pdf, date, shorted, strike, greek='vega', greekval=greekval)
                pf.add_security(list(ops), 'OTC')
                if brokerage is not None:
                    cost += brokerage * sum([op.lots for op in ops])
                if slippage is not None:
                    cost += slippage

            # Case 2-2: Adding to nonempty Portfolio
            # determine if liquidation or extending current position.
            long_calls = [x for x in pf.OTC_options if x.char ==
                          'call' and not x.shorted and x.get_product() == pdt]
            long_puts = [x for x in pf.OTC_options if x.char ==
                         'put' and not x.shorted and x.get_product() == pdt]
            short_calls = [x for x in pf.OTC_options if x.char ==
                           'call' and not x.shorted and x.get_product() == pdt]
            short_calls = [x for x in pf.OTC_options if x.char ==
                           'put' and x.shorted and x.get_product() == pdt]

            # case: current position is long some straddles
            if len(long_calls) == len(long_puts) and len(long_puts) > 0:
                curr_vega = sum([op.vega for op in long_calls]) + \
                    sum([op.vega for op in long_puts])
                if abs(curr_vega - abs(vega_reqd*signal)) < tol:
                    print('vega from long straddles within tol. skipping handling...')
                    handle_longs = False
            # case: current position is short some straddles
            elif len(short_puts) == len(short_calls) and len(short_puts) > 0:
                curr_vega = sum([op.vega for op in short_calls]) + \
                    sum([op.vega for op in short_puts])
                if abs(curr_vega - abs(vega_reqd*signal)) < tol:
                    print('vega from short straddles within tol. skipping handling...')
                    handle_shorts = False

            if handle_longs:
                pass

            if handle_shorts:
                pass


def apply_skew_signal(pf, vdf, pdf, signals, date, next_date, roll_cond, strat='dist', tol=1000, brokerage=None, slippage=None):
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
        tol (int, optional): Description
        brokerage (None, optional): Description
        slippage (None, optional): Description

    Returns:
        portfolio object: the portfolio with the requisite changes applied.
    """

    # identifying relevant columns; hardcoded due to formatting, change if
    # necessary.

    cost = 0
    cols = ['call_vol', 'put_vol',
            'signal', 'opmth', 'ftmth', 'pdt', 'vega']
    # getting inputs from signals dataframe
    print('next_date: ', next_date)

    if next_date is None:
        print('reached end of signal period')
        return pf, 0

    relevant_signals = signals[signals.value_date == next_date][cols]
    for _, row in relevant_signals.iterrows():
        cvol, pvol, sig, opmth, ftmth, pdt, vega_req = row.values
        inputs = [cvol, pvol, sig, opmth, ftmth, pdt, vega_req]
        print('________APPLYING SIGNAL_______: ', sig)
        print('Inputs: ', inputs)
        ret = None
        next_date = pd.to_datetime(next_date)

        # Case 1: flatten signal
        if sig == 0:
            ret, cost = Portfolio(), 0
        # Case 2: Nonzero signal
        else:
            # grab delta value we want each leg of the skew to have.
            dval = roll_cond[1] / 100
            # identify the net vega position on each leg of the skew.
            net_call_vega, net_put_vega = pf.net_vega_pos(ftmth)
            print('net call/put vega for ' + str(ftmth) +
                  ': ', net_call_vega, net_put_vega)
            # get the target vega position we want on each leg
            target_call_vega, target_put_vega = vega_req * sig,  -vega_req * sig
            print('target call/put vega: ', target_call_vega, target_put_vega)
            # boolean placeholders.
            handle_calls, handle_puts = True, True
            # Case 2-1: Adding to empty portfolio.
            if net_call_vega == 0 and net_put_vega == 0:
                print('empty portfolio; adding skews')
                inputs[-1]
                pf, cost = add_skew(pf, vdf, pdf, inputs, date,
                                    dval * 100, brokerage=brokerage, slippage=slippage)
                ret, cost = pf, cost
            # Case 2-2: Adding to nonempty portfolio
            else:
                # check if call vegas are within bounds
                if abs(net_call_vega - target_call_vega) < tol:
                    print('call vega within bounds. Skipping handling.')
                    handle_calls = False
                # check if put vegas are within bounds
                if abs(net_put_vega - target_put_vega) < tol:
                    print('put vega within bounds. Skipping handling.')
                    handle_puts = False

                if handle_calls:
                    calls = [op for op in pf.OTC_options if op.char == 'call']
                    # update_skew_pos(char, target_vega, curr_vega, dval, ops, pf,
                    # strat, tol, vdf, pdf, inputs, date)
                    pf, cost = update_skew_pos('call', target_call_vega, net_call_vega,
                                               dval, calls, pf, strat, tol, vdf, pdf,
                                               inputs, date, brokerage=brokerage, slippage=slippage)

                if handle_puts:
                    puts = [op for op in pf.OTC_options if op.char == 'put']
                    pf, cost = update_skew_pos('put', target_put_vega, net_put_vega,
                                               dval, puts, pf, strat, tol, vdf, pdf,
                                               inputs, date, brokerage=brokerage, slippage=slippage)

            ret, cost = pf, cost

    print('________SIGNAL APPLIED _________')
    return ret, cost


def generate_skew_op(char, vdf, pdf, inputs, date, dval, brokerage=None, slippage=None):
    """Helper function that generates the options comprising the skew position, based on inputs passed in.
    Used when dealing with individual legs of the skew position, not when dealing with adding skews as a whole.

    Args:
        char (str): 'call' or 'put'.
        vdf (pd dataframe): dataframe of vols
        pdf (pd dataframe): dataframe of prices
        inputs (list):
        date (pd Timestamp): Description
        dval (double): Description
    """
    # unpack inputs
    cost = 0
    #
    vol, sig, opmth, ftmth, pdt, vega_req = inputs
    print('generate_skew_ops inputs: ')
    print('vol: ', vol)
    print('vega_reqd: ', vega_req)
    print('sig: ', sig)
    # determining if options are to be shorted or not
    if sig < 0 and char == 'call':
        shorted = True
    elif sig < 0 and char == 'put':
        shorted = False
    elif sig > 0 and char == 'call':
        shorted = False
    elif sig > 0 and char == 'put':
        shorted = True
    print('generate_skew_op - char: ', char)
    print('generate_skew_op - shorted: ', shorted)

    vol_id = pdt + '  ' + opmth + '.' + ftmth
    op = create_vanilla_option(vdf, pdf, vol_id, char, shorted, date, delta=dval * 100, vol=vol,
                               kwargs={'greek': 'vega', 'greekval': vega_req})

    lots_req = op.lots

    if brokerage:
        cost += brokerage * lots_req

    if slippage:
        ttm = op.tau * 365
        if ttm < 60:
            s_val = slippage[0]
        elif ttm >= 60 and ttm < 120:
            s_val = slippage[1]
        else:
            s_val = slippage[-1]
        cost += s_val * lots_req

    return op, cost


def add_skew(pf, vdf, pdf, inputs, date, dval, brokerage=None, slippage=None):
    """Helper method that adds a skew position based on the inputs provided.

    Args:
        pf (TYPE): portfolio being hedged.
        vdf (TYPE): vol dataframe with info pertaining to date.
        pdf (TYPE): price dataframe with info pertaining to date.
        inputs (TYPE): list corresponding to values in signals.loc[date = next_date]
        date (TYPE): current date in the simulation
        dval (TYPE): the delta value of the skew desired.
    """

    # unpack inputs
    cost = 0
    cvol, pvol, sig, opmth, ftmth, pdt, vega_req = inputs
    print('add_skew inputs: ')
    print('cvol: ', cvol)
    print('pvol: ', pvol)
    print('sig: ', sig)
    print('vega req: ', vega_req)
    print('dval: ', dval)

    # determining if options are to be shorted or not
    shorted = True if sig < 0 else False
    # num_skews = abs(sig)
    cvol, pvol = cvol / 100, pvol / 100
    # order = find_cdist(curr_sym, ftmth, contract_mths[pdt])

    # create the underlying future
    uid = pdt + '  ' + ftmth

    try:
        ftprice = pdf[(pdf.value_date == date) &
                      (pdf.underlying_id == uid)].settle_value.values[0]
    except IndexError:
        print('inputs: ', date, uid)

    # create the options; long one dval call, short on dval put
    vol_id = pdt + '  ' + opmth + '.' + ftmth
    # kwargs = {'greek': 'vega', 'greekval': vega_req}
    callop, putop = create_skew(
        vol_id, vdf, pdf, date, shorted, dval, ftprice, greek='vega', greekval=vega_req * abs(sig))

    print('callop: ', str(callop))
    print('putop: ', str(putop))

    tobeadded = []
    lots_req = callop.lots

    print('vegas: ', callop.vega, putop.vega)
    pf.add_security([callop, putop], 'OTC')
    tobeadded.extend([callop, putop])

    if brokerage:
        cost += brokerage * lots_req * 2

    if slippage:
        ttm = callop.tau * 365
        if ttm < 60:
            s_val = slippage[0]
        elif ttm >= 60 and ttm < 120:
            s_val = slippage[1]
        else:
            s_val = slippage[-1]
        cost += s_val * lots_req * 2

    # debug statement.
    for op in tobeadded:
        print('added op deltas: ', op, abs(op.delta / op.lots))
    tobeadded.clear()

    return pf, cost


def update_skew_pos(char, target_vega, curr_vega, dval, ops, pf, strat, tol, vdf, pdf, inputs, date, brokerage=None, slippage=None):
    """
    Helper function that updates individual legs of the skew position in question. Two major cases are handled:
    1) Increasing current position. I.e. negative to negative, positive to positive, negative to positive, positive to negative.

    2) Liquidating positions. Negative to larger negative (i.e. -30,000 -> -10,000), positive to smaller positive (30,000 -> 10,000)

    Notes:
        - for readability: vpl = vega per lot.

    Args:
        char (TYPE): leg of the skew being handled.
        target_vega (TYPE): target vega for this leg.
        curr_vega (TYPE): current vega of this leg.
        dval (TYPE): delta value required.
        ops (TYPE): relevant options; when handling call leg, all call options passed in. vice versa for puts.
        pf (TYPE): portfolio object being subjected to the signal
        strat (TYPE): the regime used to determine which skew positions are liquidated first.
        tol (TYPE): tolerance value within which portfolio is left alone.
        vdf (TYPE): dataframe of volatilities
        pdf (TYPE): dataframe of prices
        inputs (TYPE): row corresponding to signal.loc[index]
        date (TYPE): current date in the simulation.

    Returns:
        portfolio: the updated portfolio with the appropriate equivalent position liquidated.

    """
    cost = 0
    cvol, pvol, sig, opmth, ftmth, pdt, vega = inputs

    vol = cvol if char == 'call' else pvol

    print('HANDLING ' + char.upper() + ' LEG')
    print('target: ', target_vega)
    print('current: ', curr_vega)

    toberemoved, tobeadded = [], []
    # shorted = -1 if curr_vega > target_vega else 1
    buy = True if target_vega > curr_vega else False
    vega_req = abs(target_vega - curr_vega)

    # Case 1: need to buy vega.
    if buy:
        # Case 1-1: negative to positive pos. (e.g. -30,000 -> 10,000)
        if curr_vega < 0 and target_vega > 0:
            print(char.upper() + ': flipping leg from neg to pos')
            # reset leg to 0, then add required vega.
            # remove all short options on this leg.
            shorts = [op for op in ops if op.shorted]
            pf.remove_security(shorts.copy(), 'OTC')
            index = 0 if char == 'call' else 1
            # get residual vega on this leg after removing all shorts.
            curr_vega = pf.net_vega_pos(ftmth)[index]
            vega_req = target_vega - curr_vega
            # create input list
            op_inputs = [vol, sig, opmth, ftmth, pdt, vega_req]
            # generate option
            op, cost = generate_skew_op(
                char, vdf, pdf, op_inputs, date, dval, brokerage=brokerage, slippage=slippage)
            tobeadded.append(op)
            # total_cost += cost
            # case 1-2: nonnegative to positive pos (e.g. 10,000 -> 20,000)
        elif curr_vega > 0 and target_vega > 0:
            print(char.upper() + ' - increasing long pos')
            vega_req = target_vega - curr_vega
            op_inputs = [vol, sig, opmth, ftmth, pdt, vega_req]
            op, cost = generate_skew_op(
                char, vdf, pdf, op_inputs, date, dval, brokerage=brokerage, slippage=slippage)
            tobeadded.append(op)
        elif curr_vega < 0 and target_vega < 0:
            print('liquidating short positions - buying ' + char + ' leg')
            shortops = [op for op in ops if op.shorted]
            resid_vega = abs(curr_vega - target_vega)
            # print('resid_vega: ', resid_vega)
            pf, cost = liquidate_skew_pos(
                char, resid_vega, shortops, pf, strat, dval, brokerage=brokerage, slippage=slippage)

    # Case 2: Need to sell vega from this leg.
    else:
        # negative to negative; add 25 delta shorts
        if curr_vega < 0 and target_vega < 0:
            print(char.upper() + ' - increasing short pos')
            vega_req = target_vega - curr_vega
            op_inputs = op_inputs = [vol, sig, opmth, ftmth, pdt, vega_req]
            op, cost = generate_skew_op(
                char, vdf, pdf, op_inputs, date, dval, brokerage=brokerage, slippage=slippage)
            tobeadded.append(op)
        # positive to negative - same as negative to positive.
        elif curr_vega > 0 and target_vega < 0:
            print(char.upper() + ': flipping leg from pos to neg')
            # remove all long options on this leg, find difference, then add
            # requisite vega.
            longs = [op for op in ops if not op.shorted]
            pf.remove_security(longs.copy(), 'OTC')
            index = 0 if char == 'call' else 1
            curr_vega = pf.net_vega_pos(ftmth)[index]
            vega_req = target_vega - curr_vega
            op_inputs = [vol, sig, opmth, ftmth, pdt, vega_req]
            # generate option
            op, cost = generate_skew_op(
                char, vdf, pdf, op_inputs, date, dval, brokerage=brokerage, slippage=slippage)
            tobeadded.append(op)

        elif curr_vega > 0 and target_vega > 0:
            print('liquidating long positions - selling ' + char + ' leg')
            longops = [op for op in ops if not op.shorted]
            resid_vega = curr_vega - target_vega
            pf, cost = liquidate_skew_pos(
                char, resid_vega, longops, pf, strat, dval, brokerage=brokerage, slippage=slippage)

    # add any securities that need adding.
    pf.add_security(tobeadded, 'OTC')

    # update any lot size changes
    pf.update_sec_by_month(False, 'OTC', update=True)

    # debug statement
    for op in toberemoved:
        print('op removed deltas: ', op, abs(op.delta / op.lots))
    toberemoved.clear()
    # print('pf afte: ', pf)
    return pf, cost


def liquidate_skew_pos(char, resid_vega, ops, pf, strat, dval, brokerage=None, slippage=None):
    """Buys/Sells (vega_req * num_req) worth of dval skew composites. For example
       if vega = 10000, num_skews =1 and dval = 25, then the function figures out
       how to liquidate 10,000 vega worth of that day's 25 Delta skew positions ON
       EACH LEG (i.e. a Long 25 Delta Call and a Short 25 Delta Put) from the
       currently held position.

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
        strat (TYPE): the regime used to determine which skew positions are liquidated first.

    Returns:
        portfolio: the updated portfolio with the appropriate equivalent position liquidated.
    """

    toberemoved = []
    cost = 0
    # handling puts
    print('HANDLING ' + char.upper())
    # print('resid_vega: ', resid_vega)

    while resid_vega > 0:
        # print('residual vega: ', resid_vega)
        if strat == 'dist':
            print('selecting skew acc to dist')
            print('portfolio at loop start: ', pf)
            ops = sorted(ops, key=lambda x: abs(
                abs(x.delta / x.lots) - dval))
            max_op = ops[-1] if ops else None

            # print('putops: ', [str(p) for p in put_ops])
        elif strat == 'filo':
            print('selecting skew acc to filo')
            print('portfolio at loop start: ', pf)
            max_op = ops[0] if ops else None
            # print('op selected: ', max_op)

        if max_op is not None:
            print('op selected: ', max_op)
            # handle puts
            vpl = abs(max_op.vega / max_op.lots)
            print('puts vega per lot: ', vpl)
            lots_req = round(abs(resid_vega / vpl))
            print('put lots req: ', lots_req)

            # Case 1: lots required < lots available.
            if lots_req < max_op.lots:
                print('l_puts: lots available.')
                resid_vega = 0
                newlots = max_op.lots - lots_req
                max_op.update_lots(newlots)
                print('puts - new lots: ', newlots)
                if brokerage:
                    cost += brokerage * lots_req
                if slippage:
                    ttm = max_op.tau * 365
                    if ttm < 60:
                        s_val = slippage[0]
                    elif ttm >= 60 and ttm < 120:
                        s_val = slippage[1]
                    else:
                        s_val = slippage[-1]
                    cost += s_val * lots_req
                    # cost += slippage * lots_req
                # break

            # Case 2: lots required > lots available.
            else:
                print('l_puts: lots unavailable.')
                resid_vega -= max_op.lots * vpl
                toberemoved.append(max_op)
                ops.remove(max_op)
                if brokerage:
                    cost += brokerage * max_op.lots
                if slippage:
                    ttm = max_op.tau * 365
                    if ttm < 60:
                        s_val = slippage[0]
                    elif ttm >= 60 and ttm < 120:
                        s_val = slippage[1]
                    else:
                        s_val = slippage[-1]
                    cost += s_val * lots_req
                    # cost += slippage * max_op.lots
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
        print('op removed deltas: ', op, abs(op.delta / op.lots))
    toberemoved.clear()
    print('pf after liquidation: ', pf)
    return pf, cost
