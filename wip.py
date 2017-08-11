from scripts.util import close_out_deltas, create_underlying, create_vanilla_option, create_composites
contract_mths = {

    'LH':  ['G', 'J', 'K', 'M', 'N', 'Q', 'V', 'Z'],
    'LSU': ['H', 'K', 'Q', 'V', 'Z'],
    'QC': ['H', 'K', 'N', 'U', 'Z'],
    'SB':  ['H', 'K', 'N', 'V'],
    'CC':  ['H', 'K', 'N', 'U', 'Z'],
    'CT':  ['H', 'K', 'N', 'Z'],
    'KC':  ['H', 'K', 'N', 'U', 'Z'],
    'W':   ['H', 'K', 'N', 'U', 'Z'],
    'S':   ['F', 'H', 'K', 'N', 'Q', 'U', 'X'],
    'C':   ['H', 'K', 'N', 'U', 'Z'],
    'BO':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'LC':  ['G', 'J', 'M', 'Q', 'V' 'Z'],
    'LRC': ['F', 'H', 'K', 'N', 'U', 'X'],
    'KW':  ['H', 'K', 'N', 'U', 'Z'],
    'SM':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
    'COM': ['G', 'K', 'Q', 'X'],
    'OBM': ['H', 'K', 'U', 'Z'],
    'MW':  ['H', 'K', 'N', 'U', 'Z']
}


# TODO: figure out what to do with hedge options during rollovers.
# TODO: handling composites during roll_overs as well.

def roll_over(pf, vdf, pdf, date, brokerage=None, slippage=None, ttm_tol=60,
              flag=None, target_product=None):
    """Utility method that checks expiries of options currently being used for hedging. 
        If ttm < ttm_tol, closes out that position (and all accumulated deltas), 
        saves lot size/strikes, and rolls it over into the
        next month.

    Args:
        pf (TYPE): portfolio being hedged
        vdf (TYPE): volatility dataframe
        pdf (TYPE): price dataframe
        brokerage (TYPE, optional): brokerage amount
        slippage (TYPE, optional): slippage amount
        ttm_tol (int, optional): tolerance level for time to maturity.

    Returns:
        tuple: updated portfolio and cost of operations undertaken
    """
    # from itertools import cycle
    total_cost = 0
    deltas_to_close = set()
    toberemoved = []
    roll_all = False

    print('simulation.roll_over - ttm_tol: ', ttm_tol)

    # check target_volid roll check

    if target_product:
        prod = target_product
        ops = [op for op in pf.get_all_options() if op.get_product() == prod]
        if ops[0].tau * 365 <= ttm_tol:
            roll_all = True

    print('roll_all: ', roll_all)
    fa_lst = pf.get_families() if pf.get_families else [pf]
    processed = {}

    for fa in fa_lst:
        # case: first time processing this family.
        if fa not in processed:
            processed[fa] = []
        else:
            # case where all options have been processed.
            if len(processed[fa]) == len(fa.OTC_options):
                continue

        dic = fa.OTC_options if flag == 'OTC' else fa.hedge_options
        processed_ops = processed[fa]

        for op in dic.copy():
            if op in processed_ops:
                continue
            composites = []
            needtoroll = (op.tau * 365 < ttm_tol) or roll_all
            if needtoroll:
                print('rolling option ' + str(op))
                toberemoved.append(op)
                # creating the underlying future object
                fa, cost, newop, old_op, iden = contract_roll(
                    fa, op, vdf, pdf, date, flag)
                composites.append(newop)
                processed_ops.append(old_op)
                deltas_to_close.add(iden)
                total_cost += cost
                for opx in op.partners:

                    if not pf.get_families():
                        tar = fa
                    else:
                        tar = pf.get_family_containing(opx)
                    fa2, cost2, new_opx, old_opx, iden_2 = contract_roll(
                        tar, opx, vdf, pdf, date, flag)
                    composites.append(new_opx)
                    if fa2 not in processed:
                        processed[fa2] = []
                    processed[fa2].append(old_opx)
                    total_cost += cost2
                composites = create_composites(composites)

    # print('roll_over: pre refresh pf.hedges - ', pf.hedges)
    pf.refresh()
    # print('roll_over: post refresh pf.hedges - ', pf.hedges)
    print('deltas to close: ', deltas_to_close)
    pf, cost = close_out_deltas(pf, deltas_to_close)
    total_cost += cost
    # print('pf after rollover: ', pf)
    print('cost of rolling over: ', total_cost)

    for x in processed:
        print('ops contract rolled belonging to ' + x.name + ':')
        print([str(i) for i in processed[x]])

    pf.refresh()
    return pf, total_cost


def contract_roll(pf, op, vdf, pdf, date, flag):
    """Helper method that deals with contract rolliing if needed.

    Args:
        pf (TYPE): Description
        op (TYPE): Description
        vdf (TYPE): Description
        pdf (TYPE): Description
    """

    # isolating a roll condition if it exists.
    d_cond = None
    if 'delta' in pf.hedge_params:
        d_cond = [x for x in pf.hedge_params['delta'] if x[0] == 'roll']
        d_cond = d_cond[0][1] if d_cond else None

    pdt = op.get_product()
    ftmth = op.underlying.get_month()
    ft_month, ft_yr = ftmth[0], ftmth[1]
    index = contract_mths[pdt].index(ft_month) + 1
    # case: rollover to the next year
    if index >= len(contract_mths[pdt]):
        ft_yr = str(int(ft_yr) + 1)

    new_ft_month = contract_mths[pdt][
        index % len(contract_mths[pdt])] + ft_yr

    new_ft, ftprice = create_underlying(
        pdt, new_ft_month, pdf, date)

    ftprice = new_ft.get_price()

    # identifying deltas to close
    iden = (pdt, ftmth, ftprice)

    # creating the new options object - getting the tau and vol
    new_vol_id = pdt + '  ' + new_ft_month + '.' + new_ft_month
    lots = op.lots

    r_delta = None
    strike = None
    # case: delta rolling value is specified.
    if d_cond is not None:
        print('d_cond not None: ', d_cond)
        if d_cond == 50:
            strike = 'atm'
        else:
            r_delta = d_cond
    print('strike, r_delta: ', strike, r_delta)
    newop = create_vanilla_option(vdf, pdf, new_vol_id, op.char, op.shorted,
                                  date, lots=lots, strike=strike, delta=r_delta)

    # cost is > 0 if newop.price > op.price
    cost = newop.get_price() - op.get_price()

    pf.remove_security([op], flag)
    pf.add_security([newop], flag)

    return pf, cost, newop, op, iden
