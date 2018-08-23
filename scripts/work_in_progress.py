

def intraday_loop():
    print('================ beginning intraday loop =====================')
    unique_ts = pdf_1.time.unique()
    # bookkeeping variable. 
    dailyhedges = []
    for ts in unique_ts:
        pdf = pdf_1[pdf_1.time == ts]
        print('pdf: ', pdf)
        if ohlc:
            print('@@@@@@@@@@@@@@@ OHLC STEP GRANULARIZING @@@@@@@@@@@@@@@@')
            init_pdf, pdf, data_order = reorder_ohlc_data(pdf, pf)
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # currently, vdf only exists for settlement anyway.
        vdf = vdf_1[vdf_1.time == ts]
        for index in pdf.index:
            # get the current row and variables
            pdf_ts = pdf[pdf.index == index]
            datatype = pdf_ts.datatype.values[0]
            uid = pdf_ts.underlying_id.values[0]
            # OHLC case: if the UID is not in the portfolio, skip.
            if uid not in pf.get_unique_uids():
                continue

            val = pdf_ts.price.values[0]
            diff = 0

            if ohlc:
                # since OHLC timestamps are set after reorder,
                # this is required to select settlement vols only
                # when handling settlement price.
                vdf = vdf_1[vdf_1.time == pdf_ts.time.values[0]]
            lp = pf.hedger.last_hedgepoints[uid]
            print('===================== time: ' +
                  str(pdf_ts.time.values[0]) + ' =====================')
            print('index: ', index)

            if datatype == 'intraday':
                dailyhedges.append(
                    {'date': date, 'time': ts, 'uid': uid, 'hedge point': val})

                print('valid price move to ' + str(val) +
                      ' for uid last hedged at ' + str(lp))
            elif datatype == 'settlement':
                print('settlement price move to ' + str(val) +
                      ' for uid last hedged at ' + str(lp))

            pf.assign_hedger_dataframes(vdf, pdf_ts)

        # Step 3: Feed data into the portfolio.
            print("========================= FEED DATA ==========================")
            # NOTE: currently, exercising happens as soon as moneyness is triggered.
            # This should not be much of an issue since exercise is never
            # actually reached.
            pf, gamma_pnl, vega_pnl, exercise_barrier_profit, exercise_futures, barrier_futures \
                = feed_data(vdf, pdf_ts, pf, init_val, flat_vols=flat_vols, flat_price=flat_price)

            print("==================== PNL & BARR/EX =====================")

        # Step 4: Compute pnl for the this timestep.
            updated_val = pf.compute_value()
            # sanity check: if the portfolio is closed out during this
            # timestep, pnl = exercise proft.
            if pf.empty():
                pnl = exercise_barrier_profit
            else:
                pnl = (updated_val - init_val) + exercise_barrier_profit

            print('timestamp pnl: ', pnl)
            print('timestamp gamma pnl: ', gamma_pnl)
            print('timestamp vega pnl: ', vega_pnl)

            # update the daily variables.
            dailypnl += pnl
            dailygamma += gamma_pnl
            dailyvega += vega_pnl

            print('pf before adding bar fts/ex fts: ', pf)

            # Detour: add in exercise & barrier futures if required.
            if exercise_futures:
                print('adding exercise futures')
                pf.add_security(exercise_futures, 'OTC')
                print('pf: ', pf)

            if barrier_futures:
                print('adding barrier futures')
                pf.add_security(barrier_futures, 'hedge')

            # check: if type is settlement, defer EOD delta hedging to
            # rebalance function.
            if 'settlement' not in pdf_ts.datatype.unique():
                print('--- intraday hedge case ---')
                hedge_engine = pf.get_hedger()
                pnl = hedge_engine.apply(
                    'delta', intraday=True, ohlc=ohlc)
                dailypnl += pnl
                print('--- updating init_val post hedge ---')
                init_val = pf.compute_value()
                print('end timestamp init val: ', init_val)

    return pf, dailypnl, dailygamma, dailyvega 