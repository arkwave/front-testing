import pandas as pd
from scripts.fetch_data import grab_data
from scripts.util import create_vanilla_option, create_underlying 
from scripts.simulation import run_simulation
from scripts.portfolio import Portfolio 
from collections import OrderedDict

# plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

# api authentication for plotly.
import plotly
plotly.tools.set_credentials_file(
    username='imran.ahmad', api_key='2HnnK4oSHSi6rWRVACVt')
plotly.tools.set_config_file(world_readable=False,
                             sharing='private')


def plot_output(log):
        # get unique values of each paramter.
        allvids = log.vol_id.unique()
        # tmp = log[log.vol_id == allvids[0]]
        tmp = log.drop_duplicates('value_date')
        # tmp.to_csv('tmp_log.csv', index=False)
        nets = tmp
        dailypnl = nets.eod_pnl_net.values
        cupnl_gross = nets.cu_pnl_gross.values
        cupnl_net = nets.cu_pnl_net.values
        gpnl = nets.cu_gamma_pnl.values
        vpnl = nets.cu_vega_pnl.values
        dates = nets.value_date
        # delta_rolled = nets.delta_rolled
        # dd_pct = nets.drawdown_pct.values
        # drd = nets.drawdown.values
        # colors = ['rgb(55, 83, 109)' for x in delta_rolled if (not x) else
        # 'rgba(204,204,204)']
        log.columns = ['future price' if x == 'px_settle' else x for x in log.columns]
        ftprices = {}
        for x in allvids:
            ftprices[x] = {}
            tmp2 = log[log.vol_id == x]
            ftprices[x]['values'] = tmp2['future price'].values
            ftprices[x]['dates'] = pd.to_datetime(tmp2.value_date.values)
        data = []
        xvals = dates
        bar = go.Bar(
            x=xvals,
            y=dailypnl,
            # text=['DD Pct: ' + str(round(x * 100, 2)) for x in dd_pct],
            name='daily pnl'
        )
        # configuring plotly data.
        gross_pnl = go.Scatter(
            x=xvals,
            y=cupnl_gross,
            name='gross cu. pnl'
        )

        net_pnl = go.Scatter(
            x=xvals,
            y=cupnl_net,
            name='net cu. pnl'
        )
        gamma_pnl = go.Scatter(
            x=xvals,
            y=gpnl,
            name='cu. gamma pnl'
        )
        vega_pnl = go.Scatter(
            x=xvals,
            y=vpnl,
            name='cu. vega pnl'
        )
        data = [bar, gross_pnl, net_pnl, gamma_pnl, vega_pnl]

        # get the net greeks for each product
        for x in ftprices:
            trace = go.Scatter(
                x=ftprices[x]['dates'],
                y=ftprices[x]['values'],
                name=x,
                yaxis='y2'
            )
            data.append(trace)

        # configuring plotly layout
        layout = go.Layout(
            title='Cu. Gross/Net/Gamma/Vega PnLs, and Future Price.',
            yaxis=dict(
                title='PnL Value'
            ),
            yaxis2=dict(
                title='Future Price',
                titlefont=dict(
                    color='rgb(148, 103, 189)'
                ),
                tickfont=dict(
                    color='rgb(148, 103, 189)'
                ),
                overlaying='y',
                side='right'
            )
        )

        fig = go.Figure(data=data, layout=layout)
        py.plot(fig)


if __name__ == "__main__":
    for year in [2010]:
        # initializing variables
        start_date = str(year) + '-05-25'
        end_date = str(year) + '-08-15'
        pdt = ['W']
        vid1 = pdt[0] + "  " + 'U' + str(year % 2010) + '.' + 'U' + str(year % 2010)
        vid2 = pdt[0] + "  " + 'Z' + str(year % 2010) + '.' + 'Z' + str(year % 2010)
        uid1 = "U" + str(year % 2010)
        uid2 = "Z" + str(year % 2010)
        #########################################

        # creating datasets
        vdf, pdf, edf = grab_data(pdt, start_date, end_date)

        # create portfolio 
        u_option = create_vanilla_option(vdf, pdf, vid1, "call", True, strike='atm', 
                                         vol=0.31, greek='vega', greekval='100000')

        z_option = create_vanilla_option(vdf, pdf, vid2, "call", False, strike='atm', 
                                         vol=0.26, greek='vega', greekval='100000')

        # specify the hedging parameters
        hedge_dict = OrderedDict({'delta': [['static', 0, 1]]})
        pf = Portfolio(hedge_dict, name='test')

        # add the securities to the portfolio
        pf.add_security([u_option, z_option], 'OTC')

        # zeroing deltas
        dic = pf.get_net_greeks()
        hedge_fts = []
        for pdt in dic:
            for mth in dic[pdt]:
                # get the delta value. 
                delta = round(dic[pdt][mth][0])
                shorted = True if delta > 0 else False
                delta = abs(delta) 
                ft, ftprice = create_underlying(pdt, mth, pdf, shorted=shorted, lots=delta)
                hedge_fts.append(ft)

        if hedge_fts:
            pf.add_security(hedge_fts, 'hedge')

        results = run_simulation(vdf, pdf, pf, flat_vols=True, plot_results=False)
        plot_output(results[0])
