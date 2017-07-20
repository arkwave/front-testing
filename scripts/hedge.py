# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-07-20 18:26:26
# @Last Modified by:   Ananth
# @Last Modified time: 2017-07-20 22:21:36

from timeit import default_timer as timer
import numpy as np


class Hedge:
    """Class that defines a hedge object. Hedge object has two main functionalities:
    1) inferring the hedging parameters from the portfolio
    2) applying the relevant hedges to the portfolio.

    Inference of hedging parameters is done via the _calibrate method, while applying hedges is done via the _apply method.
    """

    def __init__(self, portfolio, vdf, pdf, buckets=None):
        """Constructor. Initializes a hedge object subject to the following parameters. 

        Args:
            portfolio (TYPE): the portfolio being hedged
            vdf (TYPE): dataframe of volatilites 
            pdf (TYPE): Description
            buckets (None, optional): Description
        """
        self.params = {}
        self.greek_repr = {}
        self.vdf = vdf
        self.pdf = pdf
        self.pf = portfolio
        self.buckets = buckets if buckets is not None else [
            30, 60, 90, 120, 1e5]
        self.buckets.append(1e5)

    def _calibrate(self, hedges, flag='exp'):
        """Helper method that constructs the hedging parameters based on the greek representation fed into the hedge object.

        Example (1): flag == 'exp' indicates that hedging is being done basis greeks bucketed according to time to maturity. As such, the parameter dictionary generated is a dictionary mapping commodity and ttm to vol_id used to hedge that particular dictionary. 

        Example (2): 


        Args:
            flag (str, optional): the representation that hedging is being done with respect to. currently accepts 'exp' to hedge gamma theta and vega as per the net greeks according to bucketed ttm  

        """
        t = timer()
        params = {}
        net = {}
        if flag == 'exp':
            calibration_dic = self.pf.greeks_by_exp(self.buckets)
            net = calibration_dic
            for product in calibration_dic:
                for exp in calibration_dic[product]:
                    options = calibration_dic[product][exp][0]
                    if not options:
                        continue
                    loc = (product, exp)
                    # getting option and future months
                    opmths = [op.get_op_month() for op in options]
                    ftmths = [op.get_month() for op in options]
                    # sorting
                    opmths = set(sorted(opmths, key=lambda x: (x[1], x[0])))
                    ftmths = set(sorted(ftmths, key=lambda x: (x[1], x[0])))

                    # todo: implement ttm scaling and pick appropriate vol_id
                    # based on liquidity constraints

                    ttm = np.median([op.tau for op in options])
                    # hedges subcond? implement tomorrow.

                    # based on the ttm, pick the appropriate vol_id for this
                    # commodity. -> params[loc] = vol_id
                    volids = [product + '  ' + om + '.' + fm
                              if(((om[0] <= fm[0]) &
                                  (om[1] <= fm[1])) or
                                  ((om[0] > fm[0]) &
                                   (om[1] < fm[1]))) else ''
                              for om in opmths for fm in ftmths]

                    volids = [x for x in volids if x != '']
                    params[loc] = volids

        elif flag == 'uid':
            calibration_dic = self.pf.get_net_greeks()
            net = calibration_dic
            for product in net:
                for month in net:
                    options = net[product][month][0]
                    if not options:
                        continue
                    loc = (product, month)
                    opmths = (product, month)

        self.params = params
        self.greek_repr = net
        print('Hedge.calibrate - elapsed: ', timer() - t)

    def _apply(self):
        pass
