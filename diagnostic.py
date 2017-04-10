from scripts.classes import Option, Future
from scripts.portfolio import Portfolio
brokerage = 1


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
        pf.add_security([sec], 'OTC')

    for sec in OTCs:
        pf.add_security([sec], 'OTC')

    return pf


pf1 = generate_portfolio('long')
g1 = pf1.net_greeks['C']['K7']

pf2 = generate_portfolio('short')
g2 = pf2.net_greeks['C']['K7']

ft1 = Future('K7', 300, 'C')
op1 = Option(
    350, 0.301369863013698, 'call', 0.4245569263291844, ft1, 'amer', True, 'K7', ordering=1)
op3 = Option(300, 0.473972602739726, 'call', 0.14464169782291536,
             ft1, 'amer', False, 'N7',  direc='up', barrier='amer', bullet=False,
             ko=350, ordering=2)
op2 = Option(300, 0.473972602739726, 'call', 0.14464169782291536,
             ft1, 'amer', False, 'N7',  direc='up', barrier='euro', bullet=False,
             ko=350, ordering=2)
