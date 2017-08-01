import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div([
    html.Label('Security Type'),
    dcc.Dropdown(
        options=[
            {'label': 'CUI', 'value': 'CUI'},
            {'label': 'Future', 'value': 'Future'},
            {'label': 'Call', 'value': 'Call'},
            {'label': 'Put', 'value': 'Put'},
            {'label': 'ECUI', 'value': 'ECUI'},
            {'label': 'straddle', 'value': 'straddle'},
            {'label': 'strangle', 'value': 'strangle'},
            {'label': 'butterfly', 'value': 'butterfly'},
            {'label': 'fence', 'value': 'fence'},
            {'label': 'skew', 'value': 'skew'}

        ]
    ),


    html.Label('Position'),
    dcc.RadioItems(
        options=[
            {'label': 'Short', 'value': 'Short'},
            {'label': 'Long', 'value': 'Long'}
        ]
    ),


    html.Label('Vol_id'),
    dcc.Input(type='text'),

    html.Label('Start Date'),
    dcc.Input(type='text'),

    html.Label('End Date'),
    dcc.Input(type='text'),

    html.Label('Hedging Instructions'),
    dcc.RadioItems(
        options=[
            {'label': 'EOD Delta Hedging', 'value': 'EOD Delta Hedging'}
            # {'label': 'EOD Delta Hedging', 'value': 'EOD Delta Hedging'}
        ]
    )

])


if __name__ == '__main__':
    app.run_server(debug=True)
