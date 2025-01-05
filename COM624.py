import dash
from dash import dcc, html
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from dash.dependencies import Input, Output


def stock_analysis(company, start_year, end_date):
    start_date = dt.datetime(start_year, 1, 1)
    stock_data = yf.download(company, start=start_date, end=end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    pca_model = PCA(n_components=1)
    reduced_data = pca_model.fit_transform(scaled_close_prices)

    kmeans = KMeans(n_clusters=4)
    stock_data['Cluster'] = kmeans.fit_predict(reduced_data)

    look_back = min(90, len(scaled_close_prices) - 1)
    X_train, y_train = [], []
    for i in range(look_back, len(scaled_close_prices)):
        X_train.append(scaled_close_prices[i - look_back:i, 0])
        y_train.append(scaled_close_prices[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    lstm_model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=25, batch_size=32)

    prediction_sequence = scaled_close_prices[-look_back:].reshape(1, look_back, 1)
    predictions = []

    for _ in range(look_back):
        predicted_price = lstm_model.predict(prediction_sequence)
        predictions.append(predicted_price[0, 0])
        prediction_sequence = np.roll(prediction_sequence, -1, axis=1)
        prediction_sequence[0, -1, 0] = predicted_price[0, 0]

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    actual_dates = stock_data.index
    prediction_dates = pd.date_range(start=actual_dates[-1] + pd.DateOffset(days=1), periods=look_back)

    return stock_data, predictions, actual_dates, prediction_dates, scaled_close_prices


def cluster(data_scaled):
    pca = PCA(n_components=1)
    data_reduced = pca.fit_transform(data_scaled)
    cluster_init = KMeans(n_clusters=4, n_init=10)
    cluster = cluster_init.fit_predict(data_reduced)

    return cluster


app = dash.Dash(__name__)
company = 'TSCO.L'
end = dt.datetime(2025, 12, 31)
year = 2023
investment = 50

data, price_prediction, date_actual, date_prediction, data_scaled = stock_analysis(company, 2023, end)

app.layout = html.Div(children=[
    html.H1(children=f'{company} Stock Analysis/Predictions'),
    dcc.Dropdown(
        id='companies',
        options=[
            {'label': 'AAPL', 'value': 'AAPL'},
            {'label': 'NTDOY', 'value': 'NTDOY'},
            {'label': 'PFE', 'value': 'PFE'},
            {'label': 'MSFT', 'value': 'MSFT'},
            {'label': 'TSCO.L', 'value': 'TSCO.L'}
        ],
        value=company,
    ),
    dcc.Input(
        id='year',
        type='number',
        placeholder='Please enter start year',
        value=year,
    ),
    dcc.Input(
        id='init_investment',
        type='number',
        placeholder='Please enter initial investment',
        value=investment,
    ),
    html.Div(
        id='output'
    ),
    dcc.Graph(
        id='prices_graph',
        figure={
            'data': [
                {
                    'x': date_actual,
                    'y': data['Close'],
                    'mode': 'lines',
                    'name': f'{company} Price',
                    'line': {'color': 'red', 'width': 2},
                },
                {
                    'x': [date_actual[-1] + pd.DateOffset(days=1)],
                    'y': [None],
                    'mode': 'markers',
                    'name': 'Predicted Price',
                    'marker': {'color': 'purple', 'size': 10, 'symbol': 'circle-open'}
                },
                {
                    'x': date_prediction,
                    'y': price_prediction.flatten(),
                    'mode': 'lines',
                    'name': f'{company} Predicted Price',
                    'line': {'color': 'orange', 'dash': 'dashdot', 'width': 2},
                },
            ],
            'layout': {
                'title': {
                    'text': f'{company} Shares Prices (2023-2024)',
                    'font': {'size': 24, 'color': 'darkblue'}
                },
                'xaxis': {
                    'title': 'Time',
                    'titlefont': {'size': 18, 'color': 'darkblue'},
                    'tickfont': {'size': 12, 'color': 'black'},
                    'gridcolor': 'lightgrey'
                },
                'yaxis': {
                    'title': f'{company} Shares Prices',
                    'titlefont': {'size': 18, 'color': 'darkblue'},
                    'tickfont': {'size': 12, 'color': 'black'},
                    'gridcolor': 'lightgrey'
                },
                'legend': {
                    'x': 0,
                    'y': 1,
                    'bgcolor': 'lightgrey',
                    'bordercolor': 'black',
                    'borderwidth': 1
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'lightyellow',
            }
        }
    ),
    dcc.Graph(
        id='cluster',
        figure={
            'data': [
                {
                    'x': date_actual,
                    'y': cluster(data_scaled),
                    'mode': 'markers',
                    'name': 'Clusters',
                    'type': 'scatter',
                    'marker': {'color': 'red', 'size': 8, 'symbol': 'diamond'}
                }
            ],
            'layout': {
                'title': {
                    'text': 'Clusters',
                    'font': {'size': 24, 'color': 'darkblue'}
                },
                'xaxis': {
                    'title': 'Time',
                    'titlefont': {'size': 18, 'color': 'darkblue'},
                    'tickfont': {'size': 12, 'color': 'black'},
                    'gridcolor': 'lightgrey'
                },
                'yaxis': {
                    'title': 'Label',
                    'titlefont': {'size': 18, 'color': 'darkblue'},
                    'tickfont': {'size': 12, 'color': 'black'},
                    'gridcolor': 'lightgrey'
                },
                'legend': {
                    'x': 0,
                    'y': 1,
                    'bgcolor': 'lightgrey',
                    'bordercolor': 'black',
                    'borderwidth': 1
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'lightyellow'
            }
        }
    )
])


@app.callback(
    [Output('prices_graph', 'figure'), Output('output', 'children')],
    [Input('companies', 'value'), Input('year', 'value'), Input('init_investment', 'value')]
)
def graph_update(company, year):
    data, price_prediction, date_actual, date_prediction, data_scaled = stock_analysis(company, year, end)
    return {

        'data': [
            {
                'x': date_actual,
                'y': data['Close'],
                'mode': 'lines',
                'name': f'{company} Price',
                'line': {'color': 'red', 'width': 2},
            },
            {
                'x': [date_actual[-1] + pd.DateOffset(days=1)],
                'y': [None],
                'mode': 'markers',
                'name': 'Predicted Price',
                'marker': {'color': 'purple', 'size': 10, 'symbol': 'circle-open'}
            },
            {
                'x': date_prediction,
                'y': price_prediction.flatten(),
                'mode': 'lines',
                'name': f'{company} Predicted Price',
                'line': {'color': 'orange', 'dash': 'dashdot', 'width': 2},
            },
        ],
        'layout': {
            'title': {
                'text': f'{company} Shares Prices (2023-2024)',
                'font': {'size': 24, 'color': 'darkblue'}
            },
            'xaxis': {
                'title': 'Time',
                'titlefont': {'size': 18, 'color': 'darkblue'},
                'tickfont': {'size': 12, 'color': 'black'},
                'gridcolor': 'lightgrey'
            },
            'yaxis': {
                'title': f'{company} Shares Prices',
                'titlefont': {'size': 18, 'color': 'darkblue'},
                'tickfont': {'size': 12, 'color': 'black'},
                'gridcolor': 'lightgrey'
            },
            'legend': {
                'x': 0,
                'y': 1,
                'bgcolor': 'lightgrey',
                'bordercolor': 'black',
                'borderwidth': 1
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'lightyellow',
        }
    }


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
