import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

file_path = "./dataSet.xlsx"
df = pd.read_excel(file_path)

start_year = 1932
end_year = 2014


def calculate_metrics(country, column):
    filtered_df = df[
        (df["cty_name"] == country)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    ][column]

    metrics = {
        "Максимум": np.max(filtered_df),
        "Мінімум": np.min(filtered_df),
        "Медіана": np.median(filtered_df),
        "Середнє": np.mean(filtered_df),
        "Мода": filtered_df.mode().values[0] if not filtered_df.mode().empty else None,
        "Дисперсія": np.var(filtered_df),
        "Стандартне відхилення": np.std(filtered_df),
        "Коефіцієнт варіації (%)": (
            (np.std(filtered_df) / np.mean(filtered_df)) * 100
            if np.mean(filtered_df) != 0
            else None
        ),
        "Діапазон": np.max(filtered_df) - np.min(filtered_df),
        "Перший квартиль (Q1)": np.percentile(filtered_df, 25),
        "Третій квартиль (Q3)": np.percentile(filtered_df, 75),
    }

    return metrics


def plot_gas_price(country):
    filtered_df = df[
        (df["cty_name"] == country)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df["year"],
            y=filtered_df["gas_price_nom"],
            mode="lines+markers",
            name="Ціна на газ",
        )
    )
    fig.update_layout(
        title=f"Ціна на газ в {country} з {start_year} по {end_year} роки",
        xaxis_title="Рік",
        yaxis_title="Ціна на газ (номінальна)",
    )
    return fig


def plot_oil_price(country):
    filtered_df = df[
        (df["cty_name"] == country)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df["year"],
            y=filtered_df["oil_price_nom"],
            mode="lines+markers",
            name="Ціна на нафту",
        )
    )
    fig.update_layout(
        title=f"Ціна на нафту в {country} з {start_year} по {end_year} роки",
        xaxis_title="Рік",
        yaxis_title="Ціна на нафту (номінальна)",
    )
    return fig


def plot_oil_production(country):
    filtered_df = df[
        (df["cty_name"] == country)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df["year"],
            y=filtered_df["oil_prod32_14"],
            mode="lines+markers",
            name="Видобуток нафти",
        )
    )
    fig.update_layout(
        title=f"Кількість видобутої нафти в {country} з {start_year} по {end_year} роки",
        xaxis_title="Рік",
        yaxis_title="Кількість нафти (млн.)",
    )
    return fig


def plot_gas_production(country):
    filtered_df = df[
        (df["cty_name"] == country) & (df["year"] >= 1955) & (df["year"] <= 2014)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df["year"],
            y=filtered_df["gas_prod55_14"],
            mode="lines+markers",
            name="Видобуток газу",
        )
    )
    fig.update_layout(
        title=f"Кількість видобутого газу в {country} з 1955 по 2014 роки",
        xaxis_title="Рік",
        yaxis_title="Кількість газу (млн.)",
    )
    return fig


def plot_oil_exports(country):
    filtered_df = df[
        (df["cty_name"] == country)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df["year"],
            y=filtered_df["oil_exports"],
            mode="lines+markers",
            name="Експорт нафти",
        )
    )
    fig.update_layout(
        title=f"Кількість експортованої нафти в {country} з {start_year} по {end_year} роки",
        xaxis_title="Рік",
        yaxis_title="Експорт нафти (млн.)",
    )
    return fig


def plot_gas_exports(country):
    filtered_df = df[
        (df["cty_name"] == country)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df["year"],
            y=filtered_df["gas_exports"],
            mode="lines+markers",
            name="Експорт газу",
        )
    )
    fig.update_layout(
        title=f"Кількість експортованого газу в {country} з {start_year} по {end_year} роки",
        xaxis_title="Рік",
        yaxis_title="Експорт газу (млн.)",
    )
    return fig


def plot_arima_forecast(country, selected_column, p, d, q):
    if not country or not selected_column:
        return go.Figure()

    filtered_df = df[
        (df["cty_name"] == country)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    ].dropna(subset=[selected_column])

    data = filtered_df[selected_column]
    years = pd.to_datetime(filtered_df["year"], format="%Y")

    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()

    aic_value = model_fit.aic if model is not None else None

    forecast = model_fit.forecast(steps=10)
    forecast_years = pd.date_range(
        start=years.iloc[-1] + pd.offsets.YearBegin(), periods=10, freq="Y"
    )

    forecast = [data.iloc[-1]] + list(forecast)
    forecast_years = [years.iloc[-1]] + list(forecast_years)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=data,
            mode="lines+markers",
            name=f"Історичні дані ({selected_column})",
            zorder=1,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_years,
            y=forecast,
            mode="lines+markers",
            name=f"Прогноз ({selected_column})",
        )
    )

    fig.update_layout(
        title=f"Прогноз для {country} (AIC: {aic_value:.2f})",
        xaxis_title="Рік",
        yaxis_title="Значення",
    )

    return fig


app = dash.Dash(__name__)

available_countries = df["cty_name"].unique()

app.layout = html.Div(
    id="main",
    children=[
        html.H1("Дешборд для графіків нафти і газу"),
        html.Div(
            id="input-wrapper",
            children=[
                dcc.Dropdown(
                    id="country-dropdown",
                    options=[
                        {"label": country, "value": country}
                        for country in available_countries
                    ],
                    multi=True,
                    placeholder="Обери країну",
                ),
                dcc.Dropdown(
                    id="graph-dropdown",
                    options=[
                        {"label": "Ціна на газ", "value": "gas_price"},
                        {"label": "Ціна на нафту", "value": "oil_price"},
                        {"label": "Видобуток нафти", "value": "oil_prod"},
                        {"label": "Видобуток газу", "value": "gas_prod"},
                        {"label": "Експорт нафти", "value": "oil_exports"},
                        {"label": "Експорт газу", "value": "gas_exports"},
                    ],
                    placeholder="Обери категорію",
                ),
            ],
        ),
        dcc.Graph(id="graph-output", style={"width": "100%"}),
        html.Div(id="metrics-output"),
        html.H2("Прогноз за допомогою ARIMA"),
        html.Div(
            id="arima-input-wrapper",
            children=[
                dcc.Dropdown(
                    id="arima-country-dropdown",
                    options=[
                        {"label": country, "value": country}
                        for country in available_countries
                    ],
                    placeholder="Обери країну для прогнозу",
                ),
                dcc.Dropdown(
                    id="arima-graph-dropdown",
                    options=[
                        {"label": "Ціна на газ", "value": "gas_price_nom"},
                        {"label": "Ціна на нафту", "value": "oil_price_nom"},
                        {"label": "Видобуток нафти", "value": "oil_prod32_14"},
                        {"label": "Видобуток газу", "value": "gas_prod55_14"},
                        {"label": "Експорт нафти", "value": "oil_exports"},
                        {"label": "Експорт газу", "value": "gas_exports"},
                    ],
                    placeholder="Обери категорію для прогнозу",
                ),
                html.Div(
                    id="arima-params-wrapper",
                    children=[
                        html.Div(
                            id="param-p-wrapper",
                            children=[
                                html.Label("Параметр p:"),
                                dcc.Input(
                                    id="arima-p-input",
                                    type="number",
                                    placeholder="p",
                                    min=0,
                                    step=1,
                                    value=5,
                                ),
                            ],
                        ),
                        html.Div(
                            id="param-d-wrapper",
                            children=[
                                html.Label("Параметр d:"),
                                dcc.Input(
                                    id="arima-d-input",
                                    type="number",
                                    placeholder="d",
                                    min=0,
                                    step=1,
                                    value=1,
                                ),
                            ],
                        ),
                        html.Div(
                            id="param-q-wrapper",
                            children=[
                                html.Label("Параметр q:"),
                                dcc.Input(
                                    id="arima-q-input",
                                    type="number",
                                    placeholder="q",
                                    min=0,
                                    step=1,
                                    value=0,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Graph(id="arima-graph-output", style={"width": "100%"}),
        html.H2("Прогноз за допомогою Random Forest"),
        html.Div(
            id="random-forest-input-wrapper",
            children=[
                dcc.Dropdown(
                    id="rf-country-dropdown",
                    options=[
                        {"label": country, "value": country}
                        for country in available_countries
                    ],
                    placeholder="Обери країну для прогнозу",
                ),
                dcc.Dropdown(
                    id="rf-graph-dropdown",
                    options=[
                        {"label": "Ціна на газ", "value": "gas_price_nom"},
                        {"label": "Ціна на нафту", "value": "oil_price_nom"},
                        {"label": "Видобуток нафти", "value": "oil_prod32_14"},
                        {"label": "Видобуток газу", "value": "gas_prod55_14"},
                        {"label": "Експорт нафти", "value": "oil_exports"},
                        {"label": "Експорт газу", "value": "gas_exports"},
                    ],
                    placeholder="Обери категорію для прогнозу",
                ),
                html.Div(
                    id="rf-params-wrapper",
                    children=[
                        html.Label("Кількість дерев:"),
                        dcc.Input(
                            id="rf-n-estimators",
                            type="number",
                            placeholder="Кількість дерев",
                            min=10,
                            step=1,
                            value=100,
                        ),
                    ],
                ),
            ],
        ),
        dcc.Graph(id="rf-graph-output", style={"width": "100%"}),
    ],
)


@app.callback(
    [Output("graph-output", "figure"), Output("metrics-output", "children")],
    [Input("country-dropdown", "value"), Input("graph-dropdown", "value")],
)
def update_graph(countries, selected_graph):
    if not countries or not selected_graph:
        return (go.Figure(), [])

    fig = go.Figure()
    min_year = end_year
    max_year = start_year
    metrics_output = []

    for country in countries:
        column_mapping = {
            "gas_price": "gas_price_nom",
            "oil_price": "oil_price_nom",
            "oil_prod": "oil_prod32_14",
            "gas_prod": "gas_prod55_14",
            "oil_exports": "oil_exports",
            "gas_exports": "gas_exports",
        }

        column_name = column_mapping[selected_graph]
        country_data = df[
            (df["cty_name"] == country)
            & (df["year"] >= start_year)
            & (df["year"] <= end_year)
        ]

        if selected_graph == "gas_prod":
            country_data = df[
                (df["cty_name"] == country)
                & (df["year"] >= 1955)
                & (df["year"] <= end_year)
            ]

        min_year = min(min_year, country_data["year"].min())
        max_year = max(max_year, country_data["year"].max())

        fig.add_trace(
            go.Scatter(
                x=country_data["year"],
                y=country_data[column_name],
                mode="lines+markers",
                name=f"{country} - {selected_graph.replace('_', ' ')}",
                connectgaps=True,
            )
        )

        metrics = calculate_metrics(country, column_name)
        metrics_html = html.Div(
            id="metrics-item",
            children=[
                html.H3(f"Метрики для {country}"),
                html.Hr(),
                html.Ul(
                    id="metrics-list",
                    children=[
                        html.Li(
                            f"{metric}: {value:.2f}"
                            if isinstance(value, (int, float))
                            else f"{metric}: {value}"
                        )
                        for metric, value in metrics.items()
                    ],
                ),
            ],
        )
        metrics_output.append(metrics_html)

    fig.update_layout(
        title=f"Графік {selected_graph.replace('_', ' ')} з {min_year} по {max_year} роки",
        xaxis_title="Рік",
        yaxis_title=selected_graph.replace("_", " ").title(),
    )

    return fig, metrics_output


@app.callback(
    Output("arima-graph-output", "figure"),
    [
        Input("arima-country-dropdown", "value"),
        Input("arima-graph-dropdown", "value"),
        Input("arima-p-input", "value"),
        Input("arima-d-input", "value"),
        Input("arima-q-input", "value"),
    ],
)
def update_arima_graph(country, selected_column, p, d, q):
    return plot_arima_forecast(country, selected_column, p, d, q)


@app.callback(
    Output("rf-graph-output", "figure"),
    [
        Input("rf-country-dropdown", "value"),
        Input("rf-graph-dropdown", "value"),
        Input("rf-n-estimators", "value"),
    ],
)
def update_rf_graph(country, selected_column, n_estimators):
    return plot_rf_forecast(country, selected_column, n_estimators)


def plot_rf_forecast(country, selected_column, n_estimators):
    if not country or not selected_column:
        return go.Figure()

    filtered_df = df[
        (df["cty_name"] == country)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    ].dropna(subset=[selected_column])

    data = filtered_df[selected_column]
    years = filtered_df["year"]

    X = np.array(years).reshape(-1, 1)
    y = np.array(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    future_years = np.arange(years.max() + 1, years.max() + 11).reshape(-1, 1)
    forecast = rf.predict(future_years)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=data,
            mode="lines+markers",
            name=f"Історичні дані ({selected_column})",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_years.flatten(),
            y=forecast,
            mode="lines+markers",
            name=f"Прогноз ({selected_column})",
        )
    )

    fig.update_layout(
        title=f"Прогноз для {country} з Random Forest",
        xaxis_title="Рік",
        yaxis_title="Значення",
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
