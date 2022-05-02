# importing required libraries
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pickle
import xgboost

# initiate the app
external_stylesheets=["./assets/typography.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# read more about inline-block & flex
# https://www.geeksforgeeks.org/what-is-the-difference-between-inline-flex-and-inline-block-in-css/

# format the app

colors = {"background": "#111111", "text": "#009E73", "box": "#7F7F7F"}

app.layout = html.Div(
    children=[
        html.H1(
            "Fraud Detection",
            style={
                "textAlign": "center",
                "color": colors["text"],
                "paddingTop": "10px",
            },
        ),
        # the first line of menus
        html.Div(
            [
                # step
                html.Div(
                    [
                        html.H3(
                            "OTP Requested (times):", style={"paddingRight": "30px"},
                        ),  # this style controls the title
                        dcc.Input(
                            id="OTP_Req",
                            type="number",
                            min=0,
                            value=1,
                            style={
                                "fontsize": 15,
                                "width": 60,
                                "height": 25,
                                "color": colors[
                                    "text"
                                ],  # this style controls the input box
                            },
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
                # Amount in Transaction
                html.Div(
                    [
                        html.H3(
                            "Amount in Transaction:", style={"paddingRight": "30px"}
                        ),
                        dcc.Input(
                            id="amount",
                            type="number",
                            min=0,
                            value=10000,
                            style={
                                "fontsize": 15,
                                "width": 60,
                                "height": 25,
                                "color": colors["text"],
                            },
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
                # Flagged Site
                html.Div(
                    [
                        html.H3("Flagged Site:", style={"paddingRight": "30px"},),
                        dcc.Dropdown(
                            id="isFlaggedSite",
                            options=[
                                {"label": "No", "value": 0},
                                {"label": "Yes", "value": 1},
                            ],
                            value=1,
                            clearable=False,
                            style={
                                "fontsize": 15,
                                "width": 60,
                                "height": 30,
                                "color": colors["text"],
                            },
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
                # unrecognized Device
                html.Div(
                    [
                        html.H3(
                            "Unrecongnized Device: ", style={"paddingRight": "30px"},
                        ),
                        dcc.Dropdown(
                            id="isUnrecognizedDevice",
                            options=[
                                {"label": "No", "value": 0},
                                {"label": "Yes", "value": 1},
                            ],
                            value=1,
                            clearable=False,
                            style={
                                "fontsize": 15,
                                "width": 60,
                                "height": 30,
                                "color": colors["text"],
                            },
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
                # unrecognized location
                html.Div(
                    [
                        html.H3(
                            "Unrecongnized Location: ", style={"paddingRight": "30px"},
                        ),
                        dcc.Dropdown(
                            id="isOutsideLocation",
                            options=[
                                {"label": "No", "value": 0},
                                {"label": "Yes", "value": 1},
                            ],
                            value=1,
                            clearable=False,
                            style={
                                "fontsize": 15,
                                "width": 60,
                                "height": 30,
                                "color": colors["text"],
                            },
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
                # the submit button
                html.Div(
                    [
                        html.Button(
                            id="submit-button",
                            children="Find",
                            n_clicks=0,
                            style={"fontSize": 20, "color": colors["text"]},
                        )
                    ],
                    style={"display": "inline-block"},
                ),
            ],
            style={
                "display": "flex",
                "align-items": "center",  # vertical alignment
                "justify-content": "center",  # horizontal alignment
                "color": colors["text"],
                "background-color": colors[
                    "background"
                ],  # this style controls the entire first line of input
            },
        ),
        # the second line of menus
        html.Div(
            [],
            style={
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "color": colors[
                    "text"
                ],  # this style controls the entire second line of input
            },
        ),
        # the graphs
        dcc.Graph(id="Verdict"),
        dcc.Graph(id="Coef_Value"),
        dcc.Graph(id="Model_Evaluation"),
    ],
    style={"backgroundColor": colors["background"], "height": "100%", "width": "100%",},
)

# app functions
@app.callback(
    [
        Output(component_id="Verdict", component_property="figure"),
        Output(component_id="Coef_Value", component_property="figure"),
        Output(component_id="Model_Evaluation", component_property="figure"),
    ],
    [Input("submit-button", "n_clicks")],
    [
        State("OTP_Req", "value"),
        State("amount", "value"),
        State("isFlaggedSite", "value"),
        State("isUnrecognizedDevice", "value"),
        State("isOutsideLocation", "value"),
    ],
)

# start the function
def Fraud_Verdict(
    n_clicks, OTP_Req, amount, isFlaggedSite, isUnrecognizedDevice, isOutsideLocation,
):
    # test case plot
    test_case = [
        OTP_Req,
        amount,
        isFlaggedSite,
        isUnrecognizedDevice,
        isOutsideLocation,
    ]
    test_case = pd.DataFrame(test_case).T
    test_case.columns = [
        "OTP_Req",
        "amount",
        "isFlaggedSite",
        "isUnrecognizedDevice",
        "isOutsideLocation",
    ]
    loaded_model = pickle.load(open("fraud_det.dat", "rb"))
    test_case_verdict = loaded_model.predict(test_case)
    verdict = np.where(
        test_case_verdict == 0, "Regular Transaction", "Suspicious Transaction"
    )[0]
    verdict_col = np.where(test_case_verdict == 0, "#00ff00", "#ff0000")[0]
    testcase_prob = loaded_model.predict_proba(test_case).tolist()[0]

    verdict_plot = go.Bar(
        x=["Regular Case", "Fraud Case"], y=testcase_prob, marker={"color": "#008080"}
    )
    verdict_layout = go.Layout(
        title=dict(text="Verdict: " + verdict, font=dict(color=verdict_col)),
        xaxis=dict(title="Case Category"),
        yaxis=dict(title="Probability"),
    )
    verdict_fig = go.Figure(data=[verdict_plot], layout=verdict_layout)
    verdict_fig.update_layout(
        plot_bgcolor=colors["background"],
        font=dict(color=colors["text"]),
        paper_bgcolor=colors["background"],
    )

    # Model Coef Figure
    ft_importance = pd.DataFrame(
        loaded_model.best_estimator_.feature_importances_,
        index=[
            "OTP Requested",
            "Amount in Transaction",
            "Is the Site Flagged",
            "Is the Device Unrecognized",
            "Is the Location Unusual",
        ],
        columns=["Coef"],
    )

    coef_data = go.Pie(
        labels=ft_importance.index,
        values=ft_importance["Coef"],
        hole=0.3,
        name="",
        hovertemplate="%{label}: %{percent}",
    )
    coef_layout = go.Layout(
        title="Coefficient Importance",
        xaxis=dict(title=""),
        yaxis=dict(title=""),
        # width="100%",
        height=600,
    )
    coef_fig = go.Figure(data=[coef_data], layout=coef_layout)
    coef_fig.update_layout(
        plot_bgcolor=colors["background"],  # color of the plot background
        font=dict(color=colors["text"]),
        paper_bgcolor=colors["background"],  # color of the outside frame of the plot
    )

    # model evaluation table
    evaluation = pd.read_csv("report.csv")
    evaluation = evaluation.rename(columns={"Unnamed: 0": "Category"})
    evaluation["Category"][0:2] = ["0", "1"]
    evaluation.iloc[:, 1:] = evaluation.iloc[:, 1:].round(5)

    eval_table = go.Table(
        header=dict(
            values=list(evaluation.columns), fill_color="paleturquoise", align="left"
        ),
        cells=dict(
            values=[
                evaluation["Category"],
                evaluation["precision"],
                evaluation["recall"],
                evaluation["f1-score"],
                evaluation["support"],
            ],
            fill_color="lavender",
            align="left",
        ),
    )

    eval_layout = go.Layout(
        title="Model Evaluaiton", xaxis=dict(title=""), yaxis=dict(title=""),
    )

    eval_fig = go.Figure(data=[eval_table], layout=eval_layout)
    eval_fig.update_layout(
        plot_bgcolor=colors["background"],
        font=dict(color=colors["text"]),
        paper_bgcolor=colors["background"],
    )

    return (verdict_fig, coef_fig, eval_fig)


# launch the app
if __name__ == "__main__":
    app.run_server(debug=False, threaded=False)

