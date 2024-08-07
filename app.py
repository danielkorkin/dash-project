import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import requests
import flask

# Initialize the Flask server and Dash app
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

# Define global variables
type_of_token = ["bpe", "wordpiece", "unigram"]
sizes = ["5k", "15k", "30k"]

# Function to load data from URL
def load_data(size):
    if size is None:
        size = "5k"
    base_url = "https://raw.githubusercontent.com/devium335/plots-SMTB/main/data/output/{}/{}.json"

    bpe_set, wordpiece_set, unigram_set = set(), set(), set()
    
    for token_type in type_of_token:
        url = base_url.format(size, token_type)
        response = requests.get(url)
        if response.status_code == 200:
            data_dict = response.json()
            vocab = set(data_dict["model"]["vocab"])
            if token_type == "bpe":
                bpe_set = vocab
            elif token_type == "wordpiece":
                wordpiece_set = vocab
            elif token_type == "unigram":
                unigram_set = vocab
        else:
            raise Exception(f"Failed to load data from {url}")

    bpe_lengths = [len(token) for token in bpe_set]
    wordpiece_lengths = [len(token) for token in wordpiece_set]
    unigram_lengths = [len(token) for token in unigram_set]

    return bpe_lengths, wordpiece_lengths, unigram_lengths

# Function to load HTML templates
def load_html_template(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Define the layout of the app
app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(
            id="navbar",
            children=[
                html.Div(
                    className="navbar",
                    children=[
                        html.A("Home", href="/", className="nav-link"),
                        html.Div(
                            className="dropdown",
                            children=[
                                html.Button("Graphs", className="dropbtn"),
                                html.Div(
                                    className="dropdown-content",
                                    children=[
                                        html.A(
                                            "Token Length Distributions",
                                            href="/graph/1",
                                            className="dropdown-item"
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ]
        ),
        html.Div(id="page-content"),
    ]
)

# Callback to display the appropriate page content
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/graph/1":
        return graph_layout()
    else:
        home_html = load_html_template("assets/home.html")
        return dcc.Markdown(home_html, dangerously_allow_html=True)

# Function to generate graph layout
def graph_layout():
    return html.Div(
        className="container",
        children=[
            html.H1(className="title", children="Token Length Distribution"),
            dcc.Dropdown(
                id="size-dropdown",
                options=[{"label": size, "value": size} for size in sizes],
                value="5k",
                clearable=False,
            ),
            dcc.Loading(
                id="loading-1",
                type="default",
                children=dcc.Graph(id="distribution-plot")
            )
        ],
    )

# Callback to update the graph based on selected size
@app.callback(
    Output("distribution-plot", "figure"),
    [Input("size-dropdown", "value")]
)
def update_figure(selected_size):
    if not selected_size:
        raise PreventUpdate
    bpe_lengths, wordpiece_lengths, unigram_lengths = load_data(selected_size)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(x=bpe_lengths, histnorm="percent", name="BPE", opacity=0.6)
    )
    fig.add_trace(
        go.Histogram(
            x=wordpiece_lengths, histnorm="percent", name="WordPiece", opacity=0.6
        )
    )
    fig.add_trace(
        go.Histogram(x=unigram_lengths, histnorm="percent", name="Unigram", opacity=0.6)
    )

    fig.update_layout(barmode="overlay")
    fig.update_traces(opacity=0.6)

    fig.update_layout(
        title="Token Length Distribution",
        xaxis=dict(title="Token Length"),
        yaxis=dict(title="Percentage"),
        bargap=0.2,
        bargroupgap=0.1,
    )

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
