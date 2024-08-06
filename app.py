import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
import json
import os
from dash.exceptions import PreventUpdate
import flask

# Initialize the Flask server and Dash app
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

# Define global variables
type_of_token = ["bpe", "wordpiece", "unigram"]
sizes = ["5k", "15k", "30k"]
languages = ["en", "ru"]

# Function to load data from JSON files
def load_data(size):
    if size is None:
        size = "5k"
    json_directory = f"data/output/{size}/"

    bpe_set, wordpiece_set, unigram_set = set(), set(), set()
    
    for token_type in type_of_token:
        with open(f"{json_directory}{token_type}.json", "r") as file:
            data_dict = json.loads(file.read())
            vocab = set(data_dict["model"]["vocab"])
            if token_type == "bpe":
                bpe_set = vocab
            elif token_type == "wordpiece":
                wordpiece_set = vocab
            elif token_type == "unigram":
                unigram_set = vocab

    bpe_lengths = [len(token) for token in bpe_set]
    wordpiece_lengths = [len(token) for token in wordpiece_set]
    unigram_lengths = [len(token) for token in unigram_set]

    return bpe_lengths, wordpiece_lengths, unigram_lengths

# Function to load translations from JSON files
def load_translations(language):
    file_path = f"assets/i18n/{language}.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

# Function to load HTML templates
def load_html_template(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Function to translate HTML content
def translate_html(content, translations):
    for key, value in translations.items():
        content = content.replace(f'id="{key}">', f'id="{key}">{value}')
    return content

# Define the layout of the app
app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="language-store", storage_type="local"),
        html.Div(id="navbar"),
        html.Div(id="page-content"),
    ]
)

# Callback to update the navbar based on selected language
@app.callback(Output("navbar", "children"), [Input("language-store", "data")])
def update_navbar(language):
    translations = load_translations(language or "en")
    return html.Div(
        className="navbar",
        children=[
            html.A(translations.get("home", "Home"), href="/", className="nav-link"),
            html.Div(
                className="dropdown",
                children=[
                    html.Button(
                        translations.get("graphs", "Graphs"), className="dropbtn"
                    ),
                    html.Div(
                        className="dropdown-content",
                        children=[
                            html.A(
                                translations.get(
                                    "graph_1", "Token Length Distributions"
                                ),
                                href="/graph/1",
                                className="dropdown-item"
                            )
                        ],
                    ),
                ],
            ),
            html.Div(
                className="dropdown",
                children=[
                    html.Button(
                        translations.get("language", "Language"), className="dropbtn"
                    ),
                    html.Div(
                        className="dropdown-content",
                        children=[
                            html.Div(
                                translations.get("en", "English"),
                                id="set-lang-en",
                                n_clicks=0,
                                className="dropdown-item"
                            ),
                            html.Div(
                                translations.get("ru", "Russian"),
                                id="set-lang-ru",
                                n_clicks=0,
                                className="dropdown-item"
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

# Callback to display the appropriate page content
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname"), Input("language-store", "data")]
)
def display_page(pathname, language):
    if language is None:
        language = "en"
        # Set the language to local storage if it is not set
        return dcc.Store(id="language-store", data="en")
    translations = load_translations(language)
    if pathname == "/graph/1":
        return graph_layout(translations)
    else:
        home_html = load_html_template("assets/home.html")
        translated_html = translate_html(home_html, translations)
        return dcc.Markdown(translated_html, dangerously_allow_html=True)

# Function to generate graph layout
def graph_layout(translations):
    return html.Div(
        className="container",
        children=[
            html.H1(className="title", children=translations.get("graph_title", "")),
            dcc.Dropdown(
                id="size-dropdown",
                options=[{"label": size, "value": size} for size in sizes],
                value="5k",
                clearable=False,
            ),
            dcc.Graph(id="distribution-plot"),
        ],
    )

# Callback to update the graph based on selected size
@app.callback(
    Output("distribution-plot", "figure"),
    [Input("size-dropdown", "value"), Input("language-store", "data")]
)
def update_figure(selected_size, language):
    if not selected_size:
        raise PreventUpdate
    bpe_lengths, wordpiece_lengths, unigram_lengths = load_data(selected_size)
    translations = load_translations(language or "en")

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
        title=translations.get("graph_title", ""),
        xaxis=dict(title=translations.get("token_length", "")),
        yaxis=dict(title=translations.get("percentage", "")),
        bargap=0.2,
        bargroupgap=0.1,
    )

    return fig

# Callback to update the selected language
@app.callback(
    Output("language-store", "data"),
    [Input("set-lang-en", "n_clicks"), Input("set-lang-ru", "n_clicks")]
)
def update_language(en_clicks, ru_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        return no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "set-lang-en":
        return "en"
    elif button_id == "set-lang-ru":
        return "ru"
    return no_update

if __name__ == "__main__":
    app.run_server(debug=True)
