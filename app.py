from typing import List, Literal, Optional, Set, Tuple

import dash
import flask
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.graph_objs as go
import requests
import scipy
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate
from matplotlib_venn import venn2, venn3

# Use the 'Agg' backend for Matplotlib to avoid GUI issues
matplotlib.use("Agg")

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

    return (
        bpe_lengths,
        wordpiece_lengths,
        unigram_lengths,
        bpe_set,
        wordpiece_set,
        unigram_set,
    )


# Function to load HTML templates
def load_html_template(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Venn diagram generation function
RendererType = Literal[
    "plotly_mimetype",
    "jupyterlab",
    "nteract",
    "vscode",
    "png",
    "jpeg",
    "jpg",
    "svg",
    "pdf",
    "browser",
    "firefox",
    "chrome",
    "chromium",
    "iframe",
    "iframe_connected",
    "sphinx_gallery",
    "json",
    "notebook",
    "notebook_connected",
    "kaleido",
]


def venn_to_plotly(
    L_sets: List[Set[str]],
    L_labels: Optional[Tuple[str, str, str]] = None,
    title: Optional[str] = None,
    renderer: RendererType = "browser",
) -> go.Figure:
    n_sets = len(L_sets)

    if n_sets == 2:
        v = (
            venn2(L_sets, L_labels)
            if L_labels and len(L_labels) == n_sets
            else venn2(L_sets)
        )
    elif n_sets == 3:
        v = (
            venn3(L_sets, L_labels)
            if L_labels and len(L_labels) == n_sets
            else venn3(L_sets)
        )
    plt.close()

    L_shapes = []
    L_annotation = []
    L_color = ["FireBrick", "DodgerBlue", "DimGrey"]
    L_x_max, L_y_max, L_x_min, L_y_min = [], [], [], []

    for i in range(0, n_sets):
        center = v.centers[i]
        radius = v.radii[i]
        shape = go.layout.Shape(
            type="circle",
            xref="x",
            yref="y",
            x0=center.x - radius,
            y0=center.y - radius,
            x1=center.x + radius,
            y1=center.y + radius,
            fillcolor=L_color[i],
            line_color=L_color[i],
            opacity=0.75,
        )
        L_shapes.append(shape)
        set_label_position = v.set_labels[i].get_position()
        anno_set_label = go.layout.Annotation(
            xref="x",
            yref="y",
            x=set_label_position[0],
            y=set_label_position[1],
            text=v.set_labels[i].get_text(),
            showarrow=False,
        )
        L_annotation.append(anno_set_label)
        L_x_max.append(center.x + radius)
        L_x_min.append(center.x - radius)
        L_y_max.append(center.y + radius)
        L_y_min.append(center.y - radius)

    n_subsets = sum([scipy.special.binom(n_sets, i + 1) for i in range(0, n_sets)])
    for i in range(0, int(n_subsets)):
        subset_label_position = v.subset_labels[i].get_position()
        anno_subset_label = go.layout.Annotation(
            xref="x",
            yref="y",
            x=subset_label_position[0],
            y=subset_label_position[1],
            text=v.subset_labels[i].get_text(),
            showarrow=False,
        )
        L_annotation.append(anno_subset_label)

    off_set = 0.2
    x_max = max(L_x_max) + off_set
    x_min = min(L_x_min) - off_set
    y_max = max(L_y_max) + off_set
    y_min = min(L_y_min) - off_set

    p_fig = go.Figure()
    p_fig.update_xaxes(range=[x_min, x_max], showticklabels=False, ticklen=0)
    p_fig.update_yaxes(
        range=[y_min, y_max],
        scaleanchor="x",
        scaleratio=1,
        showticklabels=False,
        ticklen=0,
    )
    p_fig.update_layout(
        plot_bgcolor="white",
        margin=dict(b=0, l=10, pad=0, r=10, t=40),
        width=800,
        height=400,
        shapes=L_shapes,
        annotations=L_annotation,
        title=dict(text=title, x=0.5, xanchor="center"),
    )

    return p_fig


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
                                            className="dropdown-item",
                                        ),
                                        html.A(
                                            "Venn Diagram of Tokenizer Similarity",
                                            href="/graph/2",
                                            className="dropdown-item",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="page-content"),
    ]
)


# Callback to display the appropriate page content
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/graph/1":
        return graph_layout()
    elif pathname == "/graph/2":
        return venn_layout()
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
                children=dcc.Graph(id="distribution-plot"),
            ),
        ],
    )


# Function to generate Venn diagram layout
def venn_layout():
    return html.Div(
        className="container",
        children=[
            html.H1(className="title", children="Venn Diagram of Tokenizer Similarity"),
            dcc.Dropdown(
                id="size-dropdown-venn",
                options=[{"label": size, "value": size} for size in sizes],
                value="5k",
                clearable=False,
            ),
            dcc.Loading(
                id="loading-2", type="default", children=dcc.Graph(id="venn-plot")
            ),
        ],
    )


# Callback to update the histogram graph based on selected size
@app.callback(Output("distribution-plot", "figure"), [Input("size-dropdown", "value")])
def update_histogram(selected_size):
    if not selected_size:
        raise PreventUpdate
    bpe_lengths, wordpiece_lengths, unigram_lengths, _, _, _ = load_data(selected_size)

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


# Callback to update the Venn diagram based on selected size
@app.callback(Output("venn-plot", "figure"), [Input("size-dropdown-venn", "value")])
def update_venn(selected_size):
    if not selected_size:
        raise PreventUpdate
    _, _, _, bpe_set, wordpiece_set, unigram_set = load_data(selected_size)
    fig = venn_to_plotly(
        [bpe_set, wordpiece_set, unigram_set], ("BPE", "Wordpiece", "Unigram")
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
