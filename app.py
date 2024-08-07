from typing import List, Literal, Optional, Set, Tuple

import dash
import flask
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import scipy
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate
from matplotlib_venn import venn2, venn3
from plotly.subplots import make_subplots

# Use the 'Agg' backend for Matplotlib to avoid GUI issues
matplotlib.use("Agg")

# Initialize the Flask server and Dash app
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)
app.title = "Protein Language Models"
app._favicon = "assets/favicon.ico"

# Define global variables
type_of_token = ["bpe", "wordpiece", "unigram"]
sizes = ["5k", "15k", "30k"]

# Tokenizer performance data
performance_data = {
    "pearson": {"character": 0.5836549997329712, "bpe": 0.4030608534812927},
    "expvar": {"character": 0.3428827226161957, "bpe": 0.1470852196216583},
}

# Fluorescence data
fluorescence_data = [
    {"pearson": 0.4001138508319860, "model": 6, "layer": 0, "pooling": "mean"},
    {"pearson": 0.5680390000343320, "model": 6, "layer": 1, "pooling": "mean"},
    {"pearson": 0.5519357323646550, "model": 6, "layer": 2, "pooling": "mean"},
    {"pearson": 0.64509117603302, "model": 6, "layer": 3, "pooling": "mean"},
    {"pearson": 0.6930631995201110, "model": 6, "layer": 4, "pooling": "mean"},
    {"pearson": 0.6838599443435670, "model": 6, "layer": 5, "pooling": "mean"},
    {"pearson": 0.3985217809677120, "model": 6, "layer": 0, "pooling": "attention"},
    {"pearson": 0.575803279876709, "model": 6, "layer": 1, "pooling": "attention"},
    {"pearson": 0.6514235734939580, "model": 6, "layer": 2, "pooling": "attention"},
    {"pearson": 0.7793707251548770, "model": 6, "layer": 3, "pooling": "attention"},
    {"pearson": 0.3398455083370210, "model": 6, "layer": 4, "pooling": "attention"},
    {"pearson": 0.5133616328239440, "model": 6, "layer": 5, "pooling": "attention"},
    {"pearson": 0.4011310935020450, "model": 12, "layer": 0, "pooling": "mean"},
    {"pearson": 0.554573118686676, "model": 12, "layer": 1, "pooling": "mean"},
    {"pearson": 0.604721188545227, "model": 12, "layer": 2, "pooling": "mean"},
    {"pearson": 0.7303760051727300, "model": 12, "layer": 3, "pooling": "mean"},
    {"pearson": 0.781193733215332, "model": 12, "layer": 4, "pooling": "mean"},
    {"pearson": 0.7253535985946660, "model": 12, "layer": 5, "pooling": "mean"},
    {"pearson": 0.6893606185913090, "model": 12, "layer": 6, "pooling": "mean"},
    {"pearson": 0.6480709910392760, "model": 12, "layer": 7, "pooling": "mean"},
    {"pearson": 0.6657119393348690, "model": 12, "layer": 8, "pooling": "mean"},
    {"pearson": 0.6617504954338070, "model": 12, "layer": 9, "pooling": "mean"},
    {"pearson": 0.6625848412513730, "model": 12, "layer": 10, "pooling": "mean"},
    {"pearson": 0.6537620425224300, "model": 12, "layer": 11, "pooling": "mean"},
    {"pearson": 0.3981407284736630, "model": 12, "layer": 0, "pooling": "attention"},
    {"pearson": 0.5124710202217100, "model": 12, "layer": 1, "pooling": "attention"},
    {"pearson": 0.4464473724365230, "model": 12, "layer": 2, "pooling": "attention"},
    {"pearson": 0.4861785471439360, "model": 12, "layer": 3, "pooling": "attention"},
    {"pearson": 0.6684212684631350, "model": 12, "layer": 4, "pooling": "attention"},
    {"pearson": 0.6855170130729680, "model": 12, "layer": 5, "pooling": "attention"},
    {"pearson": 0.4844455420970920, "model": 12, "layer": 6, "pooling": "attention"},
    {"pearson": 0.5026527047157290, "model": 12, "layer": 7, "pooling": "attention"},
    {"pearson": 0.4601158499717710, "model": 12, "layer": 8, "pooling": "attention"},
    {"pearson": 0.4118598699569700, "model": 12, "layer": 9, "pooling": "attention"},
    {"pearson": 0.3407143056392670, "model": 12, "layer": 10, "pooling": "attention"},
    {"pearson": 0.1714392006397250, "model": 12, "layer": 11, "pooling": "attention"},
]


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
        autosize=True,
        shapes=L_shapes,
        annotations=L_annotation,
        title=dict(text=title, x=0.5, xanchor="center"),
    )

    return p_fig


def create_combined_histogram(data1, data2, data3, title1, title2, title3):
    fig = make_subplots(
        rows=3,  # Change to 3 rows
        cols=1,  # Change to 1 column
        subplot_titles=(title1, title2, title3),
        vertical_spacing=0.15,  # Adjust vertical spacing
    )

    fig.add_trace(
        go.Bar(x=list(data1.keys()), y=list(data1.values()), name=title1), row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=list(data2.keys()), y=list(data2.values()), name=title2), row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=list(data3.keys()), y=list(data3.values()), name=title3), row=3, col=1
    )

    fig.update_layout(
        height=1200,  # Adjust height to fit all subplots
        autosize=True,
        showlegend=False,
        title_text="Token Length Distribution",
        yaxis=dict(type="log", title="log number of tokens"),
        xaxis=dict(title="token length"),
        yaxis2=dict(type="log", title="log number of tokens"),
        xaxis2=dict(title="token length"),
        yaxis3=dict(type="log", title="log number of tokens"),
        xaxis3=dict(title="token length"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


# Create a figure for Esm2_t6_8M_UR50D
def create_esm2_t6_figure():
    layers = [1, 2, 3, 4, 5, 6]
    pearson_attention = [
        0.2227207720279693,
        0.5363559722900391,
        0.4998192191123962,
        0.4638749659061432,
        0.4435491561889648,
        0.2227590233087539,
    ]
    pearson_mean = [
        0.2770425081253052,
        0.5311498045921326,
        0.5822485685348511,
        0.5407155156135559,
        0.5545222759246826,
        0.54023277759552,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=layers,
            y=pearson_mean,
            mode="lines+markers",
            name="Esm2_t6 Mean",
            line=dict(color="#2F6690", width=4),
            marker=dict(color="#2F6690", size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=pearson_attention,
            mode="lines+markers",
            name="Esm2_t6 Attention",
            line=dict(color="#B6D094", width=4),
            marker=dict(color="#B6D094", size=9),
        )
    )

    fig.update_layout(
        showlegend=True,
        title=go.layout.Title(
            text="Esm2_t6_8M_UR50D <br><sup>Protein Stability</sup>", xref="paper", x=0
        ),
        xaxis_title="Layer",
        yaxis_title="Pearson correlation",
        plot_bgcolor="white",
        autosize=True,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


# Create a figure for Esm2_t12_35M_UR50D
def create_esm2_t12_figure():
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    mean_6 = [
        0.2770425081253052,
        0.5311498045921326,
        0.5822485685348511,
        0.5407155156135559,
        0.5545222759246826,
        0.54023277759552,
        0.2770425081253052,
        0.5311498045921326,
        0.5822485685348511,
        0.5407155156135559,
        0.5545222759246826,
        0.54023277759552,
    ]
    at_6 = [
        0.2227207720279693,
        0.5363559722900391,
        0.4998192191123962,
        0.4638749659061432,
        0.4435491561889648,
        0.2227590233087539,
        0.2227207720279693,
        0.5363559722900391,
        0.4998192191123962,
        0.4638749659061432,
        0.4435491561889648,
        0.2227590233087539,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=layers,
            y=mean_6,
            mode="lines+markers",
            name="Esm2_t12 Mean",
            line=dict(color="#FFA15A", width=4),
            marker=dict(color="#FFA15A", size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=at_6,
            mode="lines+markers",
            name="Esm2_t12 Attention",
            line=dict(color="#19D3F3", width=4),
            marker=dict(color="#19D3F3", size=9),
        )
    )

    fig.update_layout(
        showlegend=True,
        title=go.layout.Title(
            text="Esm2_t12_35M_UR50D <br><sup>Protein Stability</sup>",
            xref="paper",
            x=0,
        ),
        xaxis_title="Layer",
        yaxis_title="Pearson correlation",
        plot_bgcolor="white",
        autosize=True,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


# Create a figure for Esm2_t6_8M_UR50D Fluorescence
def create_esm2_fluorescence_t6_figure():
    layers = [1, 2, 3, 4, 5, 6]
    pearson_mean = [
        0.4001138508319860,
        0.5680390000343320,
        0.5519357323646550,
        0.64509117603302,
        0.6930631995201110,
        0.6838599443435670,
    ]
    pearson_attention = [
        0.3985217809677120,
        0.575803279876709,
        0.6514235734939580,
        0.7793707251548770,
        0.3398455083370210,
        0.5133616328239440,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=layers,
            y=pearson_mean,
            mode="lines+markers",
            name="Esm2_t6 Fluorescence Mean",
            line=dict(color="#2F6690", width=4),
            marker=dict(color="#2F6690", size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=pearson_attention,
            mode="lines+markers",
            name="Esm2_t6 Fluorescence Attention",
            line=dict(color="#B6D094", width=4),
            marker=dict(color="#B6D094", size=9),
        )
    )

    fig.update_layout(
        showlegend=True,
        title=go.layout.Title(
            text="Esm2_t6_8M_UR50D <br><sup>Protein Fluorescence</sup>",
            xref="paper",
            x=0,
        ),
        xaxis_title="Layer",
        yaxis_title="Pearson correlation",
        plot_bgcolor="white",
        autosize=True,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


# Create a figure for Esm2_t12_35M_UR50D Fluorescence
def create_esm2_fluorescence_t12_figure():
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    pearson_mean = [
        0.4011310935020450,
        0.554573118686676,
        0.604721188545227,
        0.7303760051727300,
        0.781193733215332,
        0.7253535985946660,
        0.6893606185913090,
        0.6480709910392760,
        0.6657119393348690,
        0.6617504954338070,
        0.6625848412513730,
        0.6537620425224300,
    ]
    pearson_attention = [
        0.3981407284736630,
        0.5124710202217100,
        0.4464473724365230,
        0.4861785471439360,
        0.6684212684631350,
        0.6855170130729680,
        0.4844455420970920,
        0.5026527047157290,
        0.4601158499717710,
        0.4118598699569700,
        0.3407143056392670,
        0.1714392006397250,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=layers,
            y=pearson_mean,
            mode="lines+markers",
            name="Esm2_t12 Fluorescence Mean",
            line=dict(color="#FFA15A", width=4),
            marker=dict(color="#FFA15A", size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=pearson_attention,
            mode="lines+markers",
            name="Esm2_t12 Fluorescence Attention",
            line=dict(color="#19D3F3", width=4),
            marker=dict(color="#19D3F3", size=9),
        )
    )

    fig.update_layout(
        showlegend=True,
        title=go.layout.Title(
            text="Esm2_t12_35M_UR50D <br><sup>Protein Fluorescence</sup>",
            xref="paper",
            x=0,
        ),
        xaxis_title="Layer",
        yaxis_title="Pearson correlation",
        plot_bgcolor="white",
        autosize=True,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


# Create a figure for Tokenizer Performance Comparison
def create_tokenizer_performance_figure(metric):
    x = ["Character", "BPE"]
    y_char = [performance_data[metric]["character"]]
    y_bpe = [performance_data[metric]["bpe"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y_char,
            name="Character",
            marker_color="#5a97c7",
            marker_line_color="rgb(8,48,107)",
            marker_line_width=1.5,
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=y_bpe,
            name="BPE",
            marker_color="#B6D094",
            marker_line_color="#6c8f3d",
            marker_line_width=1.5,
            opacity=0.6,
        )
    )

    fig.update_layout(
        title=go.layout.Title(
            text=f"Tokenizer Performance Comparison ({metric.capitalize()})",
            xref="paper",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Tokenizer",
        yaxis_title=metric.capitalize(),
        plot_bgcolor="white",
        autosize=True,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=4,
            ticks="outside",
            tickfont=dict(
                family="Arial",
                size=16,
                color="rgb(82, 82, 82)",
            ),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


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
                                        html.A(
                                            "Protein Model Performance",
                                            href="/graph/3",
                                            className="dropdown-item",
                                        ),
                                        html.A(
                                            "Compare Tokenizer Performance",
                                            href="/graph/4",
                                            className="dropdown-item",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.A("Source Code", href="https://github.com/devium335/dash-project", className="nav-link")
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
    elif pathname == "/graph/3":
        return esm2_layout()
    elif pathname == "/graph/4":
        return tokenizer_performance_layout()
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
            dcc.RadioItems(
                id="chart-type",
                options=[
                    {"label": "Merged Chart", "value": "merged"},
                    {"label": "Multi Chart", "value": "multi"},
                ],
                value="merged",
                labelStyle={"display": "inline-block", "margin-right": "10px"},
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


# Function to generate ESM2 layout
def esm2_layout():
    return html.Div(
        className="container",
        children=[
            html.H1(className="title", children="Protein Model Performance"),
            dcc.Dropdown(
                id="esm2-model-dropdown",
                options=[
                    {"label": "Esm2_t6_8M_UR50D", "value": "t6"},
                    {"label": "Esm2_t12_35M_UR50D", "value": "t12"},
                    {"label": "Both", "value": "both"},
                ],
                value="t6",
                clearable=False,
            ),
            dcc.Dropdown(
                id="esm2-data-dropdown",
                options=[
                    {"label": "Stability", "value": "stability"},
                    {"label": "Fluorescence", "value": "fluorescence"},
                    {"label": "Both", "value": "both"},
                ],
                value="stability",
                clearable=False,
            ),
            dcc.Loading(
                id="loading-3", type="default", children=dcc.Graph(id="esm2-plot")
            ),
        ],
    )


# Function to generate tokenizer performance layout
def tokenizer_performance_layout():
    return html.Div(
        className="container",
        children=[
            html.H1(className="title", children="Compare Tokenizer Performance"),
            dcc.Dropdown(
                id="metric-dropdown",
                options=[
                    {"label": "Pearson", "value": "pearson"},
                    {"label": "Explained Variance", "value": "expvar"},
                    {"label": "Both", "value": "both"},
                ],
                value="pearson",
                clearable=False,
            ),
            dcc.Loading(
                id="loading-4",
                type="default",
                children=dcc.Graph(id="performance-plot"),
            ),
        ],
    )


# Callback to update the histogram graph based on selected size and chart type
@app.callback(
    Output("distribution-plot", "figure"),
    [Input("size-dropdown", "value"), Input("chart-type", "value")],
)
def update_histogram(selected_size, chart_type):
    if not selected_size:
        raise PreventUpdate
    bpe_lengths, wordpiece_lengths, unigram_lengths, _, _, _ = load_data(selected_size)
    bpe_data = {length: bpe_lengths.count(length) for length in set(bpe_lengths)}
    wordpiece_data = {
        length: wordpiece_lengths.count(length) for length in set(wordpiece_lengths)
    }
    unigram_data = {
        length: unigram_lengths.count(length) for length in set(unigram_lengths)
    }

    if chart_type == "merged":
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
            go.Histogram(
                x=unigram_lengths, histnorm="percent", name="Unigram", opacity=0.6
            )
        )

        fig.update_layout(barmode="overlay")
        fig.update_traces(opacity=0.6)

        fig.update_layout(
            title="Token Length Distribution",
            xaxis=dict(title="Token Length"),
            yaxis=dict(title="Percentage"),
            bargap=0.2,
            bargroupgap=0.1,
            autosize=True,
        )
    else:
        fig = create_combined_histogram(
            bpe_data, unigram_data, wordpiece_data, "BPE", "Unigram", "WordPiece"
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


# Callback to update the ESM2 plot based on selected model and data input
@app.callback(
    Output("esm2-plot", "figure"),
    [Input("esm2-model-dropdown", "value"), Input("esm2-data-dropdown", "value")],
)
def update_esm2_plot(selected_model, selected_data):
    if selected_model == "t6":
        if selected_data == "stability":
            return create_esm2_t6_figure()
        elif selected_data == "fluorescence":
            return create_esm2_fluorescence_t6_figure()
        else:
            fig_stability = create_esm2_t6_figure()
            fig_fluorescence = create_esm2_fluorescence_t6_figure()

            fig_both = go.Figure()
            for trace in fig_stability.data:
                trace.name += " (Stability)"
                trace.line.color = "#2F6690" if "Mean" in trace.name else "#B6D094"
                fig_both.add_trace(trace)
            for trace in fig_fluorescence.data:
                trace.name += " (Fluorescence)"
                trace.line.color = "#FFA15A" if "Mean" in trace.name else "#19D3F3"
                fig_both.add_trace(trace)

            fig_both.update_layout(
                showlegend=True,
                title=go.layout.Title(
                    text="Esm2_t6_8M_UR50D <br><sup>Protein Stability and Fluorescence</sup>",
                    xref="paper",
                    x=0,
                ),
                xaxis_title="Layer",
                yaxis_title="Pearson correlation",
                plot_bgcolor="white",
                autosize=True,
                xaxis=dict(
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor="rgb(204, 204, 204)",
                    linewidth=4,
                    ticks="outside",
                    tickfont=dict(
                        family="Arial",
                        size=16,
                        color="rgb(82, 82, 82)",
                    ),
                ),
                yaxis=dict(
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor="rgb(204, 204, 204)",
                    linewidth=4,
                    ticks="outside",
                    tickfont=dict(
                        family="Arial",
                        size=16,
                        color="rgb(82, 82, 82)",
                    ),
                ),
                legend=dict(
                    orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5
                ),
            )
            return fig_both
    elif selected_model == "t12":
        if selected_data == "stability":
            return create_esm2_t12_figure()
        elif selected_data == "fluorescence":
            return create_esm2_fluorescence_t12_figure()
        else:
            fig_stability = create_esm2_t12_figure()
            fig_fluorescence = create_esm2_fluorescence_t12_figure()

            fig_both = go.Figure()
            for trace in fig_stability.data:
                trace.name += " (Stability)"
                trace.line.color = "#2F6690" if "Mean" in trace.name else "#B6D094"
                fig_both.add_trace(trace)
            for trace in fig_fluorescence.data:
                trace.name += " (Fluorescence)"
                trace.line.color = "#FFA15A" if "Mean" in trace.name else "#19D3F3"
                fig_both.add_trace(trace)

            fig_both.update_layout(
                showlegend=True,
                title=go.layout.Title(
                    text="Esm2_t12_35M_UR50D <br><sup>Protein Stability and Fluorescence</sup>",
                    xref="paper",
                    x=0,
                ),
                xaxis_title="Layer",
                yaxis_title="Pearson correlation",
                plot_bgcolor="white",
                autosize=True,
                xaxis=dict(
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor="rgb(204, 204, 204)",
                    linewidth=4,
                    ticks="outside",
                    tickfont=dict(
                        family="Arial",
                        size=16,
                        color="rgb(82, 82, 82)",
                    ),
                ),
                yaxis=dict(
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor="rgb(204, 204, 204)",
                    linewidth=4,
                    ticks="outside",
                    tickfont=dict(
                        family="Arial",
                        size=16,
                        color="rgb(82, 82, 82)",
                    ),
                ),
                legend=dict(
                    orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5
                ),
            )
            return fig_both
    else:
        fig_t6_stability = create_esm2_t6_figure()
        fig_t12_stability = create_esm2_t12_figure()
        fig_t6_fluorescence = create_esm2_fluorescence_t6_figure()
        fig_t12_fluorescence = create_esm2_fluorescence_t12_figure()

        fig_both = go.Figure()
        for trace in fig_t6_stability.data:
            trace.name += " (Esm2_t6 Stability)"
            trace.line.color = "#2F6690" if "Mean" in trace.name else "#B6D094"
            fig_both.add_trace(trace)
        for trace in fig_t12_stability.data:
            trace.name += " (Esm2_t12 Stability)"
            trace.line.color = "#FFA15A" if "Mean" in trace.name else "#19D3F3"
            fig_both.add_trace(trace)
        for trace in fig_t6_fluorescence.data:
            trace.name += " (Esm2_t6 Fluorescence)"
            trace.line.color = "#A5D8FF" if "Mean" in trace.name else "#2F4B7C"
            fig_both.add_trace(trace)
        for trace in fig_t12_fluorescence.data:
            trace.name += " (Esm2_t12 Fluorescence)"
            trace.line.color = "#F0E442" if "Mean" in trace.name else "#D55E00"
            fig_both.add_trace(trace)

        fig_both.update_layout(
            showlegend=True,
            title=go.layout.Title(
                text="Esm2_t6_8M_UR50D and Esm2_t12_35M_UR50D <br><sup>Protein Stability and Fluorescence</sup>",
                xref="paper",
                x=0,
            ),
            xaxis_title="Layer",
            yaxis_title="Pearson correlation",
            plot_bgcolor="white",
            autosize=True,
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=4,
                ticks="outside",
                tickfont=dict(
                    family="Arial",
                    size=16,
                    color="rgb(82, 82, 82)",
                ),
            ),
            yaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=4,
                ticks="outside",
                tickfont=dict(
                    family="Arial",
                    size=16,
                    color="rgb(82, 82, 82)",
                ),
            ),
            legend=dict(
                orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5
            ),
        )
        return fig_both


# Callback to update the Tokenizer Performance plot based on selected metric
@app.callback(Output("performance-plot", "figure"), [Input("metric-dropdown", "value")])
def update_performance_plot(selected_metric):
    if selected_metric == "both":
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Pearson", "Explained Variance")
        )

        fig.add_trace(
            go.Bar(
                x=["Character", "BPE"],
                y=[
                    performance_data["pearson"]["character"],
                    performance_data["pearson"]["bpe"],
                ],
                name="Pearson",
                marker_color="#5a97c7",
                marker_line_color="rgb(8,48,107)",
                marker_line_width=1.5,
                opacity=0.6,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=["Character", "BPE"],
                y=[
                    performance_data["expvar"]["character"],
                    performance_data["expvar"]["bpe"],
                ],
                name="Explained Variance",
                marker_color="#B6D094",
                marker_line_color="#6c8f3d",
                marker_line_width=1.5,
                opacity=0.6,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title=go.layout.Title(
                text="Tokenizer Performance Comparison (Pearson & Explained Variance)",
                xref="paper",
                x=0.5,
                xanchor="center",
            ),
            plot_bgcolor="white",
            autosize=True,
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=4,
                ticks="outside",
                tickfont=dict(
                    family="Arial",
                    size=16,
                    color="rgb(82, 82, 82)",
                ),
            ),
            yaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=4,
                ticks="outside",
                tickfont=dict(
                    family="Arial",
                    size=16,
                    color="rgb(82, 82, 82)",
                ),
            ),
            legend=dict(
                orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5
            ),
        )
        return fig
    else:
        return create_tokenizer_performance_figure(selected_metric)


if __name__ == "__main__":
    app.run_server(debug=True, port=8080, host="0.0.0.0")
