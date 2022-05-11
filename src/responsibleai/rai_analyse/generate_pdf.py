import argparse
import json
import logging
import os

import plotly.graph_objects as go
import base64
import plotly.io as pio

from domonic.html import *

from _rai_insight_data import get_metric


def get_wrapped_div(div_class, items, style=None):
    if style:
        style = 'style="{}"'.format(style)
    else:
        style = ""
    return '<div class="{}"{}>'.format(div_class, style) + "".join(items) + "</div>"


def get_html_elm(html_attri, content, class_name=None, style=None):
    if style:
        style = ' style="{}"'.format(style)
    else:
        style = ""

    if class_name:
        class_name = ' class="{}"'.format(class_name)
    else:
        class_name = ""

    return "<{html_attri}{class_name}{style}>{content}</{html_attri}>".format(
        html_attri=html_attri, class_name=class_name, style=style, content=content
    )


def get_full_html(body):
    return "<!DOCTYPE html><html><body><head>{}</head>{}</body></html>".format(
        get_css(), body
    )


def get_css():
    return """<style>
        * {
          font-family: Segoe UI;
        }

        .header {
          /* border: 2px solid green; */
          /*           background-color: #ccf; */
        }

        .container {
          break-inside: avoid !important;
          page-break-inside: avoid !important;
          position: relative;
          min-height: 80%;
          /* border: 4px solid red; */
          /*           height: 11in; */
          overflow: hidden;
        }

        .left {
          float: left;
          width: 3in;
          /* border: 2px solid blue; */
          /*           background-color: #cfc; */
          padding-bottom: 9999px;
          margin-bottom: -9999px;
        }

        .main {
          position: relative;
          margin-left: 3.05in;
          /* border: 2px solid yellow; */
          /*           background-color: #ffc; */
          padding-bottom: 9999px;
          margin-bottom: -9999px;
        }
        
        .left_model_overview {
          float: left;
          width: 4.2in;
          /* border: 2px solid blue; */
          /*           background-color: #cfc; */
          padding-bottom: 9999px;
          margin-bottom: -9999px;
        }

        .main_model_overview {
          position: relative;
          margin-left: 4.25in;
          /* border: 2px solid yellow; */
          /*           background-color: #ffc; */
          padding-bottom: 9999px;
          margin-bottom: -9999px;
        }

        #footer {
          background-color: #fcc;
        }

        .box {
          width: 5in;
          height: auto;
        }

        img {
          width: 5in;
          height: auto;
        }
        
        .left_img {
          width: 3in;
          height: auto;
        }
        
        .image_div {
          break-inside: avoid;
          /*border: 2px solid black;*/
          /*border-radius: 5px;*/
          /*width: 99%;*/
        }
        
        .nobreak_div {
          break-inside: avoid !important;
          page-break-inside: avoid !important;
        }
        
        .nobreak_div_padding {
          content: "";
          display: block;
          height: 100px;
          margin-bottom: -100px;
          break-inside: avoid !important;
          page-break-inside: avoid !important;
        }
        
        .cell {
          border-collapse: collapse;
          border: 0.5px solid rgba(199, 199, 199, 1);
          padding: 5px 10px;
        }
        
        .header_cell {
          border-collapse: collapse;
          border: 0px;
          padding: 5px 10px;
        }
        
        .table {
          border-collapse: collapse;
          border-style: hidden;
        }
        }

      </style>"""


def get_page_divider(text):
    elem = div(
        _style="height:36px; margin-top: 10px; page-break-after: avoid;",
        _class="nobreak_div",
    ).append(
        div(
            _style="width: 100%; height: 12px; border-bottom: 1px solid #605E5C;"
        ).append(
            span(
                text,
                _style="font-size: 14px; background-color: #605E5C; "
                "padding: 0 10px; border-radius: 18px; color: #ffffff",
            )
        )
    )
    return elem


def get_de_image(data):
    png_base64 = get_de_bar_plot(data)
    return str(
        div(img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div")
    )


def get_cohorts_performance_image(data, m, get_bar_plot_func):
    png_base64 = get_bar_plot_func(data, m)
    return div(
        img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
    )


def get_fi_image(data):
    png_base64 = get_fi_bar_plot(data)
    return div(
        img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
    )


def get_cohorts_performance_container(data, m, get_bar_plot_func, message):
    cohort_left_heading = h3("{}: {}".format(message["left_heading"], m))
    cohorts_list = ul()
    for c in data:
        cohorts_list.append(li("{}: {}".format(c["short_label"], c["label"])))

    container = div(cohort_left_heading, cohorts_list, _class="nobreak_div")

    cp_left_container = div(container, _class="left")

    cp_main_heading = h3("{}: {}".format(message["main_heading"], m))
    cp_main_image = get_cohorts_performance_image(data, m, get_bar_plot_func)

    container = div(cp_main_heading, cp_main_image, _class="nobreak_div")

    cp_main_container = div(container, _class="main")

    return str(div(cp_left_container, cp_main_container, _class="container"))


def get_feature_importance_page(data):

    fi_left_heading = p(
        "Understand factors that have impacted your model predictions the most. "
        "These are factors that may account for performance levels and differences."
    )
    feature_list = ul()
    for k, v in data.items():
        feature_list.append(li("{}: {}".format(v["short_label"], k)))

    container = div(fi_left_heading, feature_list, _class="nobreak_div")

    fi_left_container = div(container, _class="left")

    heading_main = div(h3("Feature Importance"), get_fi_image(data), _class="main")
    return div(
        get_page_divider("Feature relevance (explanability)"),
        fi_left_container,
        heading_main,
        _class="container",
    )


def get_fi_bar_plot(data):
    y_data = [v["short_label"] for k, v in data.items()]
    x_data = [v["value"] for k, v in data.items()]
    max_x = max(x_data)
    x_range = [0, max_x]
    x_data = [[x, max_x - x] for x in x_data]

    # tickvals = [0.0, 0.25, 0.5, 0.75, 1.0]
    # ticktext = [0.0, 0.25, 0.5, 0.75, 1.0]
    tickappend = ""

    return get_bar_plot(y_data, x_data, tickappend=tickappend, xrange=x_range)


def get_binary_cp_bar_plot(data, m):
    metric_name = m
    y_data = [y["cohort_short_name"] for y in data["cohorts"]]
    x_data = [int(x[metric_name] * 100) for x in data["cohorts"]]
    x_data = [[x, 100 - x] for x in x_data]
    legend = [m]
    tickvals = [0, 25, 50, 75, 100]
    ticktext = [str(x) + "%" for x in tickvals]

    return get_bar_plot(
        y_data,
        x_data,
        legend=legend,
        tickvals=tickvals,
        ticktext=ticktext,
        tickappend="%",
    )


def get_de_bar_plot(data):
    y_data = [
        y["label"] + "<br>" + str(int(y["population"] * 100)) + "% pop."
        for y in data["classes"]
    ]
    x_data = [int(float(x["prediction_0_ratio"]) * 100) for x in data["classes"]]
    x_data = [[x, 100 - x] for x in x_data]
    legend = ['Predicted as "' + y["prediction_0_name"] + '"' for y in data["classes"]]
    tickvals = [0, 25, 50, 75, 100]
    ticktext = [str(x) + "%" for x in tickvals]

    return get_bar_plot(
        y_data,
        x_data,
        legend=legend,
        tickvals=tickvals,
        ticktext=ticktext,
        tickappend="%",
    )


def get_dot_plot(center, ep, em):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[center],
            y=[0.5],
            mode="markers",
            error_x=dict(
                type="data",
                symmetric=False,
                array=[ep],
                arrayminus=[em],
                color="rgba(39, 110, 237, 1)",
                width=15,
            ),
        )
    )
    fig.update_xaxes(
        showgrid=False,
        # tickvals=[0, 25, 50, 75, 100],
        # ticktext=["0%", "25%", "50%", "75%", "100%"],
        # range=[0, 100]
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=3,
        showticklabels=False,
    )
    fig.update_layout(height=240, width=700, plot_bgcolor="white")

    fig.update_traces(
        marker=dict(
            color="white", size=20, line=dict(width=2, color="rgba(39, 110, 237, 1)")
        )
    )

    png = pio.to_image(fig)
    png_base64 = base64.b64encode(png).decode("ascii")

    return png_base64


def get_bar_plot(
    y_data,
    x_data,
    legend=None,
    threshold=None,
    tickvals=None,
    ticktext=None,
    xrange=None,
    tickappend="",
):
    fig = go.Figure()
    series = 0

    def get_colors(series, index):
        colors2 = ["rgba(39, 110, 237, 1)", "rgba(218, 227, 243, 1)"]
        return colors2[index % 2]

    def get_show_legend(legend, index):
        if not legend:
            return False
        if index == 0:
            return True
        return False

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(
                go.Bar(
                    x=[xd[i]],
                    y=[yd],
                    width=0.9,
                    orientation="h",
                    marker=dict(
                        color=get_colors(series, i),
                        line=dict(color="rgb(248, 248, 249)", width=1),
                    ),
                    showlegend=get_show_legend(legend, series),
                )
            )
            series = series + 1

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False,
            domain=[0.15, 1],
            tickvals=tickvals if tickvals else None,
            ticktext=ticktext if ticktext else None,
            range=xrange if xrange else None,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode="stack",
        bargap=0,
        paper_bgcolor="rgb(255, 255, 255)",
        plot_bgcolor="rgb(255, 2555, 255)",
        margin=dict(l=25, r=25, t=25, b=25),
    )

    annotations = []

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=0.14,
                y=yd,
                xanchor="right",
                text=str(yd),
                font=dict(family="Consolas", size=18, color="rgb(67, 67, 67)"),
                showarrow=False,
                align="right",
            )
        )
        # labeling the first percentage of each bar (x_axis)
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=0.17,
                y=yd,
                text=str(round(xd[0], 1)) + tickappend,
                font=dict(family="Consolas", size=22, color="rgb(0, 0, 0)"),
                showarrow=False,
            )
        )

    fig.update_layout(
        annotations=annotations,
        width=700,
        height=len(x_data) * 70 + 30,
    )

    def get_legend_y(barchart_length):
        lookup = {
            1: 1.55,
            2: 1.27,
            3: 1.15,
            4: 1.11,
            5: 1.09,
            6: 1.07,
            7: 1.06,
            8: 1.05,
            9: 1.04,
            10: 1.03,
        }
        return lookup[barchart_length]

    fig.update_traces(cliponaxis=False)
    if threshold:
        fig.add_vline(
            x=threshold,
            annotation_text="Threshold",
            annotation_position="bottom right",
            line_width=3,
            line_dash="dash",
            line_color="red",
        )

    if legend:
        fig["data"][0]["name"] = legend[0]
        fig.update_layout(
            legend=dict(
                yanchor="top", y=get_legend_y(len(x_data)), xanchor="left", x=0.15
            )
        )

    png = pio.to_image(fig)
    png_base64 = base64.b64encode(png).decode("ascii")

    return png_base64


def get_de_box_plot(data):
    return get_box_plot(data)


def get_de_box_plot_image(data):
    processed_label = data
    for c in processed_label["data"]:
        c["label"] = (
            c["label"] + "<br>" + str(int(100 * round(c["population"], 3))) + "%"
        )
        c["datapoints"] = c["prediction"]
    png_base64 = get_de_box_plot(processed_label)
    return div(
        img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
    )


def get_box_plot(data):
    fig = go.Figure()
    for i in data["data"]:
        fig.add_trace(
            go.Box(
                x=i["datapoints"],
                boxpoints=False,
                line_color="rgba(39, 110, 237, 1)",
                fillcolor="rgba(218, 227, 243, 1)",
                name=i["label"],
                showlegend=False,
            )
        )

    fig.update_layout(
        paper_bgcolor="rgb(255, 255, 255)",
        plot_bgcolor="rgb(255, 2555, 255)",
        margin=dict(l=25, r=25, t=25, b=25),
        width=700,
        height=len(data["data"]) * 70 + 60,
        # annotations=annotations,
        boxgap=0,
    )

    fig.update_layout()

    png = pio.to_image(fig)
    png_base64 = base64.b64encode(png).decode("ascii")

    return png_base64


def get_model_overview(data):
    model_left_items = []

    model_left_items.append(div(h3("Purpose"), p(data["ModelSummary"])))

    if data["ModelType"] == "binary_classification":
        model_left_items.append(
            div(
                p(
                    "Classification: {} vs {}".format(
                        data["classes"][0], data["classes"][1]
                    )
                )
            )
        )
    else:
        model_left_items.append(div(p("This is {} model".format(data["ModelType"]))))

    model_left_items.append(
        div(
            h3("How the model is evaluated"),
            p(
                "This model is evaluated on a test set with {} datapoints.".format(
                    len(data["y_test"])
                )
            ),
        )
    )

    model_overview_left_container = div(model_left_items, _class="left_model_overview")

    model_main_items = []
    model_main_items.extend(
        [
            h3("Target values"),
            p(
                "Here are your defined target minimum/maximum performance levels "
                "and/or target performance differences between groups/cohorts:"
            ),
        ]
    )

    metric_targets_elems = []
    for item in data["metrics_targets"]:
        metric_targets_elems.append(li(item))

    model_main_items.append(
        div(
            ul(metric_targets_elems),
            _style="border: 2px solid black; border-radius: 5px;",
        )
    )

    model_overview_main_container = div(model_main_items, _class="main_model_overview")

    heading = [h1(data["ModelName"])]
    if data["runinfo"]:
        heading.append(
            p(
                "Generated by {} on {}".format(
                    data["runinfo"]["submittedBy"], data["runinfo"]["startTimeUtc"]
                )
            )
        )

    model_overview_container = div(
        div(heading, _class="header"),
        get_page_divider("Model Summary"),
        model_overview_left_container,
        model_overview_main_container,
        _class="container",
    )

    return model_overview_container


def get_cohorts_page(data, metric_config):
    left_elems = div(
        p("Observe evidence of model performance across your passed cohorts:"),
        _class="left",
    )
    cp_heading_main = div(_class="main")
    heading_section = str(
        div(
            get_page_divider("Cohorts"), left_elems, cp_heading_main, _class="container"
        )
    )

    # start section for each predefined cohort
    def populate_cp_container(key, container, m, data):
        if key not in data.keys():
            return

        def get_regression_bar_plot(d, m):
            first_data_point = next(iter(d), None)
            threshold = None
            if first_data_point:
                threshold = first_data_point.get("threshold", None)
            y_data = [
                [y["short_label"], str(int(y["population"] * 100)) + "%"] for y in d
            ]
            y_data = ["<br>".join(y) for y in y_data]
            x_data = [x[m] for x in d]
            if m in ["accuracy_score", "recall_score", "precision_score", "f1_score"]:
                max_x = 1
            else:
                max_x = max(x_data)
            x_data = [[x, max_x - x] for x in x_data]
            legend = [m]

            return get_bar_plot(y_data, x_data, legend=legend, threshold=threshold)

        message_lookup = {
            "cohorts": {
                "left_heading": "My Cohorts",
                "main_heading": "My prebuilt dataset cohorts",
            },
            "error_analysis_max": {
                "left_heading": "Highest ranked cohorts",
                "main_heading": "Highest ranked cohorts",
            },
            "error_analysis_min": {
                "left_heading": "Lowest ranked cohorts",
                "main_heading": "Lowest ranked cohorts",
            },
        }

        filtered_data = [d for d in data[key] if m in d.keys()]
        if len(filtered_data) == 0:
            return

        container[key].append(
            get_cohorts_performance_container(
                filtered_data, m, get_regression_bar_plot, message_lookup[key]
            )
        )

    cohort_performance_containers = {
        "cohorts": [],
        "error_analysis_max": [],
        "error_analysis_min": [],
    }
    for k in ["cohorts", "error_analysis_max", "error_analysis_min"]:
        for m in metric_config:
            populate_cp_container(k, cohort_performance_containers, m, data)

    cohort_performance_section = "".join(cohort_performance_containers["cohorts"])
    cohort_performance_section = cohort_performance_section + "".join(
        cohort_performance_containers["error_analysis_max"]
    )
    cohort_performance_section = cohort_performance_section + "".join(
        cohort_performance_containers["error_analysis_min"]
    )

    return str(heading_section + cohort_performance_section)


def get_causal_page(data):
    left_elem = [
        div(
            p(
                "Causal analysis answers real world what if "
                "questions about how changes of treatments would impact a real world outcome."
            )
        )
    ]

    left_container = div(left_elem, _class="left")

    main_elems = []

    def get_causal_dot_plot(center, em, ep):
        png_base64 = get_dot_plot(center, em, ep)
        return div(
            img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
        )

    def get_table_row(data):
        table_row_elems = []
        for v in data:
            table_row_elems.append(td(v, _class="cell"))
        return tr(table_row_elems, _class="row")

    def get_table(data):
        horizontal_headings = [
            "Index",
            "Current<br>Treatment",
            "Recommended<br>Treatment",
            "Effect<br>Estimate",
        ]
        headings_td = [td(x, _class="header_cell") for x in horizontal_headings]
        headings = thead(tr(headings_td, _class="row"), _class="table-head")

        rows_elems = []
        for elem in data:
            rows_elems.append(get_table_row(elem))

        body = tbody(rows_elems, _class="table-body")

        return table(headings, body, _class="table")

    for f in data["global_effect"].values():
        main_elems.append(
            div(
                h3(f["feature"]),
                p(
                    'On average, increasing "{}" by 1 unit increases the prediction outcome by {}'.format(
                        f["feature"], round(f["point"], 3)
                    )
                ),
                get_causal_dot_plot(
                    f["point"], f["ci_upper"] - f["point"], f["point"] - f["ci_lower"]
                ),
                _class="nobreak_div",
            )
        )

        main_elems.append(
            h3(
                'Top data points responding the most to treatment on "{}":'.format(
                    f["feature"]
                )
            )
        )
        main_elems.append(
            p(
                "What data points experience the largest impact (causal effect) to changes"
                ', while treating "{}"'.format(f["feature"])
            )
        )

        main_elems.append(
            get_table(
                [
                    [
                        p["index"],
                        p["Current treatment"],
                        p["Treatment"],
                        round(p["Effect of treatment"], 2),
                    ]
                    for p in data["top_local_policies"][f["feature"]]
                ]
            )
        )

    main_container = div(main_elems, _class="main")

    return div(
        div(
            get_page_divider("Causal"),
            left_container,
            main_container,
            _class="nobreak_div",
        ),
        _class="container nobreak_div",
    )


class ClassificationComponents:
    @staticmethod
    def get_model_overview_page(data):
        return get_model_overview(data)

    @staticmethod
    def get_data_explorer_page(data):
        de_heading_left_elms = p(
            "Evaluate your dataset to assess representation of identified subgroups:"
        )

        de_heading_left_container = div(de_heading_left_elms, _class="left")

        de_main_elems = []

        def get_de_bar_plot(data):
            class_0 = data[0]["prediction"][0]

            y_data = [
                str(c["label"]) + "<br>" + str(int(100 * c["population"])) + "%"
                for c in data
            ]
            x_data = [
                100 * (list(c["prediction"]).count(class_0)) / (len(c["prediction"]))
                for c in data
            ]
            x_data = [[x, 100 - x] for x in x_data]

            tickvals = [0, 25, 50, 75, 100]
            ticktext = [str(x) + "%" for x in tickvals]
            legend = ["Predicted as {}".format(class_0)]

            png_base64 = get_bar_plot(
                y_data,
                x_data,
                legend=legend,
                tickvals=tickvals,
                ticktext=ticktext,
                tickappend="%",
            )
            return div(
                img(_src="data:image/png;base64,{}".format(png_base64)),
                _class="image_div",
            )

        for c in data:
            de_main_elems.append(h3(c["feature_name"]))
            for i in c["data"]:
                de_main_elems.append(
                    p(
                        '"{}" have {}% {}'.format(
                            i["label"],
                            round(i[c["primary_metric"]] * 100, 1),
                            c["primary_metric"],
                        )
                    )
                )
            de_main_elems.append(
                div(
                    p(
                        "Predicted classification output of the different subgroups are as follows:"
                    ),
                    get_de_bar_plot(c["data"]),
                    _class="nobreak_div",
                )
            )

        de_main_container = div(de_main_elems, _class="main")

        return str(
            div(
                get_page_divider("Data Explorer"),
                de_heading_left_container,
                de_main_container,
                _class="container",
            )
        )

    @staticmethod
    def _get_model_performance_explanation_text(metric, data):
        if metric == "accuracy_score":
            return div(
                h3(
                    "{}% Accuracy".format(
                        str(int(100 * data["metrics"]["accuracy_score"]))
                    )
                ),
                p(
                    "{}% of data points have the correct prediction.<br>".format(
                        str(int(100 * data["metrics"]["accuracy_score"]))
                    )
                ),
                p(
                    "Accuracy = correct predictions / all predictions<br>"
                    "= ({} + {}) / {}".format(
                        data["confusion_matrix"]["tp"],
                        data["confusion_matrix"]["tn"],
                        data["confusion_matrix"]["tp"]
                        + data["confusion_matrix"]["tn"]
                        + data["confusion_matrix"]["fp"]
                        + data["confusion_matrix"]["fn"],
                    )
                ),
            )

    @staticmethod
    def _get_confusion_matrix_grid(data):
        cm = data["confusion_matrix"]
        negative = data["classes"][0]
        positive = data["classes"][1]
        return table(
            tr(
                th(_class="header_cell"),
                th('Acutal<br>"{}"'.format(positive), _class="header_cell"),
                th('Acutal<br>"{}"'.format(negative), _class="header_cell"),
            ),
            tr(
                th('Predicted<br>"{}"'.format(positive), _class="header_cell"),
                td(
                    p(
                        cm["tp"],
                        _style="font-size:22px; color:#107C10; text-align: center;",
                    ),
                    p(
                        "correct prediction",
                        _style="font-size:14px; text-align: center;",
                    ),
                    _class="cell",
                ),
                td(
                    cm["fn"],
                    _class="cell",
                    _style="font-size:22px; color:#A80000; text-align: center;",
                ),
            ),
            tr(
                th('Predicted<br>"{}"'.format(negative), _class="header_cell"),
                td(
                    cm["fp"],
                    _class="cell",
                    _style="font-size:22px; color:#A80000; text-align: center;",
                ),
                td(
                    p(
                        cm["tn"],
                        _style="font-size:22px; color:#107C10; text-align: center;",
                    ),
                    p(
                        "correct prediction",
                        _style="font-size:14px; text-align: center;",
                    ),
                    _class="cell",
                ),
            ),
            _style="width: 5in",
        )

    @staticmethod
    def get_model_performance_page(data):

        left_metric_elems = [p("Observe evidence of your model performance here:")]

        def get_metric_bar_plot(mname, data):
            y_0_filtermap = [
                True if i == data["classes"][0] else False for i in data["y_test"]
            ]
            y_1_filtermap = [
                True if i == data["classes"][1] else False for i in data["y_test"]
            ]
            class_0_metric = get_metric(
                mname, data["y_pred"][y_0_filtermap], data["y_test"][y_0_filtermap]
            )
            class_1_metric = get_metric(
                mname, data["y_pred"][y_1_filtermap], data["y_test"][y_1_filtermap]
            )

            y_data = [
                'Acutal "{}"'.format(data["classes"][0]),
                'Acutal "{}"'.format(data["classes"][1]),
            ]
            x_data = [int(class_0_metric * 100), int(class_1_metric * 100)]
            x_data = [[x, 100 - x] for x in x_data]

            legend = [m]
            tickvals = [0, 25, 50, 75, 100]
            ticktext = [str(x) + "%" for x in tickvals]

            png_base64 = get_bar_plot(
                y_data,
                x_data,
                legend=legend,
                tickvals=tickvals,
                ticktext=ticktext,
                tickappend="%",
            )
            return div(
                img(_src="data:image/png;base64,{}".format(png_base64)),
                _class="image_div",
            )

        main_elems = []
        main_elems.append(ClassificationComponents._get_confusion_matrix_grid(data))

        for m in data["metrics"]:
            left_metric_elems.append(
                ClassificationComponents._get_model_performance_explanation_text(
                    m, data
                )
            )
            main_elems.append(get_metric_bar_plot(m, data))

        left_container = div(left_metric_elems, _class="left")
        main_container = div(main_elems, _class="main")
        return str(
            div(
                get_page_divider("Model Performance"),
                left_container,
                main_container,
                _class="container",
            )
        )

    @staticmethod
    def get_cohorts_page(data, metrics_config):
        return get_cohorts_page(data, metrics_config)

    @staticmethod
    def get_feature_importance_page(data):
        return get_feature_importance_page(data)

    @staticmethod
    def get_causal_page(data):
        return get_causal_page(data)

    @staticmethod
    def get_fairlearn_page(data):
        left_elems = [
            div(
                p(
                    "Understand your model’s fairness issues "
                    "using group-fairness metrics across sensitive features and cohorts. "
                    "Pay particular attention to the subgroups who receive worse treatments "
                    "(predictions) by your model."
                )
            )
        ]

        for f in data:
            left_elems.append(h3('Feature "{}"'.format(f)))
            metric_section = []
            for metric_key, metric_details in data[f]["metrics"].items():
                if metric_key in ["false_positive", "false_negative"]:
                    continue
                left_elems.append(
                    p(
                        '"{}" has the highest {}: {}'.format(
                            metric_details["group_max"][0],
                            metric_key,
                            round(metric_details["group_max"][1], 2),
                        )
                    )
                )
                left_elems.append(
                    p(
                        '"{}" has the lowest {}: {}'.format(
                            metric_details["group_min"][0],
                            metric_key,
                            round(metric_details["group_min"][1], 2),
                        )
                    )
                )
                if metric_details["kind"] == "difference":
                    metric_section.append(
                        p(
                            "Maximum difference in {} is {}".format(
                                metric_key,
                                round(
                                    metric_details["group_max"][1]
                                    - metric_details["group_min"][1],
                                    2,
                                ),
                            )
                        )
                    )
                elif metric_details["kind"] == "ratio":
                    metric_section.append(
                        p(
                            "Minimum ratio of {} is {}".format(
                                metric_key,
                                round(
                                    metric_details["group_min"][1]
                                    / metric_details["group_max"][1],
                                    2,
                                ),
                            )
                        )
                    )
            left_elems.append(div(metric_section, _class="nobreak_div"))

        left_container = div(left_elems, _class="left")

        def get_fairness_bar_plot(data):
            y_data = [
                str(c) + "<br>" + str(int(100 * data[c]["population"])) + "%"
                for c in data
            ]
            x_data = [
                100
                * (get_metric("selection_rate", data[c]["y_test"], data[c]["y_pred"]))
                for c in data
            ]
            x_data = [[x, 100 - x] for x in x_data]

            tickvals = [0, 25, 50, 75, 100]
            ticktext = [str(x) + "%" for x in tickvals]

            png_base64 = get_bar_plot(
                y_data, x_data, tickvals=tickvals, ticktext=ticktext, tickappend="%"
            )

            return div(
                img(_src="data:image/png;base64,{}".format(png_base64)),
                _class="image_div",
            )

        def get_table_row(heading, data):
            table_row_elems = []
            table_row_elems.append(th(heading, _class="header_cell"))
            for v in data:
                table_row_elems.append(td(v, _class="cell"))
            return tr(table_row_elems, _class="row")

        def get_table(data):
            metric_list = [d for d in data["metrics"]]
            # metric_list = [m for sublist in metric_list for m in sublist]

            horizontal_headings = [d.replace("_", "<br>") for d in metric_list]
            vertical_headings = list(data["statistics"].keys())

            headings_td = [td(_class="header_cell")] + [
                td(x, _class="header_cell") for x in horizontal_headings
            ]
            headings = thead(tr(headings_td, _class="row"), _class="table-head")

            rows_elems = []
            for vh in vertical_headings:
                row_data = []
                for m in metric_list:
                    row_data.append(round(data["metrics"][m]["group_metric"][vh], 2))
                rows_elems.append(get_table_row(vh, row_data))

            body = tbody(rows_elems, _class="table-body")

            return table(headings, body, _class="table")

        main_elems = []
        # Selection rate
        for f in data:
            main_elems.append(
                div(
                    h2('Feature "{}"'.format(f)),
                    h3("Selection rate"),
                    get_fairness_bar_plot(data[f]["statistics"]),
                    _class="nobreak_div",
                )
            )

            main_elems.append(
                div(
                    h3("Analysis across cohorts"),
                    get_table(data[f]),
                    _class="nobreak_div",
                )
            )

        main_container = div(main_elems, _class="main")

        return str(
            div(
                get_page_divider("Fairness"),
                left_container,
                main_container,
                _class="container nobreak_div",
            )
        )


class RegressionComponents:
    metric_name_lookup = {
        "mean_squared_error": "Mean squared error",
        "mean_absolute_error": "Mean absolute error",
    }

    @staticmethod
    def get_model_overview_page(data):
        return get_model_overview(data)

    @staticmethod
    def get_bar_plot_explanation_image():
        import base64

        with open("./box_plot_explain.png", "rb") as img_file:
            png_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        return div(
            img(_src="data:image/png;base64,{}".format(png_base64), _class="left_img"),
            _class="image_div",
        )

    @staticmethod
    def get_data_explorer_page(data):
        de_heading_left_elms = p(
            "Evaluate your dataset to assess representation of identified subgroups:"
        )

        de_heading_left_elms.append(
            RegressionComponents.get_bar_plot_explanation_image()
        )

        de_heading_left_container = div(de_heading_left_elms, _class="left")

        de_main_elems = []

        for c in data:
            de_main_elems.append(h3(c["feature_name"]))
            for i in c["data"]:
                de_main_elems.append(
                    p(
                        "For {} datapoints, {} is the average of the {} between the actual value and the predicted value".format(
                            i["label"],
                            round(i[c["primary_metric"]], 1),
                            c["primary_metric"],
                        )
                    )
                )
            de_main_elems.append(
                div(
                    p(
                        "The distribution of prediction value for the different sets of data points are as follows:"
                    ),
                    get_de_box_plot_image(c),
                    _class="nobreak_div",
                )
            )

        de_main_container = div(de_main_elems, _class="main")

        return str(
            div(
                get_page_divider("Data Explorer"),
                de_heading_left_container,
                de_main_container,
                _class="container",
            )
        )

    @staticmethod
    def get_metric_explanation_text(mname, mvalue):
        text_lookup = {
            "mean_squared_error": "{} is the average of the sqaured difference between the actual values and the predicted values.",
            "mean_absolute_error": "{} is the average of the absolute error values.",
        }
        return text_lookup[mname].format(round(mvalue, 2))

    @staticmethod
    def get_distributions_plot(data):
        bar_plot_data = {
            "data": [
                {"label": "Predicted<br>Value", "datapoints": data["y_pred"]},
                {"label": "Ground<br>Truth", "datapoints": data["y_test"]},
            ]
        }
        png_base64 = get_box_plot(bar_plot_data)
        return div(
            img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
        )

    @staticmethod
    def get_mp_error_histogram_plot(data):
        fig = go.Figure(data=[go.Histogram(y=data["y_error"], nbinsy=10)])
        fig.update_layout(
            paper_bgcolor="rgb(255, 255, 255)",
            plot_bgcolor="rgba(218, 227, 243, 1)",
            #     margin=dict(l=120, r=10, t=140, b=80),
            margin=dict(l=25, r=25, t=25, b=25),
            width=700,
            height=10 * 70 + 30,
            #     annotations=annotations,
            bargap=0.07,
            xaxis_title_text="Counts",
            yaxis_title_text="Residuals",
        )

        png = pio.to_image(fig)
        png_base64 = base64.b64encode(png).decode("ascii")
        return div(
            img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
        )

    @staticmethod
    def get_model_performance_page(data):
        left_metric_elems = []
        for m in data["metrics"]:
            left_metric_elems.append(h3(RegressionComponents.metric_name_lookup[m]))
            left_metric_elems.append(
                p(
                    RegressionComponents.get_metric_explanation_text(
                        m, data["metrics"][m]
                    )
                )
            )

        left_container = div(left_metric_elems, _class="left")

        main_distributions = div(
            h3("Distributions"),
            RegressionComponents.get_distributions_plot(data),
            _class="nobreak_div",
        )
        main_histogram = div(
            h3(
                "Histogram of your model errors "
                "(distance between a predicted value and the observed actual value)"
            ),
            RegressionComponents.get_mp_error_histogram_plot(data),
            _class="nobreak_div",
        )

        main_container = div(main_distributions, main_histogram, _class="main")
        return str(
            div(
                get_page_divider("Model Performance"),
                left_container,
                main_container,
                _class="container",
            )
        )

    @staticmethod
    def get_cohorts_page(data, metric_config):
        return get_cohorts_page(data, metric_config)

    @staticmethod
    def get_feature_importance_page(data):
        return get_feature_importance_page(data)

    @staticmethod
    def get_causal_page(data):
        return get_causal_page(data)

    @staticmethod
    def get_fairlearn_page(data):
        left_elems = [
            div(
                p(
                    "Understand your model’s fairness issues "
                    "using group-fairness metrics across sensitive features and cohorts. "
                    "Pay particular attention to the subgroups who receive worse treatments "
                    "(predictions) by your model."
                )
            )
        ]

        for f in data:
            left_elems.append(h3('Feature "{}"'.format(f)))
            metric_section = []
            for metric_key, metric_details in data[f]["metrics"].items():
                left_elems.append(
                    p(
                        '"{}" has the highest {}: {}'.format(
                            metric_details["group_max"][0],
                            metric_key,
                            round(metric_details["group_max"][1], 2),
                        )
                    )
                )
                left_elems.append(
                    p(
                        '"{}" has the lowest {}: {}'.format(
                            metric_details["group_min"][0],
                            metric_key,
                            round(metric_details["group_min"][1], 2),
                        )
                    )
                )
                if metric_details["kind"] == "difference":
                    metric_section.append(
                        p(
                            "Maximum difference in {} is {}".format(
                                metric_key,
                                round(
                                    metric_details["group_max"][1]
                                    - metric_details["group_min"][1],
                                    2,
                                ),
                            )
                        )
                    )
                elif metric_details["kind"] == "ratio":
                    metric_section.append(
                        p(
                            "Minimum ratio of {} is {}".format(
                                metric_key,
                                round(
                                    metric_details["group_min"][1]
                                    / metric_details["group_max"][1],
                                    2,
                                ),
                            )
                        )
                    )
            left_elems.append(div(metric_section, _class="nobreak_div"))

        left_container = div(left_elems, _class="left")

        def get_fairness_box_plot(data):
            box_plot_data = {"data": []}
            for c in data:
                box_plot_data["data"].append(
                    {
                        "label": c
                        + "<br>"
                        + str(int(100 * data[c]["population"]))
                        + "%",
                        "datapoints": data[c]["y_pred"],
                    }
                )

            png_base64 = get_box_plot(box_plot_data)

            return div(
                img(_src="data:image/png;base64,{}".format(png_base64)),
                _class="image_div",
            )

        def get_table_row(heading, data):
            table_row_elems = []
            table_row_elems.append(th(heading, _class="header_cell"))
            for v in data:
                table_row_elems.append(td(v, _class="cell"))
            return tr(table_row_elems, _class="row")

        def get_table(data):
            from statistics import mean

            metric_list = [d for d in data["metrics"]]
            # metric_list = [m for sublist in metric_list for m in sublist]

            horizontal_headings = [
                "Average<br>Prediction",
                "Average<br>Groundtruth",
            ] + [d.replace("_", "<br>") for d in metric_list]
            vertical_headings = list(data["statistics"].keys())

            headings_td = [td(_class="header_cell")] + [
                td(x, _class="header_cell") for x in horizontal_headings
            ]
            headings = thead(tr(headings_td, _class="row"), _class="table-head")

            rows_elems = []
            for vh in vertical_headings:
                row_data = [
                    round(mean(data["statistics"][vh]["y_pred"]), 2),
                    round(mean(data["statistics"][vh]["y_test"]), 2),
                ]
                for m in metric_list:
                    row_data.append(round(data["metrics"][m]["group_metric"][vh], 2))
                rows_elems.append(get_table_row(vh, row_data))

            body = tbody(rows_elems, _class="table-body")

            return table(headings, body, _class="table")

        main_elems = []
        # prediction distribution
        for f in data:
            main_elems.append(
                div(
                    h2('Feature "{}"'.format(f)),
                    h3("Prediction distribution chart"),
                    get_fairness_box_plot(data[f]["statistics"]),
                    _class="nobreak_div",
                )
            )

            main_elems.append(
                div(
                    h3("Analysis across cohorts"),
                    get_table(data[f]),
                    _class="nobreak_div",
                )
            )

        main_container = div(main_elems, _class="main")

        return str(
            div(
                get_page_divider("Fairness"),
                left_container,
                main_container,
                _class="container nobreak_div",
            )
        )


def to_pdf(html, output, wkhtmltopdf_path=None):
    import pdfkit

    options = {
        "page-size": "Letter",
        "margin-top": "0.75in",
        "margin-right": "0.75in",
        "margin-bottom": "0.75in",
        "margin-left": "0.75in",
        "encoding": "UTF-8",
    }

    wkhtmlconfig = None
    if wkhtmltopdf_path:
        wkhtmlconfig = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

    pdfkit.from_string(
        html, output_path=output, options=options, configuration=wkhtmlconfig
    )
