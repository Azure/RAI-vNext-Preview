from domonic.html import *


def get_wrapped_div(div_class, items, style = None):
    if style:
        style = 'style="{}"'.format(style)
    else:
        style = ''
    return '<div class="{}"{}>'.format(div_class, style) + "".join(items) + '</div>'

def get_html_elm(html_attri, content, class_name = None, style = None):
    if style:
        style = ' style="{}"'.format(style)
    else:
        style = ''
    
    if class_name:
        class_name = ' class="{}"'.format(class_name)
    else:
        class_name = ''
        
    return '<{html_attri}{class_name}{style}>{content}</{html_attri}>'.format(html_attri=html_attri,
                                                                              class_name=class_name,
                                                                              style=style,
                                                                              content=content)

def get_full_html(body):
    return '<!DOCTYPE html><html><body><head>{}</head>{}</body></html>'.format(get_css(), body)

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
        
        .image_div {
          break-inside: avoid;
          /*border: 2px solid black;*/
          /*border-radius: 5px;*/
          /*width: 99%;*/
        }

      </style>"""

def get_model_overview(model_type, model_name, model_purpose, metrics_targets, classes, y_test):
    model_left_items = []
    
    model_left_items.append('<h2>Model Summary</h2>')
    model_left_items.append('<div><h3>Purpose</h3><p>{}</p></div>'.format(model_purpose))
    if model_type == "binary_classification":
        model_left_items.append('<div><p>Binary classification: {} vs {}</p></div>'.format(classes[0], classes[1]))
    else:
        model_left_items.append('<div><p>This is {} model </p></div>'.format(model_type))
    model_left_items.append('<div><h3>How the model is evaluated</h3><p>This model is evaluated on a test set with {} datapoints.</div>'.format(len(y_test)))
    
    model_overview_left_container = get_wrapped_div("left", model_left_items)
    
    model_main_items = []
    model_main_items.append('<h3>Target values</h3><p>Here are your defined target minimum/maximum performance levels and/or target performance differences between groups/cohorts:</p>')
    
    metric_targets_elems = []
    for (k,v) in metrics_targets.items():
        metric_targets_elems.append('<li>{} target: {}</li>'.format(k, v))
        
    model_main_items.append(get_html_elm("div", get_html_elm("ul", "".join(metric_targets_elems)), style="border: 2px solid black; border-radius: 5px;"))
    
    model_overview_main_container = get_wrapped_div("main", model_main_items)
    
    model_overview_container = get_wrapped_div("container", [model_overview_left_container, model_overview_main_container])
    
    return '<div class="header"><h1>{}</h1></div>'.format(model_name) + model_overview_container

def get_page_divider(text):
    return """
    <div style="height:36px; margin-top: 10px">
      <div style="width: 100%; height: 12px; border-bottom: 1px solid #605E5C;">
        <span style="font-size: 14px; background-color: #605E5C; padding: 0 10px; border-radius: 18px; color: #ffffff">
          {}
        </span>
      </div>
    </div>
    """.format(text)

def get_data_explorer_page(data):
    de_left_items = ["<div><p>Evaluate your dataset to assess represenation of identified subgroups</p></div>"]
    de_main_items = []
    for d in data:
        de_elements = []
        de_elements.append("<h3>{}</h3>".format(d["feature_name"]))
        for c in d["classes"]:
            de_elements.append("<p>{}% of [{}] data points have the correct prediction (accuracy)</p>".format(c["accuracy"]*100, c["label"]))
        de_elements.append(get_de_image(d))
        de_main_items.append(get_html_elm("div", "".join(de_elements), "image_div"))
        
    return get_wrapped_div("container", [get_wrapped_div("left", de_left_items), get_wrapped_div("main", de_main_items)])

def get_de_image(data):
    png_base64 = get_de_bar_plot(data)
    return str(div(img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"))

def get_cohorts_performance_image(data, m):
    png_base64 = get_cp_bar_plot(data, m)
    return div(img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div")

def get_fi_image(data):
    png_base64 = get_fi_bar_plot(data)
    return div(img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div")

def get_cohorts_performance_container(data, m):
    cohort_left_heading = h3("My cohorts: {}".format(m))
    cohorts_list = ul()
    for c in data["cohorts"]:
        cohorts_list.append(li("{}: {}".format(c["cohort_short_name"], c["cohort_full_name"])))
        
    cp_left_container = div(cohort_left_heading, cohorts_list, _class="left")
    
    cp_main_heading = h3("My cohorts: {}".format(m))
    cp_main_image = get_cohorts_performance_image(data, m)
    
    cp_main_container = div(cp_main_heading, cp_main_image, _class="main")
    
    return str(div(cp_left_container, cp_main_container, _class="container"))
    

def get_cohorts_page(data):
    cp_heading_left = div(p("Observe evidence of model accuracy and performance across your passed cohorts:"), _class="left")
    cp_heading_main = div(_class="main")
    heading_section = str(div(cp_heading_left, cp_heading_main, _class="container"))
    
    # start section for each predefined cohort
    cohort_performance_containers = ""
    for m in [x for x in global_config["Metrics"]]:
        cohort_performance_containers = cohort_performance_containers + get_cohorts_performance_container(data, m)
    
    return heading_section + cohort_performance_containers

def get_feature_importance_page(data):
    heading_left = div(p("Understand factors that have impacted your model predictions the most. "
                         "These are factors that may account for performance levels and differences."),
                      _class = "left")
    heading_main = div(get_fi_image(data), _class="main")
    return str(div(heading_left, heading_main, _class="container"))

import plotly.graph_objects as go

global_config = {"Metrics":
                 {"accuracy": {"threshold": 0.75},
                  "precision": {"threshold": 0.65}}
                }

def get_fi_bar_plot(data):
    y_data = [y for y in data["features"]]
    x_data = [data["features"][x]["importance"] for x in data["features"]]
    x_data = [[x, 1-x] for x in x_data]
    tickvals = [0.0, 0.25, 0.5, 0.75, 1.0]
    ticktext = [0.0, 0.25, 0.5, 0.75, 1.0]
    tickappend = ""
    
    return get_bar_plot(y_data, x_data, tickvals=tickvals, ticktext=ticktext, tickappend=tickappend)
    

def get_cp_bar_plot(data, m):
    metric_name = m
    y_data = [y["cohort_short_name"] for y in data["cohorts"]]
    x_data = [int(x[metric_name]*100) for x in data["cohorts"]]
    x_data = [[x, 100-x] for x in x_data]
    legend = [m]
    
    return get_bar_plot(y_data, x_data, legend=legend, threshold=global_config["Metrics"][m]["threshold"]*100)

def get_de_bar_plot(data):
    y_data = [y["label"] + "<br>" + str(int(y["population"]*100))+"% pop." for y in data["classes"]]
    x_data = [int(float(x["prediction_0_ratio"])*100) for x in data["classes"]]
    x_data = [[x, 100-x] for x in x_data]
    legend = ["Predicted as \"" + y["prediction_0_name"] + "\"" for y in data["classes"]]

    return get_bar_plot(y_data, x_data, legend=legend)

def get_bar_plot(y_data, x_data, legend=None, threshold=None, tickvals=None, ticktext=None, tickappend="%"):
    import plotly.graph_objects as go
    fig = go.Figure()
    series = 0
    
    def get_colors(series, index):
        colors2 = ['rgba(39, 110, 237, 1)',
          'rgba(218, 227, 243, 1)']
        return colors2[index%2]

    def get_show_legend(legend, index):
        if not legend:
            return False
        if index == 0:
            return True
        return False

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                width=0.9,
                orientation='h',
                marker=dict(
                    color=get_colors(series, i),
                    line=dict(color='rgb(248, 248, 249)', width=1)
                ),
                showlegend=get_show_legend(legend, series)
            ))
            series = series + 1

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False,
            domain=[0.15, 1],
            tickvals= tickvals if tickvals else [0, 25, 50, 75, 100],
            ticktext = ticktext if ticktext else [str(x)+tickappend for x in [0, 25, 50, 75, 100]]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        bargap=0,
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 2555, 255)',
    #     margin=dict(l=120, r=10, t=140, b=80),
        margin = dict(l=25, r=25, t=25, b=25),
    #     showlegend=False
    )

    annotations = []

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(dict(xref='paper', yref='y',
                                x=0.14, y=yd,
                                xanchor='right',
                                text=str(yd),
                                font=dict(family='Consolas', size=18,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        # labeling the first percentage of each bar (x_axis)
        annotations.append(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text=str(xd[0]) + tickappend,
                                font=dict(family='Consolas', size=22,
                                          color='rgb(248, 248, 255)'),
                                showarrow=False))

#         space = xd[0]
#         for i in range(1, len(xd)):
#                 # labeling the rest of percentages for each bar (x_axis)
#                 annotations.append(dict(xref='x', yref='y',
#                                         x=space + (xd[i]/2), y=yd,
#                                         text=str(xd[i]) + tickappend,
#                                         font=dict(family='Consolas', size=22,
#                                                   color='rgb(248, 248, 255)'),
#                                         showarrow=False))
#                 space += xd[i]

    fig.update_layout(
        annotations=annotations,
        #autosize=True,
        width=700,
        height=len(x_data)*70+30,
    )
    
    def get_legend_y(barchart_length):
        lookup = {1: 1.55,
                 2: 1.27,
                 3: 1.15,
                 4: 1.11,
                 5: 1.09}
        return lookup[barchart_length]
        
    
    fig.update_traces(cliponaxis = False)
    if threshold:
        fig.add_vline(x=threshold, annotation_text="Threshold", annotation_position="bottom right", line_width=3, line_dash="dash", line_color="red")
    
    if legend:
        fig['data'][0]['name'] = legend[0]
        fig.update_layout(legend=dict(yanchor="top", y=get_legend_y(len(x_data)), xanchor="left", x=0.15))

    # fig.update_xaxes(automargin=True)
    # fig.update_yaxes(automargin=True)
    
    import plotly.io as pio
    png = pio.to_image(fig)
    import base64
    png_base64 = base64.b64encode(png).decode('ascii')
    
    return png_base64

class test_html:
    def generate_pdf(self, out_name="model.pdf"):
        body = self.generate_html()
        
        full_html = get_full_html(body)
        full_html = full_html.replace("\n", "")
        
        import pdfkit

        options = {
            'page-size': 'Letter',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
        }
        pdfkit.from_string(full_html, output_path=out_name, options=options)
        
    def dump_html(self, out_name="model.html"):
        body = self.generate_html()
        
        full_html = get_full_html(body)
        full_html = full_html.replace("\n", "")
        
        with open(out_name, "w") as file1:
            # Writing data to a file
            file1.write(full_html)
        
    def generate_html(self):
        full_html = []
        model_overview = get_model_overview(model_type="binary_classification",
                  model_name="Loan allocation model",
                  model_purpose="This is a binary classification model designed to predict loan approval and rejection decisions in a bank.",
                  metrics_targets = {"Accuracy": "> 75%", "Precision": "> 60%", "Fairness difference in selection rate": "< 15%", "Top important features": 10},
                  classes = ["approved", "rejected"],
                  y_test = [0]*2000)
        
        de_page_divider = get_page_divider("Data explorer")
        
        data_explorer_data = [{
            "feature_name": "Gender",
            "type": "Categorical",
            "classes": [{"label": "Female", "accuracy": 0.7, "population": 0.3, "prediction_0_ratio": 0.6, "prediction_0_name": "Approved"},
                        {"label": "Male", "accuracy": 0.4, "population": 0.65, "prediction_0_ratio": 0.4, "prediction_0_name": "Approved"},
                        {"label": "Non-binary", "accuracy": 0.2, "population": 0.05, "prediction_0_ratio": 0.7, "prediction_0_name": "Approved"}]
             },
            {
            "feature_name": "Neighbourhood",
            "type": "Categorical",
            "classes": [{"label": "Cambridge", "accuracy": 0.7, "population": 0.25, "prediction_0_ratio": 0.5, "prediction_0_name": "Approved"},
                        {"label": "Boston", "accuracy": 0.3, "population": 0.45, "prediction_0_ratio": 0.5, "prediction_0_name": "Approved"},
                        {"label": "Other", "accuracy": 0.7, "population": 0.3, "prediction_0_ratio": 0.5, "prediction_0_name": "Approved"}]
            },
            {
            "feature_name": "Previous Loan Amount",
            "type": "Continuous",
            "classes": [{"label": "0-20000", "accuracy": 0.7, "population": 0.33, "prediction_0_ratio": 0.5, "prediction_0_name": "Approved"},
                        {"label": "20000-60000", "accuracy": 0.7, "population": 0.33, "prediction_0_ratio": 0.5, "prediction_0_name": "Approved"},
                        {"label": "> 60000", "accuracy": 0.7, "population": 0.33, "prediction_0_ratio": 0.5, "prediction_0_name": "Approved"}]
            },
        ]
        
        de_page = get_data_explorer_page(data_explorer_data)
        
        cohorts_data = {"cohorts": [{"cohort_short_name": "A", "cohort_full_name": "High income applicant", "accuracy": 0.7, "precision":0.7},
                   {"cohort_short_name": "B", "cohort_full_name": "Refferal candidate", "accuracy": 0.7, "precision":0.7}]}

        cp_page = get_cohorts_page(cohorts_data)
        
        fi_data = {"features": {"Gender": {"importance": 0.2},
                    "Education": {"importance": 0.31},
                    "f3": {"importance": 0.31},
                    "s2": {"importance": 0.13},
                    "v2": {"importance": 0.61},
                    "v5": {"importance": 0.84},
                    "x2": {"importance": 0.76}
                   }}
        fi_page = get_feature_importance_page(fi_data)
        
        return "".join([model_overview, de_page_divider,
                        de_page, get_page_divider("Cohorts"),
                        cp_page, get_page_divider("Feature relevance (explainability)"),
                        fi_page, get_page_divider("Fairness Assessment")])

testclass = test_html()
testclass.generate_pdf("rai_pdf.pdf")