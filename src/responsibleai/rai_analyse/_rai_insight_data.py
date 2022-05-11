from responsibleai import RAIInsights
import json

import os
import pandas as pd
import sklearn.metrics as skm

# from erroranalysis._internal.constants import Metrics
from erroranalysis._internal.metrics import metric_to_func
from fairlearn.metrics import selection_rate

from datetime import datetime
# from raiwidgets.cohort import ClassificationOutcomes, Cohort, CohortFilter, CohortFilterMethods

from collections import OrderedDict


def false_positive(y_test, y_pred):
    tn, fp, fn, tp = skm.confusion_matrix(y_test, y_pred).ravel()
    return fp


def false_negative(y_test, y_pred):
    tn, fp, fn, tp = skm.confusion_matrix(y_test, y_pred).ravel()
    return fn


metric_func_map = metric_to_func
metric_func_map["error_rate"] = skm.zero_one_loss
metric_func_map["selection_rate"] = selection_rate
metric_func_map["confusion_matrix"] = skm.confusion_matrix
metric_func_map["false_positive"] = false_positive
metric_func_map["false_negative"] = false_negative


class ExplainerFiles:
    GLOBAL_IMPORTANCE = "global_importance_values.json"
    LOCAL_IMPORTANCE = "local_importance_values.json"
    FEATURES = "features.json"
    CLASSES = "classes.json"


def get_metric(metric, y_pred, y_test):
    return metric_func_map[metric](y_pred, y_test)


class RaiInsightData:
    def __init__(self, raiinsight_path):
        os_path = os.path.join(*os.path.split(raiinsight_path))

        self.raiinsight = RAIInsights.load(os_path)
        self.raiinsight_path = os_path

        self.components = ["causal", "counterfactual", "error_analysis", "explainer"]
        self.component_path_prefix = {}
        self.json_paths = {}
        self._set_component_paths_prefix()
        self._set_json_paths()

        self.y_pred = self.raiinsight.model.predict(self.raiinsight.test.drop(columns=self.raiinsight.target_column))

    def _set_component_paths_prefix(self):
        for c in self.components:
            first_guid = next(iter(os.listdir(os.path.join(self.raiinsight_path, c))), None)
            if first_guid:
                self.component_path_prefix[c] = os.path.join(self.raiinsight_path, c,
                                                             first_guid,
                                                             "data")
                if c == "explainer":
                    self.component_path_prefix[c] = os.path.join(self.component_path_prefix[c], "explainer")

        self.component_path_prefix["predictions"] = os.path.join(self.raiinsight_path, "predictions")

    def _set_json_paths(self):
        for c, path_prefix in self.component_path_prefix.items():
            files = os.listdir(path_prefix)
            self.json_paths[c] = {v: os.path.join(path_prefix, v) for v in files}

    def get_json_data(self, component, file):
        if file not in self.json_paths[component]:
            return None

        with open(self.json_paths[component][file], 'r') as json_file:
            data = json.load(json_file)

        return data

    def get_raiinsight(self):
        return self.raiinsight

    def filter_from_cohort(self, cohort):
        from erroranalysis._internal.cohort_filter import filter_from_cohort

        analyzer = self.raiinsight.error_analysis._analyzer
        return filter_from_cohort(analyzer,
                                  filters=cohort,
                                  composite_filters=[])

    def get_filtered_dataset(self, cohort):
        filtered_dataset = self.filter_from_cohort(cohort)
        model = self.raiinsight.error_analysis._analyzer.model
        features = self.raiinsight.error_analysis._analyzer.feature_names
        return {
            "filtered_dataset": filtered_dataset,
            "y_pred": model.predict(filtered_dataset[features]),
            "y_test": filtered_dataset["true_y"],
            "filter_conditions": [c["column"] + c["method"] + str(c["arg"]) for c in cohort]
        }

    def get_y_pred(self):
        return self.y_pred

    def get_y_test(self):
        return self.raiinsight.test[self.raiinsight.target_column].to_numpy()

    def get_test(self):
        return self.raiinsight.test

    def get_fairlearn_grouped_metric(self, sensitive_feature, metric):
        from fairlearn.metrics import MetricFrame
        grouped_metric = MetricFrame(metrics=metric_func_map[metric],
                                     y_true=self.get_y_test(),
                                     y_pred=self.get_y_pred(),
                                     sensitive_features=self.raiinsight.test[sensitive_feature].to_numpy())

        return grouped_metric

    def get_error_analysis_data(self, target_metric):
        # recompute error analysis based on target metric if requested metric does not match existing metric
        if self.raiinsight.error_analysis._analyzer._metric != target_metric:
            self.raiinsight.error_analysis._analyzer._metric = target_metric
            for c in self.raiinsight.error_analysis._ea_config_list:
                c.is_computed = False
            self.raiinsight.error_analysis._ea_report_list = []
            self.raiinsight.error_analysis.compute()

        return self.raiinsight.error_analysis.get_data()

    def get_causal_data(self):
        def visit(node, parents, leaves_collection):
            import copy
            if node.leaf:
                leaf = {
                    "n_samples": node.n_samples,
                    "treatment": node.treatment,
                    "parents": parents
                }

                leaves_collection.append(leaf)

            else:
                this_node = {
                    "feature": node.feature,
                    "right_comparison": node.right_comparison,
                    "comparison_value": node.comparison_value
                }

                # dictionary assignment is by reference. Need deepcopy for by value assignment
                node_left = copy.deepcopy(this_node)
                node_right = copy.deepcopy(this_node)

                node_left["path"] = "left"
                node_right["path"] = "right"
                pat_left = parents + [node_left]
                pat_right = parents + [node_right]

                visit(node.left, pat_left, leaves_collection)
                visit(node.right, pat_right, leaves_collection)

            return leaves_collection

        ca = self.raiinsight.causal.get_data()[0]

        return {
            "global_effect": {k["feature"]: k for k in ca.global_effects},
            "policy_treatments": {k.treatment_feature: visit(k.policy_tree, [], []) for k in ca.policies},
            "top_local_policies": {k.treatment_feature: k.local_policies[:3] for k in ca.policies}
        }

    def to_tree_map(self, tree):
        return {x["id"]: x for x in tree}

    def get_filter_conditions(self, tree_map, nodeid):
        filter_conditions = []
        current_index = nodeid
        while current_index in tree_map.keys():
            current_node = tree_map[current_index]
            if current_node["method"]:
                filter_conditions.append(current_node["condition"])
            current_index = current_node["parentId"]
        return filter_conditions

    def get_min_max_nodes(self, treemap, n):
        node_list = [{"id": v["id"], "metricValue": v["metricValue"]} for k, v in treemap.items()]
        sorted_nodes = sorted(node_list, key=lambda d: d['metricValue'])
        node_size = len(sorted_nodes)
        if node_size < 2*n:
            mid_point = int(node_size/2)
            return sorted_nodes[0:mid_point], sorted_nodes[mid_point:node_size]
        return sorted_nodes[:n], sorted_nodes[-n:]

    def get_feature_statistics(self, feature, dataset=None):
        if dataset is None:
            dataset = self.raiinsight.test
        return dataset[feature].value_counts()

    def get_cohort_data(self, filtermap):
        filtered_dataset = self.raiinsight.test[filtermap]
        filtered_y_pred = self.get_y_pred()[filtermap]
        filtered_y_test = self.get_y_test()[filtermap]
        return {
            "y_pred": filtered_y_pred,
            "y_test": filtered_y_test,
            "population": len(filtered_dataset) / len(self.raiinsight.test)
        }


class PdfDataGen:
    def __init__(self, raiinsightdata, config):
        self.data = raiinsightdata
        self.config = config

        self.primary_metric = next(iter(self.config["Metrics"].keys()), None)
        self.metrics = self.config["Metrics"].keys()
        self.tasktype = "regression" if self.config["Model"]["ModelType"] == "Regression" else "classification"

    def get_model_overview_data(self):
        return_data = self.config["Model"]

        return_data["metrics_targets"] = self.get_metrics_targets()
        return_data["runinfo"] = None
        return_data["y_test"] = self.data.get_y_test()

        if "runinfo" in self.config.keys():
            return_data["runinfo"] = self.config["runinfo"]
            parsed_datetime = datetime.strptime(return_data["runinfo"]["startTimeUtc"],
                                                '%Y-%m-%dT%H:%M:%S.%fZ')
            return_data["runinfo"]["startTimeUtc"] = parsed_datetime.strftime("%m/%d/%Y")

        if self.data.raiinsight._classes:
            return_data["classes"] = self.data.raiinsight._classes

        return return_data

    def get_metrics_targets(self):
        metric_targets = []
        for k, v in self.config['Metrics'].items():
            if 'threshold' in v.keys():
                metric_targets.append("{}: {} {}".format(k, v["threshold"][0], v["threshold"][1]))
            else:
                metric_targets.append("{}".format(k))

        if 'FeatureImportance' in self.config.keys():
            metric_targets.append("Top important features: {}".format(self.config["FeatureImportance"]["top_n"]))

        if 'Fairness' in self.config.keys():
            fc = self.config["Fairness"]
            for m in fc["metric"]:
                if 'threshold' in fc.keys():
                    metric_targets.append("Fairness {} in {}: {}".format(fc["fairness_evaluation_kind"],
                                                                         m,
                                                                         fc["threshold"]))
                else:
                    metric_targets.append("Fairness {} in {}".format(fc["fairness_evaluation_kind"],
                                                                     m))

        return metric_targets

    def get_binning_information(self, target_feature):
        dataset = self.data.get_test()
        if target_feature in self.data.get_raiinsight().categorical_features:

            def relabel(item, topnlabels):
                if item in topnlabels:
                    return item
                return 'other'

            total_labels = dataset[target_feature].nunique()
            topnlabels = dataset[target_feature].value_counts().nlargest(3).index

            if len(topnlabels) >= total_labels:
                all_labels = topnlabels.to_list()
            else:
                all_labels = topnlabels.to_list() + ["other"]
            return (
                all_labels,
                dataset[target_feature].apply(relabel, args=(topnlabels,)))
        else:
            df = pd.DataFrame()

            df['label'] = pd.qcut(dataset[target_feature], 4, duplicates="drop")
            df['label'] = df['label'].apply(lambda x: "{} to {}".format(round(x.left, 2), round(x.right, 2)))
            return (df['label'].value_counts().index.to_list(),
                    df['label'])

    def _get_categorical_feature_data(self, feature, metric):
        dataset = self.data.get_test()
        distribution = dataset[feature].value_counts()
        total = len(dataset[feature])

    def get_data_explorer_data(self):
        de_data = []
        y_test = self.data.get_y_test()
        y_predict = self.data.get_y_pred()
        for f in self.config["DataExplorer"]["features"]:
            label_list, new_labels = self.get_binning_information(f)
            counts = new_labels.value_counts()
            total = len(new_labels)
            primary_metric = self.primary_metric
            metric_func = None
            if primary_metric:
                metric_func = metric_func_map[primary_metric]

            data = {
                "feature_name": f,
                "primary_metric": primary_metric,
                "data": []
            }

            for label in label_list:
                index_filter = [True if x == label else False for x in new_labels]

                f_data = {
                    'label': label,
                    'population': counts[label]/total,
                    'prediction': y_predict[index_filter]
                }

                if metric_func:
                    f_data[primary_metric] = metric_func(y_predict[index_filter],
                                                         y_test[index_filter])

                data["data"].append(f_data)

            de_data.append(data)

        return de_data

    def get_feature_importance_data(self):
        importances = self.data.get_json_data("explainer", "global_importance_values.json")["data"]
        features = self.data.get_json_data("explainer", "features.json")["data"]

        short_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        top_n = self.config["FeatureImportance"]["top_n"]

        sorted_tuple = sorted(
            [(f, importances[index]) for index, f in enumerate(features)], key=lambda x: x[1]
        )[-top_n:]
        sorted_dict = {
            t[0]: {'value': t[1], 'short_label': short_labels[index]}
            for index, t in enumerate(sorted_tuple)
        }

        return OrderedDict(sorted_dict)

    def get_causal_data(self):
        return self.data.get_causal_data()

    def get_fairlearn_data(self):
        fair_con = self.config["Fairness"]
        fm = {}

        for f in fair_con["sensitive_features"]:
            fm[f] = {}
            fm[f]["metrics"] = {}
            fm[f]["statistics"] = {}
            for m in fair_con["metric"]:
                gm = self.data.get_fairlearn_grouped_metric(f, m)

                fm_lookup = {
                    "difference": gm.difference(method='between_groups'),
                    "ratio": gm.ratio()
                }

                sorted_group_metric = sorted(gm.by_group.to_dict().items(), key=lambda x: x[1])

                fm[f]["metrics"][m] = {
                    "kind": fair_con["fairness_evaluation_kind"],
                    "value": fm_lookup[fair_con["fairness_evaluation_kind"]],
                    "group_metric": OrderedDict(sorted_group_metric),
                    "group_max": next(iter(sorted_group_metric[-1:]), None),
                    "group_min": next(iter(sorted_group_metric[:1]), None)
                }

            feature_statistics = dict(self.data.get_feature_statistics(f))
            for k, v in feature_statistics.items():
                filtermap = self.data.get_test()[f] == k
                fm[f]["statistics"][k] = self.data.get_cohort_data(filtermap)

        return fm

    def get_model_performance_data(self):
        y_pred = self.data.get_y_pred()
        y_test = self.data.get_y_test()
        return_data = {
            "y_pred": y_pred,
            "y_test": y_test,
            "metrics": {}
        }

        if self.tasktype == "regression":
            return_data["y_error"] = list(map(lambda x, y: x-y, y_pred, y_test))

        if self.tasktype == "classification":
            tn, fp, fn, tp = metric_func_map["confusion_matrix"](y_pred, y_test).ravel()
            return_data["confusion_matrix"] = {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp
            }

            if self.data.raiinsight._classes:
                return_data["classes"] = self.data.raiinsight._classes

        for m in self.config["Metrics"]:
            return_data["metrics"][m] = metric_func_map[m](y_pred, y_test)

        return return_data

    def get_cohorts_data(self):
        cohorts_data = {
            "error_analysis_min": [],
            "error_analysis_max": [],
            "cohorts": []
        }
        short_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for m in self.config["Metrics"]:
            if "Cohorts" in self.config:
                for index, c in enumerate(self.config["Cohorts"]):
                    if c in self.config["cohorts_definition"].keys():
                        code = self.config["cohorts_definition"][c]
                        filtered_dataset = self.data.get_filtered_dataset(code)

                        cd = {
                            'label': c,
                            'short_label': short_labels[index],
                            m: metric_func_map[m](filtered_dataset["y_pred"], filtered_dataset["y_test"]),
                            'population': len(filtered_dataset["y_pred"]) / len(self.data.get_y_test())
                        }
                        if "threshold" in self.config["Metrics"][m]:
                            cd["threshold"] = self.config["Metrics"][m]["threshold"][1]

                        cohorts_data["cohorts"].append(cd)

            if "error_analysis" in self.data.component_path_prefix:
                ea_data = self.data.get_error_analysis_data(m)[0]

                tree = ea_data.tree
                treemap = self.data.to_tree_map(tree)
                min_nodes, max_nodes = self.data.get_min_max_nodes(treemap, 3)

                def get_cohorts_data(nodes):
                    ret = []
                    for index, node in enumerate(nodes):
                        filter_conditions = self.data.get_filter_conditions(treemap, node["id"])
                        cd = {
                            'label': " <br>AND<br>".join(filter_conditions) if len(filter_conditions) > 0 else "All Data",
                            'short_label': short_labels[index],
                            m: treemap[node["id"]]["metricValue"],
                            'population': treemap[node["id"]]['size'] / len(self.data.get_y_test())
                        }
                        if "threshold" in self.config["Metrics"][m]:
                            cd["threshold"] = self.config["Metrics"][m]["threshold"][1]

                        ret.append(cd)
                    return ret
                cohorts_data["error_analysis_min"].extend(get_cohorts_data(min_nodes))
                cohorts_data["error_analysis_max"].extend(get_cohorts_data(max_nodes))

        return cohorts_data



