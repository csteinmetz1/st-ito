import os
import sys
import glob
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == "__main__":

    modes = ["different"]
    methods = ["clap", "param-panns"]

    plugin_names = [
        "vst_RoughRider3",
        "vst_DragonflyPlateReverb",
        "vst_3BandEQ",
        "vst_MaGigaverb",
        "vst_MetalTone",
        "vst_TAL-Chorus-LX",
        "pb_Chorus",
        "pb_Reverb",
        "pb_Delay",
        "pb_Distortion",
        "pb_Compressor",
        "pb_ParametricEQ",
    ]

    # create a table
    scores = {}
    for method in methods:
        for mode in modes:
            # find case study plugin directories
            plugin_dirs = [
                os.path.join("output", "new_case_study", plugin_name)
                for plugin_name in plugin_names
            ]

            for plugin_idx, plugin_dir in enumerate(plugin_dirs):
                plugin_name = os.path.basename(plugin_dir)

                if plugin_name not in scores:
                    scores[plugin_name] = {}

                print(f"Processing {plugin_name} {mode} {method}")
                # load the results

                # check if file exists
                if not os.path.exists(
                    os.path.join(plugin_dir, "case_study_results.json")
                ):
                    continue

                with open(
                    os.path.join(plugin_dir, "case_study_results.json"), "r"
                ) as f:
                    results = json.load(f)

                if mode not in results:
                    continue

                results = results[mode][method]

                for param_idx, parameter_under_test in enumerate(results.keys()):

                    if parameter_under_test not in scores[plugin_name]:
                        scores[plugin_name][parameter_under_test] = {}

                    # get the list of parameter values
                    parameter_values = list(results[parameter_under_test].keys())

                    # create list of estimated parmaeters and true values
                    true_values = []
                    estimated_values = []
                    fopt_values = []

                    for parameter_value in parameter_values:
                        for estimated_param, fopt in results[parameter_under_test][
                            parameter_value
                        ]:
                            true_values.append(float(parameter_value))
                            estimated_values.append(float(estimated_param))
                            fopt_values.append(-1 * float(fopt))

                    # print(true_values, estimated_values)

                    # compute mean squared error
                    mse = np.mean(
                        (np.array(true_values) - np.array(estimated_values)) ** 2
                    )
                    # print(f"MSE: {mse}")

                    correlation = np.corrcoef(true_values, estimated_values)[0, 1]
                    # print(f"Correlation: {correlation}")
                    # print()

                    if method not in scores[plugin_name][parameter_under_test]:
                        scores[plugin_name][parameter_under_test][method] = {}

                    scores[plugin_name][parameter_under_test][method]["mse"] = mse
                    scores[plugin_name][parameter_under_test][method][
                        "correlation"
                    ] = correlation

    # print the table
    table = ""

    for plugin_name, plugin_scores in scores.items():
        print(plugin_name, plugin_scores)

        if len(list(plugin_scores.keys())) == 0:
            continue

        parameter_under_test = list(plugin_scores.keys())[0]
        parameter_under_test_name = parameter_under_test.replace("_", "\\_")
        table += f"""{plugin_name.split("_")[-1][:10]} ({parameter_under_test_name[:6]}) & """
        for metric in ["mse", "correlation"]:
            clap_score = plugin_scores[parameter_under_test]["clap"][metric]
            param_score = plugin_scores[parameter_under_test]["param-panns"][metric]

            if (clap_score > param_score and metric == "correlation") or (
                param_score > clap_score and metric == "mse"
            ):
                table += (
                    "\\textbf{" + f"{clap_score:0.3f}" + "} " + f"& {param_score:0.3f} "
                )
            else:
                table += (
                    f"{clap_score:0.3f} &" + "\\textbf{" + f"{param_score:0.3f}" + "} "
                )

            if metric == "mse":
                table += f"""& """
        table += "\\\\ \n"

    print()
    print(table)
    # create a plot
    sys.exit()

    fig, axs = plt.subplots(
        figsize=(10.0, 3.5),
        ncols=6,
        nrows=2,
        sharex=False,
        sharey=False,
    )

    method_colors = {
        "clap": "#429e79",
        "param-panns": "#405982",
    }

    full_method_names = {
        "clap": "CLAP",
        "param-panns": "Param-Enc",
    }

    axs = np.reshape(axs, -1)

    for method in methods:
        for mode in modes:
            # find case study plugin directories
            plugin_dirs = glob.glob(os.path.join("output/case_study", "*"))
            plugin_dirs = [d for d in plugin_dirs if os.path.isdir(d)]

            # list of five different shades of light colors
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            for plugin_idx, plugin_dir in enumerate(plugin_dirs):
                plugin_name = os.path.basename(plugin_dir)
                print(f"Processing {plugin_name}")
                # load the results

                # check if file exists
                if not os.path.exists(
                    os.path.join(plugin_dir, "case_study_results.json")
                ):
                    continue

                with open(
                    os.path.join(plugin_dir, "case_study_results.json"), "r"
                ) as f:
                    results = json.load(f)

                if mode not in results:
                    continue

                results = results[mode][method]

                for param_idx, parameter_under_test in enumerate(results.keys()):

                    # get the list of parameter values
                    parameter_values = list(results[parameter_under_test].keys())

                    # create list of estimated parmaeters and true values
                    true_values = []
                    estimated_values = []
                    fopt_values = []

                    for parameter_value in parameter_values:
                        for estimated_param, fopt in results[parameter_under_test][
                            parameter_value
                        ]:
                            true_values.append(float(parameter_value))
                            estimated_values.append(float(estimated_param))
                            fopt_values.append(-1 * float(fopt))

                    print(true_values, estimated_values)

                    # if mode == "same":  # blue
                    #    color = ("#405982",)
                    # elif mode == "different":  # green
                    #    color = "#4a7d4a"

                    # create a colormap and then pick a color from it
                    cmap = matplotlib.colormaps["plasma"]
                    # create an array of colors based on fopt values

                    # map fopt between 0.25 and 0.75
                    fopt_values = np.array(fopt_values)
                    fopt_values = (fopt_values - min(fopt_values)) / (
                        max(fopt_values) - min(fopt_values)
                    )
                    fopt_values = 0.25 + (0.75 - 0.25) * fopt_values
                    color = cmap(fopt_values)

                    if method == "param-panns":
                        plugin_idx += 6

                    axs[plugin_idx].scatter(
                        true_values,
                        estimated_values,
                        zorder=3,
                        s=20,
                        alpha=0.9,
                        color=method_colors[method],
                        edgecolor="none",
                    )
                    if plugin_idx == 0 or plugin_idx == 6:
                        axs[plugin_idx].set_ylabel(
                            f"{full_method_names[method]}", rotation=0, labelpad=20
                        )

                    if plugin_idx >= 6:
                        axs[plugin_idx].set_xlabel(parameter_under_test)

                    if method == "clap":
                        axs[plugin_idx].set_title(f"{plugin_name}", fontsize=10)

                    # turn off spines
                    # axs[plugin_idx].spines["top"].set_visible(False)
                    # axs[plugin_idx].spines["right"].set_visible(False)
                    # axs[plugin_idx].spines["left"].set_visible(False)
                    # axs[plugin_idx].spines["bottom"].set_visible(False)

                    # adjust vertical space between subplots

                    # compute the correlation coefficient
                    correlation = np.corrcoef(true_values, estimated_values)[0, 1]
                    # add the correlation coefficient to the plot
                    if True:
                        axs[plugin_idx].text(
                            0.1,
                            0.90,
                            f"r = {correlation:0.2f}",
                            fontsize=8,
                            horizontalalignment="left",
                            verticalalignment="top",
                            transform=axs[plugin_idx].transAxes,
                            bbox=dict(facecolor="white", edgecolor="none"),
                        )

                    min_value = min(min(true_values), min(estimated_values))
                    max_value = max(max(true_values), max(estimated_values))
                    value_range = max_value - min_value

                    min_point = min_value - (0.2 * value_range)
                    max_point = max_value + (0.2 * value_range)

                    # plot the line y = x
                    axs[plugin_idx].plot(
                        [min_point, max_point],
                        [min_point, max_point],
                        c="darkgray",
                        linestyle="--",
                        zorder=0,
                    )

                    # make the plot square
                    axs[plugin_idx].set_aspect("equal", adjustable="box")

                    axs[plugin_idx].xaxis.set_minor_locator(
                        matplotlib.ticker.AutoMinorLocator(2)
                    )
                    axs[plugin_idx].yaxis.set_minor_locator(
                        matplotlib.ticker.AutoMinorLocator(2)
                    )
                    # axs[plugin_idx].yaxis.get_ticklocs(minor=True)
                    # axs[plugin_idx].minorticks_on()

                    # axs[plugin_idx].xaxis.set_major_formatter(
                    #    ticker.FormatStrFormatter("%0.1f")
                    # )
                    # axs[plugin_idx].yaxis.set_major_formatter(
                    #    ticker.FormatStrFormatter("%0.1f")
                    # )

                    axs[plugin_idx].grid(c="lightgray", which="both")

                # plt.tight_layout()
                plt.subplots_adjust(hspace=0.0, wspace=0.5)
                plt.savefig(f"output/case_study/clap+ours_case_study.png", dpi=300)
                plt.savefig(f"output/case_study/clap+ours_case_study.pdf")
