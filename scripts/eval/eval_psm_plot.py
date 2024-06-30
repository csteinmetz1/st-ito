import os
import json
import argparse
import itertools
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=str)
    parser.add_argument("--human_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--plot_ci", action="store_true")
    args = parser.parse_args()

    with open(args.results_json) as f:
        results = json.load(f)

    effect_names = list(results[list(results.keys())[0]]["multi-effects"].keys())
    model_names = list(results.keys())

    # create a new "effect_name" that contains the results for all effects for each model
    if False:
        new_results = results.copy()
        for model_name in model_names:
            for test_type in ["inter-effect", "intra-effect-easy"]:
                model_results = []
                for effect_name in effect_names:
                    model_results.extend(results[model_name][test_type][effect_name])
                new_results[model_name][test_type]["all"] = model_results

    if args.human_json is not None:
        with open(args.human_json) as f:
            human_results = json.load(f)
        model_names.insert(0, "human")

    # results = new_results

    if args.human_json is not None:
        results["human"] = human_results["human"]
    # update the effect_names to include the "all" effect
    effect_names = list(results[list(results.keys())[0]]["multi-effects"].keys())

    colors = [
        "#306439",
        "#467a49",
        "#5d905a",
        "#74a76b",
        "#8dbe7d",
        "#a6d68f",
        "#3c6d96",
        "#2e5d88",
        "#204d79",
        "#113e6b",
        "#002f5c",
    ]

    name_def = {
        "human": "Human",
        "vggish": "VGGish",
        "wav2clip": "Wav2Clip",
        "wav2vec2": "wav2vec2",
        "clap": "CLAP",
        "fx-encoder": "FX-Encoder",
        "deepafx-st": "DeepAFx-ST",
        "deepafx-st+": "DeepAFx-ST+",
        "mfccs": "MFCCs",
        "mir-features": "MIR Features",
        "param-clap": "Param CLAP",
        "param-dstcn": "Param DSTCN",
        "param-panns-l2-diff": "Param PANNs (L2+Diff)",
        "param-panns-l2-concat": "AFx-Rep (ours)",
        "param-panns-diff": "Param PANNs (Diff)",
        "param-panns-concat": "Param PANNs (Concat)",
        "param-panns-blind": "Param (Blind)",
        "param-panns-l2-concat-adv": "Param-P (adv)",
        "param-panns-l2-concat-adv-32": "Param (Adv)",
        "sim-panns-l2": "Similarity PANNs",
        "beats": "BEATs",
    }

    color_def = {
        "human": "#616363",
        "vggish": "#429e79",
        "wav2clip": "#429e79",
        "wav2vec2": "#429e79",
        "clap": "#429e79",
        "beats": "#429e79",
        "fx-encoder": "#bf6082",  #
        "deepafx-st": "#799bd1",  # light blue
        "deepafx-st+": "#799bd1",  # light blue
        "mfccs": "#cfab48",
        "mir-features": "#cfab48",
        "param-clap": "#405982",
        "param-dstcn": "#405982",
        "param-panns-l2-diff": "#405982",
        "param-panns-l2-concat": "#405982",
        "param-panns-diff": "#405982",
        "param-panns-concat": "#405982",
        "param-panns-blind": "#405982",
        "param-panns-l2-concat-adv": "#405982",
        "param-panns-l2-concat-adv-32": "#405982",
        "sim-panns-l2": "#405982",
    }

    alpha_def = {
        "human": 0.9,
        "vggish": 0.9,
        "wav2clip": 0.8,
        "wav2vec2": 0.7,
        "clap": 0.6,
        "beats": 0.5,
        "fx-encoder": 0.4,
        "deepafx-st": 0.9,
        "deepafx-st+": 0.4,
        "mfccs": 0.9,
        "mir-features": 0.4,
        "param-clap": 0.9,
        "param-panns-l2-concat": 0.4,
        "param-panns-l2-concat-adv": 0.4,
        "param-panns-l2-concat-adv-32": 0.4,
        "sim-panns-l2": 0.4,
    }

    colors = [color_def[model] for model in model_names]

    # create a list of unique linestyles
    mystyles = ["-", "--", "-.", ":"]

    line_def = {
        "param-panns-l2-concat": "-",
        "param-panns-l2-concat-adv": "--",
        "param-panns-l2-concat-adv-32": "-.",
        "param-clap": "--",
        "deepafx-st+": "-",
        "deepafx-st": "--",
        "mfccs": "-",
        "clap": "-",
        "fx-encoder": "-",
        "wav2clip": "--",
        "beats": "--",
    }
    print(model_names)

    # if False:
    model_names = [
        "param-panns-l2-concat",
        # "param-panns-l2-concat-adv-32",
        # "deepafx-st",
        "deepafx-st+",
        "mfccs",
        "clap",
        "fx-encoder",
        # "clap",
        # "wav2clip",
    ]

    overall_results = {}
    # plot results from multi-effect
    test_type = "multi-effects"
    fig, ax = plt.subplots(
        figsize=(4.5, 3.6), ncols=3, nrows=2, sharex=True, sharey=True
    )
    ax = np.reshape(ax, -1)
    plt.minorticks_on()
    for effect_case in np.arange(1, 6):
        subplot_idx = effect_case - 1
        for model_idx, model_name in enumerate(model_names):
            model_results = []
            effect_name = f"effects={effect_case}"
            for class_name in np.arange(1, 25):
                class_name = str(class_name)
                model_case_results = np.array(
                    results[model_name][test_type][effect_name][class_name]
                )
                model_case_results = np.mean(model_case_results) * 100
                model_results.append(model_case_results)

            if model_name not in overall_results:
                overall_results[model_name] = []

            overall_results[model_name].append(model_results)

            # compute confidence interval
            model_results = np.array(model_results)
            model_ci = 1.96 * np.std(model_results) / np.sqrt(len(model_results))

            linestyle = line_def[model_name] if model_name in line_def else "-"

            index = np.arange(1, 25)
            ax[subplot_idx].plot(
                index,
                model_results,
                label=name_def[model_name],
                color=color_def[model_name],
                linestyle=linestyle,
            )
            if args.plot_ci:
                ax[subplot_idx].fill_between(
                    index,
                    model_results - model_ci,
                    model_results + model_ci,
                    color=color_def[model_name],
                    alpha=0.2,
                )

        # ax[subplot_idx].set_xlim([1, 50.0])
        ax[subplot_idx].set_title(f"N={effect_case}", fontsize=10)
        ax[subplot_idx].set_xticks(np.arange(0, 26, 10))
        if subplot_idx == 0 or subplot_idx == 3:
            ax[subplot_idx].set_ylabel("Accuracy (%)")
            # ax[subplot_idx].set_xlabel("Number of classes")

        # plot line for random performance
        x = np.arange(1, 25)
        y = 1 / (x + 1)
        ax[subplot_idx].plot(x, y * 100, label="Random", color="gray", linestyle="-")

        # ax[subplot_idx].legend()
        # plt.tight_layout()

        # Shrink current axis by 20%
        # box = ax[subplot_idx].get_position()
        # ax[subplot_idx].set_position(
        #    [box.x0, box.y0 + box.height * 0.35, box.width, box.height * 0.65]
        # )
        # plt.tight_layout()

        if subplot_idx == 4:
            ax[subplot_idx].set_xlabel("Retrieval Set Size")

        ax[subplot_idx].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax[subplot_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax[subplot_idx].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))
        ax[subplot_idx].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))
        ax[subplot_idx].grid(c="lightgray", zorder=0, which="both")

        filepath = os.path.join(
            args.output_dir, "psm", f"multi_effect_plot_effects={effect_case}.png"
        )
        plt.savefig(filepath, dpi=300)
        plt.savefig(filepath.replace(".png", ".pdf"))

    # plot overall results
    subplot_idx += 1
    for model_idx, model_name in enumerate(model_names):
        mean_overall_results = np.array(overall_results[model_name])
        print(mean_overall_results.shape)
        mean_overall_results = np.mean(mean_overall_results, axis=0)

        linestyle = line_def[model_name] if model_name in line_def else "-"

        # plot with sequence of line styles
        ax[subplot_idx].plot(
            np.arange(1, 25),
            mean_overall_results,
            label=name_def[model_name],
            color=color_def[model_name],
            # marker=".",
            # alpha=alpha_def[model_name],
            linestyle=linestyle,
        )
    ax[subplot_idx].set_title(f"Overall", fontsize=10)

    # ax[subplot_idx].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    # ax[subplot_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    ax[subplot_idx].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))
    ax[subplot_idx].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))
    ax[subplot_idx].grid(c="lightgray", zorder=0, which="both")
    # ax[subplot_idx].set_xticks(np.arange(0, 26, 5))

    # plot line for random performance
    x = np.arange(1, 25)
    y = 1 / (x + 1)
    ax[subplot_idx].plot(x, y * 100, label="Random", color="gray", linestyle="-")

    # Shrink current axis by 20%

    ax[subplot_idx].legend(
        loc="upper center",
        bbox_to_anchor=(-0.8, -0.55),
        ncol=3,
        fancybox=False,
        shadow=False,
        frameon=False,
        # linewidth=2.0,
    )
    plt.subplots_adjust(hspace=0.5)

    # box = ax[subplot_idx].get_position()
    # ax[subplot_idx].set_position(
    #    [box.x0, box.y0 + box.height * 0.35, box.width, box.height * 0.65]
    # )

    fig.subplots_adjust(bottom=0.3)

    filepath = os.path.join(args.output_dir, "psm", f"multi_effect_plot_overall.png")
    plt.savefig(filepath, dpi=300)
    plt.savefig(filepath.replace(".png", ".pdf"))
    plt.close("all")

    # cycle through linestyles
    linestyles = itertools.cycle(["-", "--", "-.", ":"])

    # create a separate plot for each model
    # -------------------------------------
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5, 6), sharey=True)
    axs = [axs]
    plt_idx = 0
    for model_idx, model_name in enumerate(name_def.keys()):
        if model_name not in overall_results:
            continue
        mean_overall_results = np.array(overall_results[model_name])
        print(mean_overall_results.shape)
        mean_overall_results = np.mean(mean_overall_results, axis=0)

        # plot with sequence of line styles
        axs[plt_idx].plot(
            np.arange(1, 25),
            mean_overall_results,
            label=name_def[model_name],
            linestyle=next(linestyles),
        )

        # axs[plt_idx].set_title(f"{name_def[model_name]}", fontsize=10)
        axs[plt_idx].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
        axs[plt_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
        axs[plt_idx].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))
        axs[plt_idx].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))
        axs[plt_idx].grid(c="lightgray", zorder=0, which="both")
        # ax[subplot_idx].set_xticks(np.arange(0, 26, 5))

    # plot line for random performance
    x = np.arange(1, 25)
    y = 1 / (x + 1)
    axs[plt_idx].plot(x, y * 100, label="Random", color="gray", linestyle="-")
    axs[plt_idx].set_xlabel("Retrieval Set Size")
    axs[plt_idx].set_ylabel("Accuracy (%)")
    # plt_idx += 1

    axs[plt_idx].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.55),
        ncol=3,
        fancybox=False,
        shadow=False,
    )
    plt.subplots_adjust(bottom=0.35)

    filepath = os.path.join(args.output_dir, "psm", f"full_mulit_effect_plot.png")
    plt.savefig(filepath, dpi=300)
    plt.savefig(filepath.replace(".png", ".pdf"))

    for test_type in ["inter-effect", "intra-effect-easy"]:
        fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(14, 6), sharey=True)
        axs = np.reshape(axs, -1)

        # each subplot will be dedicated to a different effect
        for idx, effect_name in enumerate(effect_names):
            # collect the results from each model on this test_type
            models_true = []
            models_false = []
            for model_name in model_names:
                model_results = np.array(results[model_name][test_type][effect_name])
                model_true = np.mean(model_results) * 100
                model_false = 100 - model_true
                models_true.append(model_true)
                models_false.append(model_false)

            # now create a stacked bar plot for this model
            axs[idx].set_title(effect_name.capitalize())
            bar_width = 0.8

            capital_model_names = [name_def[model] for model in model_names]

            axs[idx].barh(
                capital_model_names,
                models_true,
                color=colors,
                label="True",
                # edgecolor="black",
                # width=bar_width,
                zorder=5,
            )

            # add a dotted line vertical for human performance
            if "human" in model_names:
                axs[idx].axvline(
                    x=(results["human"][test_type][effect_name] * 100) - 1,
                    color="darkgray",
                    linestyle="--",
                    zorder=4,
                )

            # add the value of each bar as white text on top of the bar
            fontweight = "heavy"
            for i, v in enumerate(models_true):
                if v == 100.0:
                    axs[idx].text(
                        v - 17.5,
                        i - 0.25,
                        f"{v:.0f}",
                        color="white",
                        fontsize=8,
                        zorder=5,
                        fontweight=fontweight,
                    )
                elif v == 0.0:
                    axs[idx].text(
                        v + 1.25,
                        i - 0.25,
                        f"{v:.1f}",
                        color="black",
                        fontsize=8,
                        zorder=5,
                        fontweight=fontweight,
                    )
                elif v < 15.0:
                    axs[idx].text(
                        v + 1.25,
                        i - 0.25,
                        f"{v:.1f}",
                        color="black",
                        fontsize=8,
                        zorder=5,
                        fontweight=fontweight,
                    )
                else:
                    axs[idx].text(
                        v - 17.5,
                        i - 0.25,
                        f"{v:.1f}",
                        color="white",
                        fontsize=8,
                        zorder=5,
                        fontweight=fontweight,
                    )

            if True:
                axs[idx].barh(
                    capital_model_names,
                    models_false,
                    left=models_true,
                    color="lightgray",
                    label="False",
                    # edgecolor="black",
                    # width=bar_width,
                    zorder=3,
                )

            # disable spines
            axs[idx].spines["right"].set_visible(False)
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["left"].set_visible(False)
            axs[idx].spines["bottom"].set_visible(False)

            # axs[idx].grid(c="lightgray", zorder=0, axis="x")

            # rotate the x-axis labels by 90 deg
            axs[idx].tick_params(axis="x", rotation=-45)

            # left align the y-axis labels
            axs[idx].tick_params(axis="y", pad=2.0)

            # make the tick labels be integers
            axs[idx].xaxis.set_major_locator(plt.MaxNLocator(5))

            # add space between bars

        # disable the last subplot
        axs[-1].axis("off")
        plt.tight_layout()
        filepath = os.path.join(args.output_dir, "psm", f"abx_barplot_{test_type}.png")
        plt.savefig(filepath, dpi=300)
        plt.close("all")
