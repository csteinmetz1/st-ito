import os
import json
import numpy
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", help="Root directory for PST benchmark outputs.", type=str
    )
    parser.add_argument(
        "--omit-random", help="Omit random method.", action="store_true"
    )
    parser.add_argument("--omit-input", action="store_true", help="Omit input method.")
    args = parser.parse_args()

    fig = None

    # search for guitar, music, speech, and vocals results
    for mode_idx, mode in enumerate(["music", "speech", "vocals"]):

        input_json_filepath = os.path.join(args.root_dir, mode, "results.json")
        with open(input_json_filepath, "r") as fp:
            data = json.load(fp)

        # first look at the metrics and methods
        metric_names = list(data["input"].keys())
        method_names = list(data.keys())

        metric_names = [
            metric_name
            for metric_name in metric_names
            if metric_name not in ["time_elapsed"]
        ]

        # create a box plot
        if fig is None:
            fig, axs = plt.subplots(
                nrows=3,
                ncols=len(metric_names),
                figsize=(14, 4),
                sharey=True,
                sharex="col",
            )

        if args.omit_random:
            method_names = [
                method_name
                for method_name in method_names
                if "random" not in method_name
            ]

        if args.omit_input:
            method_names = [
                method_name
                for method_name in method_names
                if "input" not in method_name
            ]

        # 5 different color options
        gray = "#a6a6a6"
        yellow = "#f3b263"
        blue = "#8db9f3"
        darkblue = "#335D94"
        green = "#97d9a6"
        red = "#de4e5e"

        colors = [gray, yellow, green, blue, darkblue, red]

        # adjust space between the subplots
        plt.subplots_adjust(wspace=0.3)
        plt.minorticks_on()

        # iterate over each metric
        for idx, metric_name in enumerate(metric_names):
            # create a list of lists for each method
            method_values = []
            for method_name in method_names:
                if "random" in method_name and args.omit_random:
                    continue
                method_values.append(data[method_name][metric_name])

            # create a box plot
            bplot = axs[mode_idx, idx].boxplot(
                method_values, patch_artist=True, showfliers=False, vert=False
            )
            # put title at the bottom
            if mode_idx == 0:
                axs[mode_idx, idx].set_title(
                    metric_name.capitalize(),
                    fontsize=10,
                )

            if idx == 0:
                axs[mode_idx, idx].set_ylabel(mode.capitalize(), fontsize=10)
                # rotate y-axis label to horizontal
                axs[mode_idx, idx].yaxis.label.set_rotation(0)
                # add padding to y-axis label
                axs[mode_idx, idx].yaxis.labelpad = 20
            # axs[idx].set_ylabel(metric_name)
            # axs[idx].set_xlabel("Method")

            # remove y-axis ticks and labels
            axs[mode_idx, idx].set_yticks([])
            axs[mode_idx, idx].set_yticklabels([])

            # increase the numbner of ticks on the x-axis
            # axs[mode_idx, idx].xaxis.set_major_locator(plt.MaxNLocator(5))

            # turn off axis spines
            # axs[mode_idx, idx].spines["top"].set_visible(False)
            # axs[mode_idx, idx].spines["right"].set_visible(False)
            # axs[mode_idx, idx].spines["bottom"].set_visible(False)
            # axs[mode_idx, idx].spines["left"].set_visible(False)

            axs[mode_idx, idx].grid(c="lightgray", linestyle="-")

            # add color to each box
            for patch, color in zip(bplot["boxes"], colors):
                patch.set_facecolor(color)

            # set median line color to black
            for median in bplot["medians"]:
                median.set(color="black", linewidth=1)

            if idx == 3 and mode_idx == 0:
                # create a legend for the boxplot using the colors as lines
                legend_elements = [
                    plt.Line2D([0], [0], color=color, lw=4, label=method_name)
                    for method_name, color in zip(method_names, colors)
                ]

                axs[mode_idx, idx].legend(
                    handles=legend_elements,
                    loc="upper center",
                    bbox_to_anchor=(0.0, 1.7),
                    ncols=5,
                    frameon=False,
                )

            pos = axs[mode_idx, idx].get_position()
            axs[mode_idx, idx].set_position(
                [pos.x0, pos.y0, pos.width, pos.height * 0.85]
            )

        # plt.tight_layout()
        filepath = os.path.join("output", "pst", f"boxplot.png")
        plt.savefig(filepath, dpi=300)
