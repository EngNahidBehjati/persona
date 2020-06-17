from matplotlib import pyplot
from pandas import DataFrame

from topic_extraction.advisor import Advisor


class ComparativeView:

    @classmethod
    def plot_metrics(cls, metrics: dict, explore_param: str, lang, model_name, data_version, dict_version, model_version):
        plot_no = len(metrics)
        fig, axes = pyplot.subplots(plot_no, 1, figsize=(2 * 8.0, (plot_no + 1) * 5.0))
        if plot_no > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i, metric in enumerate(metrics):
            data = DataFrame(metrics[metric], columns=[explore_param, metric])
            ax = axes[i]
            ax.plot(data[explore_param], data[metric], marker='o', )
            ax.set_title("Coherence of %s" % metric)
            ax.set_xlabel(explore_param)
            ax.set_xticks(data[explore_param])
        file_name = "%s-%s-%s" % (model_name, explore_param, model_version)
        fig_file_name = Advisor.get_model_version_folders_file_path(lang, data_version, dict_version,
                                                                    model_version,
                                                                    file_name, "png")
        fig.savefig(fig_file_name)
        pyplot.close(fig=fig)
        return
