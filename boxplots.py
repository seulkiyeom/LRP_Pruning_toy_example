import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

if __name__ == '__main__':

    columns = ["dset", "Criterion", "n", "s", "scenario", "stage",
                 "Train Accuracy"]
    df = pd.read_csv("output/log.txt", sep="-| ", names=columns)

    # Remove weight criterion (does not depend on n)
    df = df[~df.Criterion.str.contains("criterion:weight")]

    # Clean up dataframe
    df.Criterion = df.Criterion.str.replace("criterion:", "")
    df.dset = df.dset.str.replace("dataset:", "")
    df.Criterion = df.Criterion.apply(lambda x: x[0].upper())
    df.n = df.n.str.replace("n:", "").astype(int)
    df["n_orig"] = df.n.copy()
    stage = "stage:post"
    scenario_train = "scenario:train"
    df = df[df.stage == stage][df.scenario == scenario_train]

    # Map n in dataframe to number of reference samples
    n_mapping = {
        "moon": 2,
        "circle": 2,
        "mult": 4,
    }
    num_reference_samples = [1, 5, 20, 100]

    # Define colormap
    default_cycler = cycler(color=['yellowgreen', 'cornflowerblue', 'indianred', "k"])
    plt.rc('axes', prop_cycle=default_cycler)

    for i, dset in enumerate(["moon", "circle", "mult"]):
        df["n"] = df.n_orig // n_mapping[dset]
        df_dset = df[df.n.isin(num_reference_samples)]
        df_dset = df_dset[df_dset.dset == dset]

        fig = plt.figure(figsize=(3.5, 3.5))
        plt.subplots_adjust(left=0.19, right=0.99, top=0.93, bottom=0.13)
        plt.ylim([33, 100])

        sns.boxplot(data=df_dset, y="Train Accuracy", x="n", hue="Criterion", showmeans=False, ax=plt.gca(), hue_order=list("GTL"), fliersize=0)
        if i > 0:
            plt.yticks([], [])
            plt.ylabel(None)
            plt.gca().get_legend().remove()
        plt.title(dset)
        plt.savefig(f"toy_experiment_train-{dset}.pdf")
    plt.show()
