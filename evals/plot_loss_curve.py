from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os
from matplotlib.backends.backend_pdf import PdfPages

MAX_EPOCHS = 30

def main(args):
    result = {'train_job_id': [], 'epoch_idx': [], 'test_loss': []}
    for train_job_id in args.train_job_ids:
        for epoch in range(MAX_EPOCHS):
            # evals/epoch24/output/inference/204b38e5/rosetta_global/standard_tr0.9_tu0.05_te0.05_w2a93d88bac32_r2098/full_dataset/metrics.txt
            result_file = os.path.join(f"{epoch}", "output", "inference", f"{train_job_id}",
                                       "rosetta_global", 
                                       "standard_tr0.9_tu0.05_te0.05_w2a93d88bac32_r2098",
                                       "full_dataset",
                                       "metrics.txt")
            if not os.path.exists(result_file):
                print(f"Couldn't find eval for train id: {train_job_id}, epoch: {epoch}")
                continue
            result['train_job_id'].append(train_job_id)
            result['epoch_idx'].append(epoch)
            with open(result_file, "r") as iFile:
                scoreLine = iFile.readlines()[2]
                result['test_loss'].append(float(scoreLine.split(',')[1].strip()))
    
    result_df = pd.DataFrame(result)
    mean_loss = []
    epoch_idx = []
    for i in range(MAX_EPOCHS):
        epoch_idx.append(i)
        mean_loss.append(result_df[result_df['epoch_idx'] == i]['test_loss'].mean())

    result_df = result_df.rename(columns={'train_job_id': 'Train Job ID'})
    #result_df = pd.concat([result_df, 
    #                      pd.DataFrame({'Train Job ID': 'Avg. across runs',
    #         'epoch_idx': epoch_idx,
    #         'test_loss': mean_loss
    #        }
    #    )], ignore_index=True
    #)
    
    print(result_df)
    fig, ax = plt.subplots(figsize = (15, 12))
    fig.suptitle(f"Test Loss per epoch for each NAIRR training run.")

    
    sns.set_style("whitegrid")
    g = sns.lineplot(x="epoch_idx",
                    y="test_loss",
                    data=result_df,
                    err_style="band",
                    markers=True, 
                    dashes=False,
                    hue="Train Job ID",
                    legend='full',
                    ci='sd',
                    linewidth=2,         # Makes the main line much thicker
                    palette="muted",      # Uses a darker color palette for better contrast
                    err_kws={
                        # "alpha": 0.4,    # Increases band opacity (0.2 is default)
                        "edgecolor": None # Removes the thin line around the band for a cleaner look
                    })
        # ax[j].xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax[j].set_xticklabels(,)
      
    sns.despine()
    ax.tick_params(axis='x',  rotation=90, labelsize='small')
    plt.ylabel("Test Loss")
    plt.xlabel("Training Epoch")
    
    plt.savefig("loss_curve_nairr.png")
    plt.show()
    plt.close()


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_job_ids', nargs="+", type=str,
                        default=[], help="List of job ids to plot loss curves for.", required=True)

    args = parser.parse_args()
    main(args)