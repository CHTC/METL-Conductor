from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    completed_jobids = {'204b38e5', '01378e4a', '9208b80f', 'e68502e5', 'f045a936', 'fadd9684'}
    nairr_eval_df = pd.read_csv("nairr_completed_best_evals.csv")
    nairr_eval_df = nairr_eval_df[nairr_eval_df['JobId'].isin(completed_jobids)]
    nairr_eval_df['platform'] = f"NAIRR (N = {len(nairr_eval_df)})"
    chtc_eval_df = pd.read_csv("CHTC_best_model_evals.csv")
    chtc_eval_df['platform'] = f"CHTC (N = {len(chtc_eval_df)})"

    eval_df = pd.concat([nairr_eval_df, chtc_eval_df], ignore_index=True)

    fig, ax = plt.subplots(figsize = (10, 6))

    sns.violinplot(ax = ax, data = eval_df, x = 'platform', y = 'test_loss')
    sns.swarmplot(ax = ax, data = eval_df, x = 'platform', y = 'test_loss', color = 'black')

    plt.ylim(20, 30)
    plt.ylabel('Best Model Test Loss')
    plt.title('Distribution of Test losses for Best Models trained on  NAIRR and CHTC')
    plt.savefig('best_model_test_losses_boxplot.png')
    plt.show()
    

if __name__ == "__main__":
    main()