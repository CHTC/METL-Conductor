from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    nairr_eval_df = pd.read_csv("nairr_completed_last_evals.csv")
    # nairr_eval_df = nairr_eval_df[nairr_eval_df['JobId'].isin(completed_jobids)]
    eval_df = pd.read_csv("CHTC_best_model_evals.csv")
    eval_df['platform'] = f'CHTC (N={len(eval_df)})'

    nairr_eval_df['platform'] = f'NAIRR (N={len(nairr_eval_df)})'
    eval_df = pd.concat([eval_df, nairr_eval_df], ignore_index=True)
    
    fig, ax = plt.subplots(figsize = (10, 6))

    sns.violinplot(ax = ax, data = eval_df, x = 'platform', y = 'test_loss')
    sns.swarmplot(ax = ax, data = eval_df, x = 'platform', y = 'test_loss', color = 'black')

    plt.ylim(20, 30)
    plt.ylabel('Last Model Test Loss')
    plt.xlabel('Compute Platform')
    plt.title('Distribution of Test losses for Last Models trained on  NAIRR and CHTC')
    plt.savefig('last_model_test_losses_boxplot.png')
    plt.show()
    

if __name__ == "__main__":
    main()