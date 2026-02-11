from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    jobids_with_29_epochs = {'204b38e5', '01378e4a', '9208b80f', 'e68502e5', 'f045a936', 'fadd9684'}
    nairr_eval_df = pd.read_csv("nairr_completed_evals.csv")
    nairr_eval_df = nairr_eval_df[nairr_eval_df['JobId'].isin(jobids_with_29_epochs)]
    nairr_eval_df['platform'] = f"NAIRR (N = {len(nairr_eval_df)})"
    chtc_eval_df = pd.read_csv("CHTC_best_model_evals.csv")
    chtc_eval_df['platform'] = f"CHTC (N = {len(chtc_eval_df)})"
    nairr_eval_df['best_model_idx'] = nairr_eval_df['best_model_name'].str.extract(r'epoch=(\d+)')[0].dropna().astype(int).to_list()
    chtc_eval_df['best_model_idx'] = chtc_eval_df['best_model_name'].str.extract(r'epoch=(\d+)')[0].dropna().astype(int).to_list()

    eval_df = pd.concat([nairr_eval_df, chtc_eval_df], ignore_index=True)

    fig, ax = plt.subplots(figsize = (10, 6))

    sns.violinplot(ax = ax, data = eval_df, x = 'platform', y = 'best_model_idx')
    sns.swarmplot(ax = ax, data = eval_df, x = 'platform', y = 'best_model_idx', color = 'black')
    plt.ylim(20, 30)
    plt.ylabel('Best Model Epoch Index')
    plt.title('Distribution of Best Model Epoch Indices for NAIRR and CHTC')
    plt.savefig('best_model_epoch_indices_boxplot.png')
    plt.show()
    

if __name__ == "__main__":
    main()