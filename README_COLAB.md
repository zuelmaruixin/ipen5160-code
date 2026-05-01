# README for Colab

1. Upload the whole `colab_package` directory to Colab or Google Drive.
2. Recommended locations: `/content/colab_package` or `/content/drive/MyDrive/colab_package`.
3. If the notebook cannot auto-detect the root, set `MANUAL_PROJECT_ROOT` in the notebook setup cell.
4. Open `notebooks/02_colab_training.ipynb`.
5. Run the token length analysis cell first. It will write `outputs/tables/token_length_distribution.csv` and `outputs/tables/truncated_examples.csv`.
6. Keep `SELECTED_CONFIG_KEY = 'maxlen_256'` for the first run. Only switch to `maxlen_384` or `maxlen_512` after checking truncation ratios and macro-F1.
7. After training, download the generated `outputs/` files and place them back into the local project.
