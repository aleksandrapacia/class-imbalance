from mla.evaluate import run_cross_entropy, run_focal_loss
from mla.plots import save_roc_auc_figures


def main():
    run_cross_entropy()
    run_focal_loss()
    save_roc_auc_figures()


if __name__ == "__main__":
    main()
