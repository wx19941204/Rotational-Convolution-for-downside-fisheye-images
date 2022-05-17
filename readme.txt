Train:
1.covert label:"run convert_label_color2id.py"
2.train:"run train_unet.py"

Evaluate MIOU MPA:
1.load model file to "eval_on_val_for_metrics.py" and run.
2. "run compute_iouPR.py"
