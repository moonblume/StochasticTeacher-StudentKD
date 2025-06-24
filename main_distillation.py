import os
import subprocess


os.environ['PATH'] = os.environ['PATH'] + "path to environment"


os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + "path to src folder"


def run_distillation(fold_id, teacher_model_path, config_file, results_dir):
    command = [
        "path to environment python",
        "path to Tdistillation.py",
        "--config", config_file,
        "--device", "0",
        "--fold_id", str(fold_id),
        "--teacher_model_path", teacher_model_path,
        "--results_dir", results_dir
    ]
    subprocess.run(command, check=True)


base_dir = "path to /saved_modelsT0T4pre/LOSO_0_vs_1"
results_dir = "path to output folder"
config_file = "path to src/config.json"

# Loop over fold directories and run distillation
for fold_dir in os.listdir(base_dir):
    full_fold_dir = os.path.join(base_dir, fold_dir)
    if os.path.isdir(full_fold_dir):
        # Assuming the directory name contains 'fold' followed by the fold ID
        fold_id = int(fold_dir.split("fold")[-1])
        teacher_model_path = os.path.join(full_fold_dir, "model_best.pth")
        
        if os.path.exists(teacher_model_path):
            print(f"Running distillation for fold {fold_id} with model at {teacher_model_path}")
            try:
                run_distillation(fold_id, teacher_model_path, config_file, results_dir)
            except subprocess.CalledProcessError as e:
                print(f"Distillation failed for fold {fold_id}: {e}")
        else:
            print(f"Skipping fold {fold_id}: model_best.pth not found.")
