from clearml import Task

from vit_trainer.config import CLEARML


def load_clearml_config(epochs, seed, batch_size, data_dir):
    Task.set_credentials(
        api_host=CLEARML["api_host"],
        web_host=CLEARML["web_host"],
        files_host=CLEARML["files_host"],
        key=CLEARML["api_access_key"],
        secret=CLEARML["api_secret_key"],
    )
    # Инициализация задачи
    task = Task.init(
        project_name="ViT-Trainer",
        task_name=f"Train_img_ep{epochs}",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        auto_connect_arg_parser=True,
        auto_connect_frameworks=True
    )
    task.connect({
        "epochs": epochs,
        "seed": seed,
        "batch_size": batch_size,
        "split_dir": data_dir or 'data/split',
    })