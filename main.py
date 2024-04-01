import math
import wandb
from main_training import train
from Params import get_run_parameters
from LogFunctions import print_and_log_message
from LogFunctions import print_run_info_to_log
from OutputHandler import make_folder
from Params import load_config_parameters
import traceback
from io import StringIO


def main(wb_flag=False, sweep_configuration=''):
    p = get_run_parameters()
    if wb_flag:
        wandb.login(key='')
        wandb.init()
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="Optimal Diffuser")
        wandb.agent(sweep_id, function=main, count=4)
    else:
        # search_parameters(p)
        run_model(p, False)


def search_parameters(p):
    for z_dim in [28, 128, 512]:
        p['z_dim'] = z_dim
        for bs in [2, 4, 8]:
            p['batch_size'] = bs
            for weight_decay in [1e-7, 5e-7, 10e-7]:
                p['weight_decay'] = weight_decay
                for TV_beta in [0.1, 0.5, 1.0, 10, 100]:
                    p['TV_beta'] = TV_beta
                    for cr in [1, 2, 5, 10]:
                        p['cr'] = cr
                        p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
                        run_model(p, False)
    print('finished successfully')


def run_model(p, wb_flag):
    try:
        folder_path = make_folder(p)
        log_path = print_run_info_to_log(p, folder_path)
        train(p, log_path, folder_path, wb_flag)
    except Exception as e:
        trace_output = StringIO()
        traceback.print_exc(file=trace_output)
        error_message1 = f"Error occurred for this parameters. Exception: {str(e)}"
        error_message2 = trace_output.getvalue()
        print_and_log_message(error_message1, log_path)
        print_and_log_message(error_message2, log_path)
        print_and_log_message(folder_path, log_path)


if __name__ == '__main__':
    main()

