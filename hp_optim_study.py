import sys, optuna

storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("/dccstor/aml-e/datasets/optuna/eth_1.log"))

study = optuna.load_study(study_name="eth_1", storage=storage)

best_trial = study.best_trial

""" for trial in study.trials:
   if trial.value != None and trial.value > 0.5:
      print('')
      print('Value: ', trial.value)
      for key, value in trial.params.items():
         print("{}: {}".format(key, value)) """

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
print('Value: ', best_trial.value)