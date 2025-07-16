cd residuals_for_ed
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.virtualenvs/Residuals_env/bin/activate
### sr_bench Feynman dataset with 3 seed:
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_cvgp.py           --exp_name "normal" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_E2E.py            --exp_name "normal" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_gplearn.py        --exp_name "normal" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "normal" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "normal" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_symbolicgpt.py    --exp_name "normal" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10


/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_cvgp.py           --exp_name "normal" --seed 1 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_gplearn.py        --exp_name "normal" --seed 1 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "normal" --seed 1 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "normal" --seed 1 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_symbolicgpt.py    --exp_name "normal" --seed 1 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10

/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_cvgp.py           --exp_name "normal" --seed 2 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_gplearn.py        --exp_name "normal" --seed 2 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "normal" --seed 2 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "normal" --seed 2 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_symbolicgpt.py    --exp_name "normal" --seed 2 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10


# sr_bench How often Residuals
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_cvgp.py           --exp_name "normal" --seed 0 --max_num_residuals 100 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_gplearn.py        --exp_name "normal" --seed 0 --max_num_residuals 100 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "normal" --seed 0 --max_num_residuals 100 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "normal" --seed 0 --max_num_residuals 100 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_symbolicgpt.py    --exp_name "normal" --seed 0 --max_num_residuals 100 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10

# sr_bench Noise

/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "noise" --seed 0 --max_num_residuals 10 --noise_factor 0.1 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "noise05" --seed 0 --max_num_residuals 10 --noise_factor 0.5 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "noise10" --seed 0 --max_num_residuals 10 --noise_factor 1 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "noise03" --seed 0 --max_num_residuals 10 --noise_factor 0.3 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10

# sr_bench  number_samples Nesymres


/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 500 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 200 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 100 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 50 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 20 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 10 --dataset_folder datasets_srbench --pysr_niterations 10


# sr_bench  number_samples PySR
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 500 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 200 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 100 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 50 --dataset_folder datasets_srbench --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "number_samples" --seed 0 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 20 --dataset_folder datasets_srbench --pysr_niterations 10


### dso dataset with 3 seed:

/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_cvgp.py           --exp_name "normal" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_gplearn.py        --exp_name "normal" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "normal" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "normal" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_symbolicgpt.py    --exp_name "normal" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10

/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_cvgp.py           --exp_name "normal" --seed 5 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_gplearn.py        --exp_name "normal" --seed 5 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "normal" --seed 5 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "normal" --seed 5 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_symbolicgpt.py    --exp_name "normal" --seed 5 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10

/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_cvgp.py           --exp_name "normal" --seed 6 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_gplearn.py        --exp_name "normal" --seed 6 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_nesymres.py       --exp_name "normal" --seed 6 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "normal" --seed 6 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_symbolicgpt.py    --exp_name "normal" --seed 6 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_dso_1000 --pysr_niterations 10

# pysr iterations:
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "pysr_only" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 10  --only_classic True
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "pysr_only" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 50  --only_classic True
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "pysr_only" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 100 --only_classic True
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "pysr_only" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 150 --only_classic True
/home/jbrugger/.virtualenvs/Residuals_env/bin/python3.10 src/fit_func_pysr.py           --exp_name "pysr_only" --seed 4 --max_num_residuals 10 --noise_factor 0 --max_dataset_size 300 --dataset_folder datasets_srbench --pysr_niterations 200 --only_classic True
