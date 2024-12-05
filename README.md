# Auto-Multilift
The **Auto-Multilift** is a novel learning framework for cooperative load transportation with quadrotors. It can automatically tune various MPC hyperparameters, which are modeled by DNNs and difficult to tune manually, via reinforcement learning in a distributed and closed-loop manner.

|                     A Diagram of a multilift system and Auto-Multilift Learning Piplines             |
:----------------------------------------------------------------------------------------------------------------------------------:
![learning framework](https://github.com/user-attachments/assets/50dbedaa-f669-425d-a6d6-5e724c43abca)



Please find out more details in our paper: "Auto-Multilift: Distributed Learning and Control for Cooperative Load Transportation With Quadrotors" [arXiv](http://arxiv.org/abs/2406.04858)


## Table of contents
1. [Project Overview](#project-Overview)
2. [Dependency Packages](#Dependency-Packages)
3. [How to Use](#How-to-Use)
      1. [A: Distributed Learning of Adaptive Weightings](#A-Distributed-Learning-of-Adaptive-Weightings)
      2. [B: Distributed Learning of Adaptive References](#B-Distributed-Learning-of-Adaptive-References)
4. [Contact Us](#Contact-Us)

## 1. Project Overview
The project consists of two folders, which correspond to the two experiments in the paper that show the following three advantages of our method.
1. Auto-Multilift enjoys fast convergence in just a few iterations, with its convergence speed unaffected by the number of quadrotors.
2. Auto-Multilift is able to learn adaptive MPC weightings directly from trajectory tracking errors. Additionally, it significantly improves training stability and tracking performance over a state-of-the-art open-loop learning method [[1]](#1).
3. Beyond its improved training ability to learn adaptive MPC weightings, our method can effectively learn an adaptive tension reference, enabling the multilift system to reconfigure itself when traversing through obstacles.



## 2. Dependency Packages
Please make sure that the following packages have already been installed before running the source code.
* CasADi: version 3.5.5 Info: https://web.casadi.org/
* ACADOS(stable version v0.3.5): Info: https://docs.acados.org/ (Note that DO NOT use the lastest version of Acados!)
* Numpy: version 1.23.0 Info: https://numpy.org/
* Pytorch: version 1.12.0+cu116 Info: https://pytorch.org/
* Matplotlib: version 3.3.0 Info: https://matplotlib.org/
* Python: version 3.9.12 Info: https://www.python.org/
* Scipy: version 1.8.1 Info: https://scipy.org/
* Pandas: version 1.4.2 Info: https://pandas.pydata.org/
* scikit-learn: version 1.0.2 Info: https://scikit-learn.org/stable/whats_new/v1.0.html

## 3. How to Use
First and foremost, the implementation for Auto-Multilift is straightforward to setup. The source code has been comprehensively annotated to facilitate ease of use. To reproduce the simulation results presented in the paper, simply follow the steps outlined below, sequentially, after downloading and decompressing all the necessary folders.


### A: Distributed Learning of Adaptive Weightings
 Auto-Multilift       |      Safe-PDP (Open-loop Learning)
:---------------------------------------------------------------:|:--------------------------------------------------------------:
![cl](https://github.com/user-attachments/assets/008e356f-c9e6-475e-8230-2686ab05b600) | ![ol](https://github.com/user-attachments/assets/f3ff696e-0614-469d-8195-515795b77962)




1. Open the Python file '**main_distributed_autotuning_acados.py**' in the folder '**Source code A**'
2. Before running, please do the following settings:
   * Set the number of quadrotors on line 42 (i.e., the fifth number in the 'uav_para' array).
   * Set the load mass value on line 43 (i.e., the first number in the 'load_para' array).
   * Set the MPC horizon on line 52 (the default value is 10).
   * Set the higher-level loss horizon on line 53 (the default value is 20).
4. After completing the above settings, run the file '**main_distributed_autotuning_acados.py**'. In the prompted terminal interface, you will be asked to select the control and sensitivity propagation modes.
   * In our settings, 's' and 'p' denote 'sequential' and 'parallel' computing, respectively.
   * 'c' and 'o' represent 'closed-loop' (our method) and 'open-loop' (the Safe-PDP method [[1]](#1)) training modes.
5. To evaluate the trained model, run the Python file '**main_distributed_autotuning_evaluation_acados.py**'
   * You can skip Step 4 and evaluate the saved models that were previously trained and employed in the paper. To do so, copy the files that end with '.pt' from the folder '**Previously trained models**' (within the folder '**Source code A**') to the folder '**trained data**' (where the retained models via Step 4 will be saved).


### B: Distributed Learning of Adaptive References
|                     Flying Through Narrow Slots With Adaptive Tension References                                                                |
:----------------------------------------------------------------------------------------------------------------------------------------:
![tensionref_dt_edit](https://github.com/user-attachments/assets/03ce359b-ccad-440c-9d9a-60ba35bc7a2a)


1. Open the Python file '**main_distributed_autotuning_acados_tensionref.py**' in the folder '**Source code B**'
2. Before running, please do the following settings:
   * Set the number of quadrotors on line 40 to the same as that used in Section A (i.e., learning of adaptive weightings). 
   * Set the load mass value on line 43 (i.e., the first number in the 'load_para' array).
   * Set the MPC horizon on line 52 (the default value is 10).
   * Set the higher-level loss horizon on line 53 (the default value is 20).
4. After completing the above settings, run the file '**main_distributed_autotuning_acados_tensionref.py**'. In the prompted terminal interface, you will be asked to select the control and sensitivity propagation modes.
   * In our settings, 's' and 'p' denote 'sequential' and 'parallel' computing, respectively.
5. To evaluate the trained model, run the Python file '**main_distributed_acados_tensionref_evaluation.py**'
   * You can skip Step 4 and evaluate the saved models that were previously trained and employed in the paper. To do so, copy the files that end with '.pt' from the folder '**Previously trained models**' (within the folder '**Source code B**') to the folder '**trained data**' (where the retained models via Step 4 will be saved).

## 4. Contact Us
If you encounter a bug in your implementation of the code, please do not hesitate to inform me.
* Name: Dr. Bingheng Wang
* Email: wangbingheng@u.nus.edu

## References
<a id="1">[1]</a> 
Jin, Wanxin and Mou, Shaoshuai and Pappas, George J, "Safe pontryagin differentiable programming", Advances in Neural Information Processing Systems, 34, 16034--16050, 2021
