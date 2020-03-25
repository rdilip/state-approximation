# Decomposing states using the Moses Move
This library uses the Moses Move (see https://arxiv.org/abs/1902.05100) to decompose an MPS into a low entanglement entropy MPS and a series of two site unitary gates. For the bulk of the code, see `state_approximation.py'. 

## Results
### Dimerized Heisenberg Model
![image](img/dimerized_heisenberg_var.png)
![image](img/dimerized_heisenberg_nonvar.png)

### Sheng-Hsuan's wavefunctions
![image](img/T0.0.png) ![image](img/T0.1.png)
![image](img/T0.2.png) ![image](img/T0.3.png)
![image](img/T0.4.png) ![image](img/T0.5.png)
![image](img/T0.6.png) ![image](img/T0.7.png)
![image](img/T0.8.png) ![image](img/T0.9.png)
![image](img/T1.0.png)

The variational optimization (starting from the Moses Move guess) consistently
improves the fidelity with the original state.

## TODO
* Bug for complex wavefunctions -- it's unclear what the correct conjugations should be and I'm not sure why.
* Add the remaining images to README
* Benchmark on different states
* Experiment with more disentanglers
