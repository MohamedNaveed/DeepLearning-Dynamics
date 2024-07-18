# Deep learning for modeling dynamical systems

Evaluating different neural network architectures for modeling the response of a dynamical system. 
From preliminary results, all the models predict the response in the next time-step well but fail in long-term predictions/forecasting. 

## Systems:
1. Pendulum

## Neural Network Models:
1. MLP
2. ResNet
3. RK4Net (Similar to ResNet but with 4th order Rukka-Kutta integrator)
4. LSTM  

## Packages used:
1. Pytorch
2. Pandas
3. matplotlib
4. numpy

## Results on the pendulum model:
1. Phase Portrait of data sets:
 <table>
  <tr>
    <td>
      <figure>
        <img src="../main/results/pendulum_exps/near90deg/phase_portrait_near90deg_traindata.png" alt="" width="400">
        <figcaption><br> Dataset 1</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="../main/results/pendulum_exps/diffInitialConditions/phasePortrait_1M.png" alt="" width="400">
        <figcaption><br> Dataset 2</figcaption>
      </figure>
    </td>
  </tr>
</table>
<br>

2. Performance using MLP (trained on dataset 1) on a single trajectory in the distribution of Dataset 1 .<br>
<table>
  <tr>
    <td>
      <figure>
        <img src="../main/results/pendulum_exps/near90deg/simplenn_1step_prediction.png" alt="" width="400">
        <figcaption><br> 1-step prediction performance using MLP</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="../main/results/pendulum_exps/near90deg/simplenn_seq_prediction.png" alt="" width="400">
        <figcaption><br> Recursive prediction/Forecasting</figcaption>
      </figure>
    </td>
  </tr>
</table>
<br>
3. Performance using ResNet (trained on dataset 2) on a single trajectory in the distribution of Dataset 2.<br>
<table>
  <tr>
    <td>
      <figure>
        <img src="../main/results/pendulum_exps/near90deg/resnet_1step_prediction.png" alt="" width="400">
        <figcaption><br> 1-step prediction performance</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="../main/results/pendulum_exps/near90deg/resnet_seq_prediction.png" alt="" width="400">
        <figcaption><br> Recursive prediction/Forecasting</figcaption>
      </figure>
    </td>
  </tr>
</table>




