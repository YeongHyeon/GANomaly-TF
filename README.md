[TensorFlow] GANomaly
=====

TensorFlow implementation of GANomaly with MNIST dataset.  
<a href="https://github.com/YeongHyeon/GANomaly-PyTorch">PyTorch Version</a> is also implemented.

## Summary

### GANomaly architecture
<div align="center">
  <img src="./figures/ganomaly.png" width="650">  
  <p>Simplified GANomaly architecture.</p>
</div>

### Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="650">  
  <p>Graph of GANomaly.</p>
</div>

### Problem Definition
<div align="center">
  <img src="./figures/definition.png" width="450">  
  <p>'Class-1' is defined as normal and the others are defined as abnormal.</p>
</div>

## Results

### Training Procedure
<div align="center">
  <p>
    <img src="./figures/GANomaly_loss_enc.svg" width="300">
    <img src="./figures/GANomaly_loss_con.svg" width="300">
    </br>
    <img src="./figures/GANomaly_loss_adv.svg" width="300">
    <img src="./figures/GANomaly_loss_target.svg" width="300">
  </p>
  <p>Loss graph in the training procedure. </br> Each graph shows encoding loss, reconstruction loss, adversarial loss, and total (target) loss respectively.</p>
</div>

<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by GANomaly.</p>
</div>

### Test Procedure
<div align="center">
  <img src="./figures/test-box.png" width="400">
  <p>Box plot with encoding loss of test procedure.</p>
</div>

<div align="center">
  <p>
    <img src="./figures/in_in01.png" width="130">
    <img src="./figures/in_in02.png" width="130">
    <img src="./figures/in_in03.png" width="130">
  </p>
  <p>Normal samples classified as normal.</p>

  <p>
    <img src="./figures/in_out01.png" width="130">
    <img src="./figures/in_out02.png" width="130">
    <img src="./figures/in_out03.png" width="130">
  </p>
  <p>Abnormal samples classified as normal.</p>

  <p>
    <img src="./figures/out_in01.png" width="130">
    <img src="./figures/out_in02.png" width="130">
    <img src="./figures/out_in03.png" width="130">
  </p>
  <p>Normal samples classified as abnormal.</p>

  <p>
    <img src="./figures/out_out01.png" width="130">
    <img src="./figures/out_out02.png" width="130">
    <img src="./figures/out_out03.png" width="130">
  </p>
  <p>Abnormal samples classified as abnormal.</p>
</div>

## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] S Akcay, et al. (2018). <a href="https://arxiv.org/abs/1805.06725">Ganomaly: Semi-supervised anomaly detection via adversarial training.</a>. arXiv preprint arXiv:1805.06725.
