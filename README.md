### Resnet - 32 implementation for CIFAR - 10 (pytorch)

- **For training**

  ~~~ 
  python main.py 
  # argparser Default 
  --print_freq 32 --save_dir ./save_model/ --save_every 10 --lr 0.1 --weight_decay 1e-4 --momentum 0.9 --Epoch 80 --batch_size 128 --test_batch_size 100 
  ~~~

---

Resnet Motivation : As the model gets deeper, shouldn't the performance at least be greater than equal to the shallow model? 

-Dataset : CIFAR - 10
##### Preprocessing 

- As written in paper 
  - **Data augmentation**
    - 4pixels padded on each side
    - Randomly 32x32 crop from the padded image or horizontal flip
    - For the test, 
      - Original 32x32 image
    - Normalization with the per-pixel mean substracted 

##### parameters

- Weight initialization
  - kaiming_normal from Reference paper[13]

- Optimizer
  - SGD 
  - Learning Rate 
    - Start weight 0.1 divide it by 10 at 32k and 48k iterations 
  - Weight decay : 1e-4
  - momentum : 0.9

##### Others

- Global Average pooling

- 10-way fully-connected layer (softmax)

- 6n + 2 stacked weights (This implementation n = 5)

- Use identity shortcuts in all cases (option A in paper, IdentityPadding) 

- subsampling : convolution with stride = 2

  