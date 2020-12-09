# Robot Open Autonomous Racing (ROAR)

### To Contribute

- Please click the [Fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) button on the upper right corner and submit a pull request to master branch.
  - For a more in-depth tutorial on recommended setup [video](https://youtu.be/VA13dAZ9iAw)
- Please follow suggested guidelines on Pull Request.

### Quick start

For quick start documentation, please visit our documentation site: [https://augcog.github.io/ROAR/quickstart/](https://augcog.github.io/ROAR/quickstart/)

### Submission Notes

The submission agent is the ImitationAgent. This loads 2 Tensorflow Keras models and feeds them the front camera input to determine the appropriate control output. One model determines whether to steer, the other determines how much to steer if the first determines to steer. The models were trained from data collected during a manual driving session in which I completed multiple laps around the course. Data collection only occurred every 10 frames to maintain smooth rendering. Many adjustments had to be made to the keyboard controls in order to produce smooth control.

The main challenge was sparsity of data and lack of fine vehicle control. Most segments in this track were straight, which means that there was little to no steering involved with the original control scheme. Thus, I modified steering to behave more like a joystick such that there was continuous steering over frames. Because during my manual driving I kept the throttle at 0.8, there was no reason for the model to learn this, and so the only control it outputs is steering.

Improvements that could be made include diversified environments, finer vehicle control in the form of virtual/physical steering wheel, more training data, and using more inputs such as the depth field camera and past control outputs. The last of these was not implemented due to computational limitations as the training was done on my local device.

In my testing, the final model was only able to complete the first turn. The link to the model weights are below. Download and place them in the root folder (same folder as this readme).

ImiCarla model weights: https://drive.google.com/file/d/1yS0gSI2Nksk9U8TcfIvS4ZbpJ5_zAzOE/view?usp=sharing
ImiSteer model weights: https://drive.google.com/file/d/1yDc_5nH9vvW10Dweov6ll-KnkHoXbfuV/view?usp=sharing

The models were trained on CUDA 10.1 and CuDNN 7.6 on an RTX 2060 Super. The backend is Tensorflow 2.3, specifically the gpu version. The module requirements are specified in the requirements.txt file.
