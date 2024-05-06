# Meta-Learning-and-Few-Shot-Learning-on-Taxi-Driver-Trajectory-Dataset

In myprevious project on Sequence Classification with Taxi Driver Dataset, the task involved classifying the driver of a given trajectory based on a 100-step length sub-trajectory. However, this Project presented a more challenging scenario, where the training data consisted of 400 drivers with only 5-day trajectories for each driver. The objective was to utilize few-shot learning techniques to build a classification model capable of identifying whether two trajectories belong to the same driver.

![image](https://github.com/santhoshraghu/Meta-Learning-and-Few-Shot-Learning-on-Taxi-Driver-Trajectory-Dataset/assets/55938970/d5849a82-62ea-4916-870d-80d1b45538f7)


**Dataset Description:**

The dataset is structured as a dictionary, where each key represents the ID of a driver, and the corresponding value is a list of trajectories associated with that driver. Each trajectory is represented by elements containing plate, longitude, latitude, second_since_midnight, status, and time. The training data comprises trajectories for 400 drivers, each spanning 5 days. Additionally, a validation dataset is provided, consisting of trajectories for 20 different drivers, each spanning 5 days.

**Feature Description:**

- Plate: Represents the taxi's plate, anonymized to values ranging from 0 to 500. The plate serves as the target label for classification.
- Longitude: Indicates the longitude of the taxi.
- Latitude: Indicates the latitude of the taxi.
- Second_since_midnight: Denotes the number of seconds that have passed since midnight.
- Status: Indicates whether the taxi is occupied (1) or vacant (0).
- Time: Represents the timestamp of the record.

**Problem Definition:**

Given two 100-step length sub-trajectories, the task is to predict whether these trajectories belong to the same driver.

**Approach:**

To address this problem, several few-shot learning techniques were explored, including meta-learning algorithms and other state-of-the-art approaches. The primary steps involved:

1. Data Pre-processing: Standard data pre-processing techniques were applied, including normalization, feature scaling, and generating pairs of 100-step length sub-trajectories.
  
2. Model Development: Various binary classification models were developed, each trained to predict whether two trajectories belong to the same driver. Meta-learning algorithms and other relevant techniques were employed to adapt the models to the few-shot learning scenario.

