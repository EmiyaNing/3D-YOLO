## 3D-transformer-yolo
This repro implement a Sparse 3D Detection algorithm.

### First Step
1. I will combine the yolo3d and yolox's code.
2. Try to use the yolov5's target balance code to enhance the convergence performance.
3. Try the FaFPN to the algorithm.

### Probablely useful method
1. Use transformer to extract 2D feature from 3D point cloud.
2. Try another representation of 3D point cloud.


### 2021/10/6 work
1. Add yolox's model code
2. Add yolox's bboxes.py file
3. Add some unit_test code to verify the yolox code
4. Test the yolo_fpn classes
5. Test the yolo_pafpn classes
6. Test the yolo_head classes in eval state

### Todo list
1. Test the yolo_head in train state
2. Add dataset.py && dataloader.py
3. Test the dataloader.py
4. Test the dataaugment method
5. Try to forward once in yolox
6. Training and test in 3d-yolox model
7. Evalution 3d-yolox model
8. Add FaFPN in yolo fpn
9. Add transformer in yolo head 

### 2021/10/7 work
1. Let the code can load yolox model
2. Visualize the yolox_l model's result
3. Fix some bug in utils dir's file

### Todo list
1. Fix the bug in utils dir's file, which bring by yolo3d's config.py
2. Add an FPN layer in neck
3. Try FaFPN
4. Add transformer in yolo head


### 2021/10/9 work
1. Understanding the workflow of yolox_head classes.S

### 2021/10/12 work
1. Change the structure of yolox_head to yolo3d head
2. Test the effect of this class
3. Finish the get_box_infor function of yolo3d_head
4. Fix the box_iou compute function for yolo3d_head
5. Finish the dynamic_k_match function for yolo3d_head
6. Finish the get_assignment function for yolo3d_head
7. Finish the get_loss function for yolo3d_head

### Todo list
1. Test the yolo3d_head
2. Create a training config file for yolo3d_head
3. Create a train file and eval file for yolo3d_head

### 2021/10/13 work
1. Finisht the yolo3d_head's test
2. FInish the yolo3d_head's loss function

### Todo list
1. Create a training config file
2. Create a training script file
3. Create a eval script file
4. Try to add a attention map visualize tool
5. Training the model and test it's result.

### 2021/10/22
1. Finish the train.py file
2. Add a config file
3. Add a model_utils in models
4. Add a yolo3dx top class of yolo3d-yolox model
5. Add a warmupcosine lr schedule(don't test now)

### Todo list
1. Debug for train.py file
2. Finish the test.py file
3. Test the evaluate.py result and try to test the mAP in kitti's way
4. Draw attention map for yolo3d-yolox
5. Make a list about bag of trick and bag of free
6. Try to finetune the yolox model in 3d detection 

### 2021/10/23
1. Fix some basic bug for train.py
2. Fix some bug for config.py
3. Debug for Dataloader.py

### Todo list
1. Change the label's load function for kitti_dataset.py function
2. Change the build_yolo_target function for kitti_bev_utils
3. Still debug for train.py
4. Still optimize for evaluate function
5. 