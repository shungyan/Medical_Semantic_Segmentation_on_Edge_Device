# Medical_Semantic_Segmentation_on_Edge_Device
This repo shows the code and training process of Final Year project

Overview of this project: 
Optimize the performance of U-Net on Medical Semantic Segmentation and implement it on edge device (Google Coral Dev Board Mini).

2 datasets are trained with 2 different U-Net. 
1. Brain Tumor Segmentation Dataset is a 3D MRI brain tumor dataset and it is trained with 3D U-Net.
2. Lung Segmentation Dataset is a 2D Chest X-ray dataset and it is trained with 2D U-Net.

The performance of U-Net is optimize by different methods which include dropout regularization and early stopping.

Performance of different pooling layers on 2D U-Net and 3D U-Net is evaluated.

Result of different pooling layer in 3D U-Net
![image](https://github.com/shungyan/Medical_Semantic_Segmentation_on_Edge_Device/assets/84812149/7c28e592-2984-4fb7-9738-7cb9462fc5e4)
Result of different pooling layer in 2D U-Net
![image](https://github.com/shungyan/Medical_Semantic_Segmentation_on_Edge_Device/assets/84812149/961befb5-40de-4e99-9bb7-6b52940f3f11)

Result of 3D U-net compared to other models
![image](https://github.com/shungyan/Medical_Semantic_Segmentation_on_Edge_Device/assets/84812149/0d981f0d-ea36-48d9-81e1-d7117c0e0a82)

Result of 2D U-net compared to other models
![image](https://github.com/shungyan/Medical_Semantic_Segmentation_on_Edge_Device/assets/84812149/867dc5dd-eda3-4cf2-801d-203d4f7567fc)

2D U-Net is compressed and implemented on Google Coral Dev Board Mini.
Performance of 2D U-Net on Google Coral Dev Board Mini.
![Uploading image.png…]()


Conclusion:
In this project, I find out the performance of pooling layer depends on the nature of the dataset. 
Max Pooling performed best in 3D U-Net and Average Pooling performed best in 2D U-Net. 
Brain Tumor Segmentation Dataset (BRATS) is a class-imbalanced dataset which means that some MRI scans are occupied by background.
Max pooling perform well to capture the small tumor while average pooling will ignore the small tumor.
Lung Segmentation Dataset does not have class imbalance data so average pooling better.

