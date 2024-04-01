File -> [Wrapper.py]
Line 67,68,69,70
We need to make directory manually inside "Data" to save the generated patches
Data->
      Train
      Trained_ ->
                 Patch_A
                 Patch_B
                 Img
      Val
      Validated_ ->
                   Patch_A
                   Patch_B
                   Img

File -> Test_Supervised.py

In line 72, write image name manually to read

File -> Test_Unsupervised.py

In line 81, write image name manually to read

Run Train_Supervised and Train_Unsupervised to train the models
Make folder Checkpoints