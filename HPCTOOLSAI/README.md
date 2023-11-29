BaseLine code: https://machinelearningmastery.com/building-a-regression-model-in-pytorch/
Follow the instructions: https://awesome-archduke-bec.notion.site/Lab-AI-HPC-Tools-e647da3f04dc4e66a40692da0d5f9c27


TRAINING WITH 500 EPOCHS (to increase the execution time of the baseline code)
The training distribution was made with pytorch lightning to ease the process

1 Nvidia A100; BASELINE Training Time: 4 minutes 27 seconds
2 Nodes w/ 2 Nvidia A100 -> DDP; Execution time: 03 minute 30 seconds 
2 Nodes w/ 2 Nvidia A100 -> ZeRO; Execution time: 09 minute 19 seconds 

            DDP     ZeRO
Speedup     1.27     -

In this case we are comparing the execution times of parallelizing a linear  regresion model using 2 different strategyes (DDP and ZeRO). It becames  clear that in this case ZeRO is not a great option for distributing this  specific training process as it increases severly the execution time. ZeRO is a good strategy in terms of memory which is not a concern in training such a small model. DDP shows a bit of improvement as it is an efficient strategy to scale training across multiple GPUs. 
Even with DDP the speedup is not very good with this model and probably the main reason might be the communication Overhead. 

Files of this part;
BASELINE.py lightninDistTrainingDDP.py  lightningZeRO.py    sbatchBASELINE.sh   sbatchLightningDDP_v2.sh    slurm-BASELINE.out  slurm-DDP.out   slurm-ZeRO.out

----------------------------------------------------------------------------------

For education purposes I attempted to distribute the training of a more computationally demanding model to gain a bigger insight into the impact of training efficiency and performance.

BASELINE CODE: https://github.com/pytorch/examples/tree/main/vision_transformer
The number of epochs was reduced to 2 in order to decrease the execution time for this educational experiment. This reduction may result in a poorer model performance, but the primary goal is to explore parallel distribution.

1 Nvidia A100; BASELINE Training Time: 19 minutes 50 seconds
2 Nodes w/ 2 Nvidia A100 -> DDP; Execution time: 07 minute 52 seconds 

            DDP     
Speedup     2.52

Now we can see a good speedup in the training process.

Files of this part;
BASELINE_v2.py  lightningDDP_v2.py  sbatchBASELINE_v2.sh    sbatchLightningDDP_v2.sh    slurm-BASELINE_v2.out   slurm-DDP_v2.out