# An example for Neural Network Pruning with LRP and other measures, based on 2D toy datasets.
[Link to Preprint](reference link: https://arxiv.org/abs/1912.08881)

![The impact Neural Network Pruning to the model's decision boundary](toy-figure-mk2-test.png)
**Qualitative comparison of the impact of the pruning criteria on the models’ decision function on three toy datasets.**
*1st column:* scatter plot of the training data and decision boundary of the trained model,
*2nd column:* data samples randomly selected for computing the pruning criteria,
*3rd to 6th columns:* changed decision boundaries after the application of pruning w.r.t. different criteria.




![The influence of the number of referenece samples to the performance of the pruned model](output/combined-processed.png)
**Pruning performance comparison of criteria depending on the number of reference samples per class used for criterion computation.** 
*1st row:* Model evaluation on the training data.
*2nd row:* Model evaluation on unseen samples, which have also been used for the computation of pruning criteria.
*Columns:* Results over different datasets. Solid lines show the average post-pruning performance of the models pruned w.r.t. to the evaluated criteria “weight” (black), “taylor” (blue), “grad(ient)” (green) and “lrp” (red) over 50 repetitions ofthe experiment. The dashed black line indicates the model’s evaluation performance without pruning. Shaded areas around the lines show the standard deviation over the repetition of experiments.


