# predict the ligand pathway of Nuclear receptor

# Input
To prepare the input for model prediction, MM/GBSA decomposition is proposed to generate the MIEC spectrum. Before predicting the pathway probability, sequence alignment should be conducted according to the MSA provided in the module_seq_align.fa file. After aligning to the template, zeros should be used to fill the gaps of the MIEC spectrum. Currently, only ligands targeting the 21 NRs recorded in the MSA file can be used for the prediction. See data.csv for an input example.
![Image](https://github.com/SutongXiang/MIEC_ResMLP/blob/main/align.png)


# Output
The model outputs the probability distribution of eight paths.
![Image](https://github.com/SutongXiang/MIEC_ResMLP/blob/main/pathway1.png)
