# predict the ligand pathway of Nuclear receptor

# Input
First, you need  to calculate the MIECs spectrum by MM/PBSA method. In Here, we calculated the MIECs by MMPBSA.py in AMBER/20.
![Image](https://github.com/SutongXiang/MIEC_ResMLP/blob/main/align.png)
Align the MIECs matrix according to module sequence alignment that shown in the module_seq_align.fa, and the gaps are filled with zero. See data.csv for example data.

# Output
The model outputs the probability distribution of eight paths.
![Image](https://github.com/SutongXiang/MIEC_ResMLP/blob/main/pathway1.png)
