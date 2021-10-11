# NILM-FinalTermPaper

This work has been made for the main reasearch of the author's Final Term Paper. Therefore the research and the codes are still in development.

The REED database was chosen for the research. These datas are free and are about power consumption of different appliances in different houses [1].
Moreover, this code is an execution of an algorithm proposed by J. Yu, . Gao, Y. Wu et al in a paper of theirs submitted to Applied Sciences in 2019. That paper is called "Non-Intrusive Load Disaggregation by Linear Classifier Group Considering Multi-Feature Integration" [2].


The codes have been organized in four files: main.py, functions.py, Load.py and draft.py.

The main.py is the main file that will execute every function necessary to run the algorithm. 
The functions.py file is where is coded every single step of the algorithm to be run. 

Futhermore, the Load.py is file used to ease the access to the database. This file have a class called Appliance, since the main purpose of this research is to lead with power signals of appliances. Nonetheless, the Load.py file could be improved to another datas with differents natures.

The last file (draft.py) is a combination of some drafts made to test what could be done with the dataset and such nature of problem. 


## Bibliography
[1] J. Z. Kolter and M. J. Johnson; Redd: A public data set for energy disaggregation research, Workshop on data mining applications in sustainability (SIGKDD), San Diego, CA, vol. 25, pp. 59–62, 2011.

[2] J. Yu, Y. Gao, Y. Wu, D. Jiao, C. Su, and X. Wu; Non-intrusive load disaggregation by linear classifier group considering multi-feature integration. Applied Sciences, vol. 9, no. 17, p. 3558, 2019.

