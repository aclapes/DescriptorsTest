DescriptorsTest
===============

This program read some RGB-D pcd files, separated into train (higher-resolution models) and test instances. The paths to this pcds need to be specified by argument, as many others. The only parameter that needs to be hardcoded as a #define in the code is the kind of description, e.g. FPFH_DESCRIPTION. The information about the program arguments is the main function as commentaries and replicated below:

    //
    // The system need to be parametrized
    //
    
    // ./descriptors_test -m "book,cup,dish,grimper,scissors,pillbox,tetrabrick" -v "3,1,3,2,2,5,3" -g 7,3
    
    //
    // Optional arguments came later:
    //
    
    // Specify the output paths
    //   -O "summary.txt,scores.txt"
    // Specify the arguments for the computation of scores
    //   -S -t "../../data/Models/" -e "../../data/Test/" -d 0 -l "0.01,0.02,0.03" -n "0.03,0.05,0.07" -f "0.05,0.075,0.10,0.125,0.15,0.20" -p 0,1,0.1
    // Choose the validation
    // -V (-i "3,4")
    // -Vr "0,0.025,1" -i "3,4", validation with rejection
    
    
Detailed information:
    
a) Data loading: 
The number of views are specified in an argument (-v) together with the models' names (-m). At the end, "-g" refers to the grid dimensions. In the example above, h x w = 7 x 3.
 
b) Data loading and computation of classification scores
 
The option -S specifies the scores are required to be computed. This will use the data in "../../data/Models/" or "../../data/Test/". To those paths, it will be added "PCDs/" and the model name, e.g. "book/", in order to locate the files. Therefore, the PCDs are looked for in a directory path of this form: "../../data/Models/PCDs/book/". It should contain as many PCDs as the {height of the grid} x {width of the grid} x {the number of views}. 
  
 Cause this is a costly procedure, it can be executed independently and stored the results into file. These can be used later as a pre-computed set of scores in the two kinds of validation. In that case, the "-O" parameter has to specify the correct path to results files.
 
c.1) Validation without rejection threshold
 Simply use the option -V. Additionaly, and optionally, a set of models to be ignored in the calculation can be specified by "-i", e.g. '-i "3,4"'.
 
c.2) Validation WITH rejection threshold
 Use the option -Vr. Additionaly, you should specify the limits and step of the rejection threshold to validate in the following format: <start,end.step>. For instance: 0,1,0.025. Optionally, also a set of models to be ignored in the calculation can be specified by "-i", e.g. '-i "3,4"'.