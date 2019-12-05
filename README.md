# Machine Learning Engineer - Sentiance Application

## Python Database

The relevant code is contained in files: pythonDB.py and pythonDBTest.py. The testing file pythonDBTest.py
is simply used to determine the magnitude that a change in latitude or longitude position generates. 
The code for Question 1 can be executed by running the shell command "python pythonDB.py" from the
terminal in the correct working directory. The required information is obtained by standard input and
all generated out is displayed using standard output. An initial prompt will be displayed asking for 
the desired user to be analyzed. Simply enter 1, 2, or 3 for the desired .csv file corresponding to 
person1, person2, or person3. Then press enter and subequently enter 0 followed by enter to progress 
to the next step. To specify and position to check if the user has visited you will be promted to enter 
a latitude value up to 6 decimal places in precision, then press enter. Then you will be promted to enter 
a longitude value up to 6 decimal places in precision, then press enter. The output will then be generated.

The primary file pythonDB.py consists of two main component: an input parser and a user location history
object, which stores the data and provide functions for interacting and manipulating the data. The user 
location history object stores the data as Panda DataFrames since this data structure can be incorporated 
with SQL databases, csv files, and excel files. The columns of the Panda DataFrames consist of the whole 
degree values and single digit values of the fractional degrees in the latitude and longitude position. 
The remaining irrelevant data is dropped to conserve memory. The input values are split using the decimal. 
If the whole degree values of the input match with any entries in the database then the user is considered 
to have visited that location, if not the they have never visited. If the initial check is satisfied, then 
a similar procedure is applied to the fractional digits until there is no match. Depending on how many digits 
are satisfied, an output statement is generating approximating the distance that the user has visited the 
specifed location. The reasoning behined this techniques is to mimic an array of linked-lists, which is quite 
efficient for searching a database and has order n*log(n) time complexity on average. After each check, the 
irrelevant data is dropped from the table so the memory usage is also n*log(n), but approximatedly doubled 
since there are two tables, latitude and longitude.

## Home-Work-Classifier

The relevant code is contained in the Jupyter notebook file: homeWorkClassifier.ipynb. The primary reason 
for utilizing a Jupyter notebook file was for the ease of use for visualization and for employing the sklearn 
library. The code for Question 2 can be executed by by simply opening the .ipynb file and selecting "run all" 
under the cell tab. A PDF of the code and results can also be seen in homeWorkClassifier.pdf.

The code for Question 2 consists of three main components: a class (KMeansObj) that stores that stores the 
data and provides functions for performing a variety kmeans clustering. an analsyis component where many 
of the clustering functions are applied and the results are plotted to investigate the results, and finally
the full techniques is implemented using person1.csv and the user under investigation and person2 and person3
as the database. The cell[1] imports the required libraries, and cell[2] contains the KMeansObj class, while
cell[3] contains global versions of functions that are contained in the KMeansObj class. Cells[4-6] perform
"hard" kmeans clustering iteratively and determines the average silhouette score over the entire dataset for 
each iteration, which is displayed as a plot. Then, hard kmeans is applied using the number of clusters with 
the highest silhouettte score. Cells[7-9] perform soft soft kmeans clustering and any data point in which the 
model does not have a confidence of atleast 0.99 in any cluster is remove and kmeans is applied again and the 
updated cluster centers is obtained, which is shown in a plot as well as the original cluster centers. 
Cells[10-11] show selected cluster centers from each dataset that confirm the results given in cell[13] for 
person1.csv. Cell[12] constructs a list of KMeanObj object for person2 and person3, which form the database 
of the system. Then cell[13] determines the distance between the cluster centers, which is compared with the 
cluster centers in the databases, if the calculated distance is less than the maximum radius of the cluster in 
the dataset under investigation, then the location is assigned a work location label, else it is assigned a 
home location label.

In this approach, the idea that if the location that a user has vistied is shared with other unrelated users,
then the location is most likely a public area and is assigned a work location label, else it is most likely a 
private area and is assigned a home location label. This is the inductive bias of the model. Kmeans clustering 
using the Euclidean distance metric is used since the data has no labels. Additionally, soft kmeans clustering 
and silhouette score are used to eliminate noisey or wrong datapoints and to determine the optimal number of 
cluster locations. This because the database uses a list of list it runs with a time complexity of n*log(n).

Note: Although I initially constructed a table consisting of postion, day of the week, time of day, and 
duration to be used. I only had time to use only the position as that was the initital testing set since it
could be plotted to verify that the model was working.