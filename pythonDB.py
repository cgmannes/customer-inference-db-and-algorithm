'''
Author: Christopher Mannes

Look-up system that generates a location history object, userLocHist for a given user
from a .csv file. The userLocHist object contains pandas dataframes as its member data 
and provides a set of functions for intacting with and manipulating the data.
'''

'''Import libraries'''
import sys
import string
import math as m
import numpy as np
import pandas as pd


'''Class UserLocHist generates objects containing dataframes of lat. and long. values.'''
class UserLocHist():

    def __init__(self): 
        self.locHistDF = pd.DataFrame()
        self.latCoordHistDF = pd.DataFrame()
        self.longCoordHistDF = pd.DataFrame()
        self.locationDF = pd.DataFrame()

    '''
    The function import_user_data reads the specified csv file from the
    current working directory as a Pandas DataFrame.
    '''
    def import_user_data(self, idNumber):
        userID = "person" + str(idNumber) + ".csv"
        self.locHistDF = pd.read_csv(userID, 
                                    sep = ';',
                                    names = ['latitude', 'longitude', 'start_time(YYYYMMddHHmmZ)', 'duration(ms)'])
        self.locHistDF = self.locHistDF.iloc[1:,]
        self.locHistDF = self.locHistDF.drop(['start_time(YYYYMMddHHmmZ)', 'duration(ms)'], axis = 1)

    def print_data(self):
        print(self.locHistDF)

    '''
    Generates columns containing each digit in the latitude position and
    drops irrelevant data to conserve memory.
    '''
    def generate_latitude_df(self):
        self.latCoordHistDF = pd.DataFrame(self.locHistDF.latitude.str.split('.',1).tolist(),
                                            columns = ['Degrees','Decimals'])

        self.latCoordHistDF['lat_column1'] = pd.DataFrame(self.latCoordHistDF['Degrees'].str[0:])
        self.latCoordHistDF['lat_column2'] = pd.DataFrame(self.latCoordHistDF['Decimals'].str[0:1])
        self.latCoordHistDF['lat_column3'] = pd.DataFrame(self.latCoordHistDF['Decimals'].str[1:2])
        self.latCoordHistDF['lat_column4'] = pd.DataFrame(self.latCoordHistDF['Decimals'].str[2:3])
        self.latCoordHistDF['lat_column5'] = pd.DataFrame(self.latCoordHistDF['Decimals'].str[3:4])
        self.latCoordHistDF['lat_column6'] = pd.DataFrame(self.latCoordHistDF['Decimals'].str[4:5])
        self.latCoordHistDF = self.latCoordHistDF.drop(['Degrees','Decimals'], axis = 1)

        self.latCoordHistDF.replace('', 0, inplace = True)

    '''
    Generates columns containing each digit in the longitude position and
    drops irrelevant data to conserve memory.
    '''
    def generate_longitude_df(self):
        self.locHistDF.sort_values('longitude')
        self.longCoordHistDF = pd.DataFrame(self.locHistDF.longitude.str.split('.',1).tolist(),
                                            columns = ['Degrees', 'Decimals'])

        self.longCoordHistDF['long_column1'] = pd.DataFrame(self.longCoordHistDF['Degrees'].str[0:])
        self.longCoordHistDF['long_column2'] = pd.DataFrame(self.longCoordHistDF['Decimals'].str[0:1])
        self.longCoordHistDF['long_column3'] = pd.DataFrame(self.longCoordHistDF['Decimals'].str[1:2])
        self.longCoordHistDF['long_column4'] = pd.DataFrame(self.longCoordHistDF['Decimals'].str[2:3])
        self.longCoordHistDF['long_column5'] = pd.DataFrame(self.longCoordHistDF['Decimals'].str[3:4])
        self.longCoordHistDF['long_column6'] = pd.DataFrame(self.longCoordHistDF['Decimals'].str[4:5])
        self.longCoordHistDF = self.longCoordHistDF.drop(['Degrees','Decimals'], axis = 1)

        self.longCoordHistDF.replace('', 0, inplace = True)

    '''
    The function calculate_dist compares the user input latitude and longitude
    position to the latitude and longitude positions in the database in a digit
    by digit sequence from the left. If the whole degrees do notmatch then the 
    user has never visited and the calculation ends. Otherwise, the procedure
    is repeated until it fails and outputs an approximation to the visited point.
    '''
    def calculate_dist(self, latString, longString):

        latString = latString.split('.') 
        longString = longString.split('.') 

        self.locationDF = pd.concat([self.latCoordHistDF, self.longCoordHistDF], axis = 1)

        if len( self.locationDF[self.locationDF['lat_column1'].str.match(latString[0])].index ) > 0:
            self.locationDF = self.locationDF[self.locationDF['lat_column1'].str.match(latString[0])]

            if len( self.locationDF[self.locationDF['long_column1'].str.match(longString[0])].index ) > 0:
                self.locationDF = self.locationDF[self.locationDF['long_column1'].str.match(longString[0])]

            else:
                sys.stdout.write("The specified location has not been visisted by the user. \n")
                return

        else:
            sys.stdout.write("The specified location has not been visisted by the user. \n")
            return


        counter = 2

        for i in range ( min( len(latString[1]), len(longString[1]) ) - 1 ):

            if len( self.locationDF[self.locationDF['lat_column' + str(counter)].str.match(latString[1][i])].index ) > 0:
                self.locationDF = self.locationDF[self.locationDF['lat_column' + str(counter)].str.match(latString[1][i])]

                if len( self.locationDF[self.locationDF['long_column' + str(counter)].str.match(longString[1][i])].index ) > 0:
                    self.locationDF = self.locationDF[self.locationDF['long_column' + str(counter)].str.match(longString[1][i])]
                    counter += 1

        # Function for generating the output approximation.
        def switch_function(arg):
            outputString = "User has visited the specified location within approximately "
            switch = {
                3: outputString + "4000 meters",
                4: outputString + "400 meters",
                5: outputString + "40 meters",
                6: outputString + "4 meters",
                7: outputString + "0.4 meters"
            }
            return switch.get(arg, "The specified location has not been visisted by the user. \n")

        sys.stdout.write(switch_function(counter) + '\n\n')

# Instantiation of the object.
userLocHist = UserLocHist()

''' Function for calculating changes in latitude and longitude position. '''
def get_distance(lat1, lon1, lat2, lon2):
    R = 6731000
    distLat = m.radians(lat2 - lat1)
    distLon = m.radians(lon2 - lon1)
    a = (m.sin(distLat/2))**2 + m.cos(m.radians(lat1))*m.cos(m.radians(lat2))*((m.sin(distLon/2))**2)  
    c = m.atan2(m.sqrt(a), m.sqrt(1 - a))
    d = round(R*c, 0)

    return d

'''
Entry point for program. 
Input parser for look-up system.
'''
def main():

    dataLoaded = False
    while True:
        try:
            sys.stdout.write("Enter userID (integer 1-3) to generate look up table for that user,\n" 
                            "else enter 0 to specify a current user location, or q to quit: \n")
            inputID = sys.stdin.readline()
            inputID.strip()
            inputID.lower()

            if inputID[0] == 'q':
                break
            elif inputID[0] == 'p' and dataLoaded == True:
                inputID = -1
                userLocHist.print_data()
            elif inputID[0] == 'p':
                inputID = -1

            idNumber = int(inputID)
        except:
            sys.stdout.write("Error: Invalid input. \n")

        if idNumber > 0 and idNumber < 4:
            userLocHist.import_user_data(idNumber)
            userLocHist.generate_latitude_df()
            userLocHist.generate_longitude_df()
            dataLoaded = True
        elif idNumber == 0 and dataLoaded == False:
            sys.stdout.write("Error: Specify user first. \n")
        elif idNumber == 0 and dataLoaded == True:
            counter1 = 0
            while True:
                try:
                    if counter1 == 0:
                        sys.stdout.write("Enter current latitude as a signed float to 6 decimal palces: \n")
                        latFloat = sys.stdin.readline()
                        latFloat.strip()

                        if latFloat[0] == 'b':
                            break

                        counter1 += 1
                        
                    elif counter1 == 1:
                        sys.stdout.write("Enter current longitude as a signed float to 6 decimal palces: \n")
                        longFloat = sys.stdin.readline()
                        longFloat.strip()

                        if longFloat[0] == 'b':
                            break

                        userLocHist.calculate_dist(latFloat, longFloat)
                        break
                except:
                    sys.stdout.write("Error: Invalid input. \n")



if __name__ == "__main__":

    main()

