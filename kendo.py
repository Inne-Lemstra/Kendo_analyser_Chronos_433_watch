""" Read out and visualization of Chronos watch accelerometer data for kendo

code written by Inne Lemstra, 27-07-15. With help of Kasper loopstra.

Module consist of serveral functions the help visualize accelerometer data of the chronos watch. Specialised for aquirement of kendo strike data.

functions:

input_slag -- General functions for recording strikes, able to detect multiple input methods.
make_time_stamp -- calculates time difference in milliseconds
check_timer -- reads out watch till time limit is reached, when mode input_slag is time.
check_beweging -- reads out watch till postion is held, when mode input_slag is hold_still.
check_slagen -- read out watch till amount of strikes is reached, when mode input_slag is aantal_slagen.
convert_signed = converts unsigned intergers to signed inetergers.
lees_acc -- gets current accelerometer values from dongle.
get_idle_parameters -- find out which values dongle returns when watch does not broadcasts.
plot_slag -- visualizes aquired accelerometer in different graphs.
read_camera_frame -- starts video capture from webcam. (under construction)

Dependencies:
eZ430 -- has classes for getting data from dongle.
numpy -- use of C arrays for vector operations.
cv2 -- for video capture. (under development, not yet working)
time -- Getting system time.
matplotlib.pyplot -- plotting of graphs with accelerometer data.

notes:
The dongle might not work every single time. Sometimes one or more of the axis of the watch do not work (value does not change). The best work around is to cancel current function and run it again.
When encountering an error when running input_slag for the first time, try running it two more times. This usually resolves this problem.
Code was written using python 2.7.4 with divisions and print_function imported from future using:
	from __future__ import division
	from __future__ import print_function

"""
from lib import eZ430 
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

def input_slag(stop_threshold, method = 'time'):
	"""Gather accelerometer data from watch, till stop condition is met.
	
	The main function of the module that uses all the other functions beside the plotting. It currently encorporates 3 different methods of aquiering data of kendo strikes. These methods are based on a time limit, holding the sword in an end position or when a certain number of strikes is reached. The stop_threshold is supossed to be a value that shows the program when to terminate. This can have different formats based on the method chosen. The method should be a string that determines what method of data acquiering is needed and therefore acquiering of data is over.

	parameters:
	method -- Either 'time', 'hold_still' or 'aantal_slagen'. deterpending of if a time limit, end position or number of strikes should be used for termination of the funtion.
	stop_threshold (method = 'time') -- should be a scalar with the number of seconds the data should be read.
	stop_threshold (method = 'hold_still') -- should be a list with [x, y, z] values of the end position of a strike.
	stop_threshsold (method = 'aantal_slagen') -- should be a interger with the number of strikes to be done before termination
	"""
	watch = eZ430.watch()
	acc_data = []
	idle = get_idle_parameters(watch)
	time_start = time.time()
	time_now = time.time()

	stop_method = {'time':check_timer ,'hold_still':check_beweging \
		, 'aantal_slagen':check_slagen}

	acc_data = stop_method[method](watch, stop_threshold, \
				       acc_data, idle, time_start)
	watch.stop()
	return acc_data

def make_time_stamp(time_start):
	"""calculate difference in time, returns integer in milliseconds"""
	time_now = time.time()
	return int((time_now - time_start)*1000)

def check_timer(watch, time_limit, acc_data, idle, time_start):
	"""Acquier accelerometer data from watch, till time_limit is reached
	
	Function which is used when method of input_slag is 'time'. Is based on the function time() form the module time. Which uses the system time and not the cpu time. Therefore when acquiring data the system time should be altered.

	parameters:
	watch -- handle of class, from eZ430, of the chronos watch.
	time_limit -- scalar with number of seconds, data should be acquiered.
	acc_data -- pyhton lists in which the new accelerometer can be appended.
	idle -- values of [x, y, z] that the dongle return when no new data is acwuired
	time_start -- output of time.time() that was run at the start of the function, used for making time stamps.
	"""
	time_now = time.time()
	while time_now - time_start < time_limit:
		time_stamp = make_time_stamp(time_start) 
		acc_data = lees_acc(watch, acc_data, idle, time_stamp)
		time_now = time.time()
	return acc_data

def check_beweging(watch, eind_stand, acc_data, idle, time_start):
	"""Acquier accelerometer data from watch, till acc data is between certain values.
	
	Function which is used when method of input_slag is 'hold_still'. It will make a zone of x,y and z values, between wich the watch is considered in its end position. The gathering of data is terminated when 10 consecutive accelerometer datapoints are within the boundries around the eind_stand for all dimensions of the data.

	parameters:
	watch -- handle of class, from eZ430, of the chronos watch.
	eind_stand -- list with values of [x, y, z], for when the function should be terminated.
	acc_data -- pyhton lists in which the new accelerometer can be appended.
	idle -- values of [x, y, z] that the dongle return when no new data is acwuired
	time_start -- output of time.time() that was run at the start of the function, used for making time stamps.

	notes:
	Sytem time is used for making time_stamps. Therefor system time should not be altered when running this function. Except of course for the natural passage of time.
	"""
	time_now = time.time()
	# fix with try exempt and index -1
	# houd er rekening mee dat idle ook zorgt voor held_still
	if eind_stand == 'standard':
		eind_stand = [-60,5,19]
	eind_stand = np.array(eind_stand)
	boundries = np.array([eind_stand - 7, eind_stand + 7])
	laatste_10_metingen = np.zeros(shape = (10,4))
	#acc_data = lees_acc(watch, acc_data, idle, time_stamp)
	
	while len(acc_data) <= 10:
	# while loop om de eerste tien getallen te vullen voor check
		time_stamp = make_time_stamp(time_start)
		acc_data = lees_acc(watch, acc_data, idle, time_stamp)

	laatste_10_metingen[:] = acc_data[-10:]
	#print((laatste_10_metingen[:,0:3] < boundries[0]),\
	#	(laatste_10_metingen[:,0:3]))
	while not((laatste_10_metingen[:,0:3] > boundries[0]).all() and\
	      (laatste_10_metingen[:,0:3] < boundries[1]).all()):
	#[0:3] moet erachter omdat time stap niet mee wordt genomen.
	# het was niet mogelijk om op een makkelijke manier de laatste
	#kolom van een python list (of lists) niet toe te voegen aan een np array
		time_stamp = make_time_stamp(time_start)
		acc_data = lees_acc(watch,acc_data, idle, time_stamp)
		laatste_10_metingen[:] = acc_data[-10:]
	return	acc_data

def check_slagen(watch, max_slagen, acc_data, idle, time_start):
	"""Acquier accelerometer data from watch, till a number of strikes is reached
	
	Function which is used when method of input_slag is 'aantal_slagen'. It uses the function check_beweging to determine is an end_point of a strike is reached. If this is the case, 500 milisecond of cooldown time is started in which accelerometer data is gathered but the function can not be terminated. This is to get the values of data outside of the zone for which check_beweging terminates the function. In contrast to the other methods which return only the acc_data, this function returns also the time_stamps of the moment the end of a strike is registered. For testing purposes or fine tuning. (not yet used)

	parameters:
	watch -- handle of class, from eZ430, of the chronos watch.
	max_slagen -- The number of strikes to be done before, the data acquiering should be terminated.
	acc_data -- pyhton lists in which the new accelerometer can be appended.
	idle -- values of [x, y, z] that the dongle return when no new data is acwuired
	time_start -- output of time.time() that was run at the start of the function, used for making time stamps.

	notes:
	Sytem time is used for making time_stamps and maintaining the cooldown period. Therefor system time should not be altered when running this function. Except of course for the natural passage of time.
	This funtion returns beside the acc_data collected also the time_stamps of the end of a strike
	"""	
	time_now = time.time()
	tijd_eind_slagen = []
	aantal_slagen = 0
	cooldown = 1000
	time_stamp = make_time_stamp(time_start)
	while aantal_slagen < max_slagen:
		acc_data = check_beweging(watch, 'standard',\
			   acc_data, idle, time_start)
		aantal_slagen += 1
		tijd_eind_slagen.append(acc_data[-1][-1])
		print('slag geregistreerd \n', tijd_eind_slagen[-1])
		
		while time_stamp - tijd_eind_slagen[-1] < cooldown:
		# om de acc data tijd te gunnen uit boundries te komen
			time_stamp = make_time_stamp(time_start)
			acc_data = lees_acc(watch, acc_data, idle, time_stamp)
	return acc_data, tijd_eind_slagen


def convert_signed(raw_acc_data):
	"""Convert newly aquired acc data from 2 is complement to signed intergers, 	raw_acc_data should a dictionary"""
	temp_acc_data ={}
	for key, value in raw_acc_data.items():
		if value > 128:
			temp_acc_data[key] = value - 255
		else:
			temp_acc_data[key] = value
	return temp_acc_data

def lees_acc(watch, acc_data, idle, time_stamp):
	"""leest een keer de accelerometer data uit van de chronos watch. Input is het object van de watch, de acc_data tot nu toe, idle waardes en de huigdige time stamp"""
	#acc_data = None
	data = watch.read()
	acc = {'x':ord(data[0]),'y':ord(data[1]), 'z':ord(data[2])}

	if [acc['x'],acc['y'],acc['z']] in [[0,0,0],[0,6,7],idle]: 
		#idle waarde bepalden is een goed pricipe
		# maar [0,0,0] werkt vooralsnog beter. 
		return acc_data
	#Voorkomt registeren data, terwijl geen 
	#nieuwe accerlometer data binnen is gekomen
	acc = convert_signed(acc)
	print("x: %s\ty:%s\tz:%s"%(acc['x'],acc['y'],acc['z'])) 
	#status nieuwe data
	acc_data.append([acc['x'],acc['y'],acc['z'], time_stamp]) 
	return acc_data

def get_idle_parameters(watch):
	"""Determine values dongle provides when no data is broadcasted"""
	short_sample = np.zeros(shape = (10,3), dtype = int)
	for row in range(10):
		acc_data = watch.read()
		short_sample[row,:] =\
		 [ord(acc_data[0]),ord(acc_data[1]),ord(acc_data[2])]
	#geeft de frequenty van de getallen weer,
	# en zoekt daar vervolgens de hoogste uit
	x = np.argmax(np.bincount(short_sample[:,0])) 
	y = np.argmax(np.bincount(short_sample[:,1])) 
	z = np.argmax(np.bincount(short_sample[:,2])) 
	idle = [x,y,z]
	return idle

def plot_slag(acc_slag, save_path = './images/default.png'):
	"""plot accelerometer data in 4 graphs, acc_slag should be list of lists with [x, y, z, timestamp]"""
	acc_slag = np.array(acc_slag)	#zet de data in een matrix
	plt.figure(1)
	
	plt.subplot(411)
	plt.hold(True)
	x = plt.plot(acc_slag[:,3], acc_slag[:,0], 'b', label = 'x')
	y= plt.plot(acc_slag[:,3], acc_slag[:,1], 'g', label = 'y')
	z = plt.plot(acc_slag[:,3], acc_slag[:,2], 'r', label = 'z')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc = 3,\
		 ncol = 3, mode = "expand", borderaxespad = 0. )
	plt.ylabel('Acc')	
	plt.hold(False)
	
	plt.subplot(412)
	plt.plot(acc_slag[:,3], acc_slag[:,0], 'b', label = 'x')
	plt.ylabel('Acc')	
	
	plt.subplot(413)
	plt.plot(acc_slag[:,3], acc_slag[:,1], 'g', label = 'y')
	plt.ylabel('Acc')	
	
	plt.subplot(414)
	plt.plot(acc_slag[:,3], acc_slag[:,2], 'r', label = 'z')

	plt.xlabel('Time(ms)')
	plt.ylabel('Acc')	
	plt.savefig(save_path,bbox_inches='tight')
	plt.show()

def read_camera_frame():
	"""capture video from webcam while getting data, under construction"""
	cap = cv2.VideoCapture(1)

	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    # Our operations on the frame come here
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    # Display the resulting frame
	    cv2.imshow('frame',gray)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	cap.release()
	cv2.destroyAllWindows()
