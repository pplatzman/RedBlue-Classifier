import requests, BeautifulSoup, bs4, json

####I will try to get a list of the relevant URLs, then 'for loop' through them so I don't have to write a different script for each debate.####

##Saving home page as text file.##
res = requests.get('http://www.presidency.ucsb.edu/debates.php')
print type(res)
print res.raise_for_status()
print len(res.text)
print (res.text[:250])
home = open('DebateTextHomePage.txt', 'wb')
for chunk in res.iter_content(1000000):
    home.write(chunk)
home.close()

##Opening newly created home page text file.##
homepage = open('DebateTextHomePage.txt')
homepageHTML = bs4.BeautifulSoup(homepage.read(), "html.parser")

##Defining and executing function to retrieve all of the home page's URLs.##
urllist = []
def geturl(x):
	global urllist
	for link in x.find_all('a'):
		urllist.append(link.get('href'))
		
geturl(homepageHTML)
print urllist

##Now that I have my URL list, I want to find the specific elements of the list that I want so that I can extract them.##
print len(urllist)
a = urllist.index("http://www.presidency.ucsb.edu/ws/index.php?pid=116995") #Last Democratic debate link. Output is 40.#
b = urllist.index("http://www.presidency.ucsb.edu/ws/index.php?pid=110903") #First Democratic debate link. Output is 48.#
c = urllist.index("http://www.presidency.ucsb.edu/ws/index.php?pid=115148") #Last Republican debate link. Output is 49.#
d = urllist.index("http://www.presidency.ucsb.edu/ws/index.php?pid=110757") #First Republican debate link. Output is 67.# 
print a,b,c,d
DEMdebates2016list = urllist[a:b+1]
REPUBdebates2016list = urllist[c:d+1]
print DEMdebates2016list
print REPUBdebates2016list

##Now I'm going to download every debate transcript as a text file, separated by Democratic and Republican transcripts.
##This also creates a list of the file names, which I will need in the HTML parsing section.##

DEMdebatefiles = []

def downloadDEMdebate(x):
	global DEMdebatefiles
	for i in x:
		res = requests.get(i)
		print type(res)
		print res.raise_for_status()
		print len(res.text)
		print (res.text[:250])
		playFile = open('DEM'+i[-6:], 'wb') ##Naming each file by its 'pid' as indicated in its URL.##
		for chunk in res.iter_content(1000000):
		    playFile.write(chunk)
		playFile.close()
		DEMdebatefiles.append('DEM'+i[-6:])

downloadDEMdebate(DEMdebates2016list)

print DEMdebatefiles


REPUBdebatefiles = []

def downloadREPUBdebate(x):
	global REPUBdebatefiles
	for i in x:
		res = requests.get(i)
		print type(res)
		print res.raise_for_status()
		print len(res.text)
		print (res.text[:250])
		playFile = open('REP'+i[-6:], 'wb') ##Naming each file by its 'pid' as indicated in its URL.##
		for chunk in res.iter_content(1000000):
		    playFile.write(chunk)
		playFile.close()
		REPUBdebatefiles.append('REP'+i[-6:])

downloadREPUBdebate(REPUBdebates2016list)

print REPUBdebatefiles

####Now I will try to extract the relevant text portions of each transcript's HTML file.##

#str1 = "".join(str(x) for x in DEMdebatefiles) #This was a failed attempt to convert a list into a string.#

def textofDEMdebates(x):
	for i in x:
		DEMtranscriptdebate = open(i)
		res = bs4.BeautifulSoup(DEMtranscriptdebate.read(), "html.parser")
		elems = res.select('p')
		print type(elems)
		print len(elems)
		#print str(elems[0])
		#print type(elems[0])
		textfield = elems[0].getText()
		#print textfield
		#print type(textfield)

		#Alternative way to format 'textfield,' which ends up going into the json file.#
		#textfield = str(elems[0])

		with open('%s.json' % (i), 'w') as outfile:
		    json.dump(textfield, outfile)

textofDEMdebates(DEMdebatefiles)		


def textofREPUBdebates(x):
	for i in x:
		REPUBtranscriptdebate = open(i)
		res = bs4.BeautifulSoup(REPUBtranscriptdebate.read(), "html.parser")
		elems = res.select('p')
		print type(elems)
		print len(elems)
		#print str(elems[0])
		#print type(elems[0])
		textfield = elems[0].getText()
		#print textfield
		#print type(textfield)

		#Alternative way to format 'textfield,' which ends up going into the json file.#
		#textfield = str(elems[0])

		with open('%s.json' % (i), 'w') as outfile:
		    json.dump(textfield, outfile)

textofREPUBdebates(REPUBdebatefiles)