import requests
import json
import bs4
from bs4 import BeautifulSoup
import urllib2
import lxml
import os

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

## Creating Democrat documents for each line

DEMdebates2016list = urllist[a:b+1]
REPUBdebates2016list = urllist[c:d+1]
print DEMdebates2016list
print REPUBdebates2016list
arry = []
x = 1

for i in DEMdebates2016list:
    soup = BeautifulSoup(urllib2.urlopen(i), "lxml")
    for tag in soup.find_all('p'):
        arry.append(tag)
        for line in arry:
            fname = "dem_%s.txt" % (x)
            outpath = os.path.abspath(fname)
            with open(outpath, 'w') as f:
                f.write(line.text.encode('utf-8') + '\n')
            x+=1
