import urllib.request
import sys
import os

# Read file and gets lines
txtfile = open(sys.argv[1], "r")
urls = txtfile.readlines()

# Create directory to store images
if not os.path.exists('raw'):
    os.mkdir('raw')

# Error file
errfile = open("downloadErr.txt", "w")

# Add itr in case the files have similar names
for itr, url in enumerate(urls):
    try:
        urlseparated = url.split('/')

        # only want last part of link
        urlpostfix = urlseparated[-1]

        # file names cant end with \n
        if urlpostfix[-1] == '\n':
            urlpostfix = urlpostfix[:-1]

        # file names limited to 255
        if len(urlpostfix) >= 255:
            urlpostfix = urlpostfix[-255:]
        
        # Add header to bypass 403
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, "raw/" + str(itr) + urlpostfix)
        # urllib.request.urlretrieve(url, "images/" + str(itr) + urlpostfix)
    except:
        # Log error
        errfile.write(url + '\n')
        print("Error for " + url)

errfile.close()