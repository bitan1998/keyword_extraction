# importing required modules

import PyPDF2
import collections
import pytesseract
import pandas as pd
import numpy as np
import re

filename ='E:/JavaBasics-notes.pdf' 
pdfFileObj = open(filename,'rb')  
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
num_pages = pdfReader.numPages
count = 0
text = ""
                                                            
while count < num_pages:                       #The while loop will read each page
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()

if text != "":
    text = text
    
else:
    text = pytesseract.process('http://bit.ly/epo_keyword_extraction_document', method='tesseract', language='eng')

text = text.encode('ascii','ignore').lower()
regex = b'<title>(,+?)</title>'
from gensim.summarization import keywords
import warnings
values = keywords(text=text,split='\n',scores=True)
data = pd.DataFrame(values,columns=['keyword','score'])
data = data.sort_values('score',ascending=False)
print(data.head(30))




