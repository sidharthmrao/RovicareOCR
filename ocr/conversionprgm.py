# IMPORTS

from __future__ import print_function
from google.cloud import vision
from google.oauth2 import service_account
from pdf2image import convert_from_path
import tabula.io
import pandas as pd
import pytesseract
from pytesseract import Output
import cv2
import os, subprocess
import json


import configparser
import sys
args = sys.argv[1:]

config = configparser.ConfigParser()
config.read("config.ini")
keys = config["KEYS"]
store = config["STORAGE"]

#### VISION DEF

def cloudvision(conf = False):

    credentials = service_account.Credentials.from_service_account_file(keys["visionsa"])
    client = vision.ImageAnnotatorClient(credentials=credentials)
    
    path = 'pdfimage-1.jpg'

    with open(path, 'rb') as image_file:

        content = image_file.read()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
    

    responsetext = response.text_annotations[0].description

    confidences = []

    
    
    total = 0
    for conf in confidences:
        total+=conf

    #avg = total/ len(confidences)

    if  conf:
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                #print(f'Block Confidence: {block.confidence}')
                confidences.append(block.confidence)

    responsesplit = responsetext.split("\n")

    newresponses = []

    for line in responsesplit:
        line = line.replace("_", "")
        newresponses.append(line.split(":"))

    infoheaders = ["Patient current location", "Phone number", "Contact Name",  "DOB", "Age", "Marital Status", "Employment Status", "Referral Source", "Source"]
    newheaders  = ["Location", "Phone Number", "Contact Name",  "Birth Date", "Age", "Marital Status", "Employment Status", "Referral Source", "Referral Source"]
    information = {}

    information["Patient's First Name"] = ""
    information["Patient's Last Name"] = ""

    for header in newheaders:
        information[header] = ""



    for word in newresponses:

        

        if word[0] == "Patient First Name/Last Initial" or word[0] == "Patient's Full Name":
            
            firstname, lastname = word[1].split()[0], word[1].split()[-1]

            information["Patient's Full Name"] = " ".join(word[1].split())

            information["Patient's First Name"] = firstname
            information["Patient's Last Name"] = lastname

        if word[0] == "Employment Status":
            information["Employment Status"] = " ".join(word[-1].split())

        else:

            for header in range(len(infoheaders)):
                if infoheaders[header] == word[0]:
                    information[newheaders[header]] = " ".join(word[1].split())

    return information

    
#### TABULA

def tabled(pdf_path):

    dfs = tabula.io.read_pdf(pdf_path, pages='all')

    df = pd.DataFrame(dfs[0])

    labels = ["Resident Name", "Admission Date", "Init. Adm. Date", "Orig.Adm.Date", "Resident #", "Previous address", "Previous Phone #", "Sex", "Birthdate", "Age", "Marital Status", "Religion", "Race", "Occupation(s)", "Primary Lang.", "Admission Location", "Birth Place", "Citizenship", "Social Security #"]
    labeled = {}

    for column in df:
        #print(df[column])
        for row in range(len(df[column])):
            for label in labels:
                if df[column][row] == label:
                    labeled[label] = df[column][row+1]

    #for key in labeled:
        #print(f"{key} : {labeled[key]}")

    return labeled


#### TESSERACT

def tessertxt(img):

    #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    d = pytesseract.image_to_data(img, output_type=Output.DICT)

    e = pytesseract.image_to_string(img)

    import numpy as np
    #import image
    image = cv2.imread('pdfimage-1.jpg')

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.waitKey(0)

    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    cv2.waitKey(0)

    kernel = np.ones((7,20), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.waitKey(0)

    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    cvbox = []

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        #print(x, y, w, h)
        cvbox.append([x,y,w,h])

        # Getting ROI
        roi = image[y:y+h, x:x+w]

        # show ROI
        # cv2.imshow('segment no:'+str(i),roi)
        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        cv2.waitKey(0)

    cv2.waitKey(0)


    #######################################################################


    poplist = []
    iter = 0

    for x in range(len(d["text"])):
        if len(d["text"][x])<=1 and (d["text"][x] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&" or d["text"][x] in ['']):
            poplist.append(x-iter)
            iter+=1

    for x in poplist:
        for column in d.keys():
            d[column].pop(x)

    textbox = []

    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h, t) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i], d['text'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        textbox.append([x,y,w,h,t])

    cv2.imwrite('cv2testimg.jpg',img)
    cv2.waitKey(0)


    collapsedboxes = []
    possibleboxes = []

    for cv in range(len(cvbox)):

        cvx, cvy, cvw, cvh = cvbox[cv]
        cvl = cvx
        cvr = cvx+cvw
        cvu = cvy
        cvd = cvy+cvh

        if cvy <840:
            collapsedboxes.append([cvx, cvy, cvw, cvh, ""])

        for tv in textbox:
            tvx, tvy, tvw, tvh, tvt = tv
            tvxmid = tvx+(tvw/2)
            tvymid = tvy+(tvh/2)
            if tvxmid > cvl and tvxmid < cvr and tvymid > cvu and tvymid < cvd and tvy<840:
                collapsedboxes[-1][4] += " " + tvt


    labels = ["Case No", 'Patient (first name, middle name, last name)', 'Attending Doctor', 'Religion', 'Soc Sec No', "Co-Attending Doctor", "P. Language", "County", "Race & Ethnicity", "Preferred Pronouns", "Zip", "City", "ST", "Sex", "BirthDaie", "Street", "Date", "Patient Phone", "Marital", "Religion", 'Admit Date/Time']
    newboxes = {}

    blacklisted = ["Zip", "Relation", "Policy No.", "Policy No"]

    for box in range(len(collapsedboxes)):
        collapsedboxes[box][4] = " ".join(collapsedboxes[box][4].split())


    def sorter(li):
        return li[1]

    for box in range(len(collapsedboxes)):
        possibleboxes = []
        cvx, cvy, cvw, cvh, cvt = collapsedboxes[box]
        cvl = cvx
        cvr = cvx+cvw
        cvu = cvy
        cvd = cvy+cvh

        if collapsedboxes[box][4] == 'SocSecNo Policy No. Policy No':
            collapsedboxes[box][4] = 'Soc Sec No'

        if collapsedboxes[box][4] in labels:
            newboxes[collapsedboxes[box][4]] = ""

            for tv in collapsedboxes:
                tvx, tvy, tvw, tvh, tvt = tv
                tvxmid = tvx+(tvw/2)
                tvymid = tvy+(tvh/2)
                if tvxmid > cvl and tvxmid < cvr and tvy > cvy:
                    if tvt not in newboxes.values() and tvt not in newboxes.keys() and tvy<840:
                        possibleboxes.append([tvx, tvy, tvw, tvh, tvt])
        
            try:
                possibleboxes.sort(key = sorter)
                newboxes[collapsedboxes[box][4]] = possibleboxes[0][4]
            except:
                pass


    for line in newboxes.keys():
        for nope in blacklisted:
            newboxes[line] = newboxes[line].replace(nope, "").strip()

    collapsedboxes = []
    possibleboxes = []

    for cv in range(len(cvbox)):

        cvx, cvy, cvw, cvh = cvbox[cv]
        cvl = cvx
        cvr = cvx+cvw
        cvu = cvy
        cvd = cvy+cvh

        if cvy >840:
            collapsedboxes.append([cvx, cvy, cvw, cvh, ""])

        for tv in textbox:
            tvx, tvy, tvw, tvh, tvt = tv
            tvxmid = tvx+(tvw/2)
            tvymid = tvy+(tvh/2)
            if tvxmid > cvl and tvxmid < cvr and tvymid > cvu and tvymid < cvd and tvy>840:
                collapsedboxes[-1][4] += " " + tvt


    mainlabels = ["Prim. Insurance", "2nd Primary Insurance", "Secondary Insurance"]
    sublabels = ["Group No.", "Policy No.", "Subscriber Name", "Subscriber DOB", "Verified", "Relation", "Eligibility Date"]
    bottomnewboxes = {}

    blacklisted = ["Zip", "Relation", "Policy No.", "Policy No"]

    for box in range(len(collapsedboxes)):
        collapsedboxes[box][4] = " ".join(collapsedboxes[box][4].split())
        #print(collapsedboxes[box])


    def sorter(li):
        return li[1]

    for box in range(len(collapsedboxes)):
        possibleboxes = []
        cvx, cvy, cvw, cvh, cvt = collapsedboxes[box]
        cvl = cvx
        cvr = cvx+cvw
        cvu = cvy
        cvd = cvy+cvh

        if collapsedboxes[box][4] in mainlabels:
            bottomnewboxes[collapsedboxes[box][4]] = ""

            for tv in collapsedboxes:
                tvx, tvy, tvw, tvh, tvt = tv
                tvxmid = tvx+(tvw/2)
                tvymid = tvy+(tvh/2)
                if tvxmid > cvl and tvxmid < cvr and tvy > cvy:
                    if tvt not in bottomnewboxes.values() and tvt not in bottomnewboxes.keys() and tvy>840:
                        possibleboxes.append([tvx, tvy, tvw, tvh, tvt])
        
            try:
                possibleboxes.sort(key = sorter)
                bottomnewboxes[collapsedboxes[box][4]] = possibleboxes[0][4]
            except:
                pass
        
        if collapsedboxes[box][4] in sublabels:

            suffix = "A"

            if abs(collapsedboxes[box][1]-1234) < 30:
                suffix = "A"
            elif abs(collapsedboxes[box][1]-1360) < 30:
                suffix = "B"
            elif abs(collapsedboxes[box][1]-1486) < 30:
                suffix = "C"

            bottomnewboxes[collapsedboxes[box][4]+suffix] = ""

            for tv in collapsedboxes:
                tvx, tvy, tvw, tvh, tvt = tv
                tvxmid = tvx+(tvw/2)
                tvymid = tvy+(tvh/2)
                if tvxmid > cvl and tvxmid < cvr and tvy > cvy:
                    if tvt not in bottomnewboxes.values() and tvt not in bottomnewboxes.keys() and tvy>840:
                        possibleboxes.append([tvx, tvy, tvw, tvh, tvt])
        
            try:
                possibleboxes.sort(key = sorter)
                bottomnewboxes[collapsedboxes[box][4]+suffix] = possibleboxes[0][4]
            except:
                pass


    for line in bottomnewboxes.keys():
        for nope in blacklisted:
            bottomnewboxes[line] = bottomnewboxes[line].replace(nope, "").strip()

    newestboxes = {}
    seta = {
        "Insurance" : "",
        "Group No." : "",
        "Policy No." : "",
        "Subscriber Name" : "", 
        "Subscriber DOB" : "",
        "Verified" : "",
        "Relation" : "",
        "Eligibility Date" : ""
    }
    setb = {}
    setc = {}

    for line in bottomnewboxes.keys():
        
        if line == mainlabels[0]:
            seta["Insurance"] = bottomnewboxes[line]
        elif line == mainlabels[1]:
            setb["Insurance"] = bottomnewboxes[line]
        elif line == mainlabels[2]:
            setc["Insurance"] = bottomnewboxes[line]
        else:
            if line[-1] == "A":
                seta[line[:-1]] = bottomnewboxes[line]
            elif line[-1] == "B":
                setb[line[:-1]] = bottomnewboxes[line]
            elif line[-1] == "C":
                setc[line[:-1]] = bottomnewboxes[line]

    newestboxes["Prim. Insurance"] = seta
    newestboxes["2nd Primary Insurance"] = setb
    newestboxes["Secondary Insurance"] = setc
    
    return newboxes, newestboxes


def readmed(pdf_path):
    ### IMAGE CONVERSION

    # Store Pdf with convert_from_path function
    """
    poppler_path = r''

    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    images[0].save('pdfimg' +'.jpg', 'JPEG')
    """
    #pdftoppm_path = r"execsrc/poppler-0.67.0_x86/bin/pdftoppm.exe"
    #pdftoppm_path = r"usr/bin/pdftoppm"
    process = subprocess.Popen('"%s" -jpeg -f 1 -l 1 "%s" pdfimage' % ("pdftoppm", pdf_path), shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()

    #### PDF SEPARATOR AND FUNCTION RUNNER

    img = cv2.imread('pdfimage-1.jpg')

    #pytesseract.pytesseract.tesseract_cmd = 'execsrc/Tesseract-OCR/tesseract'
    #pytesseract.pytesseract.tesseract_cmd = 'usr/bin/tesseract'

    e = pytesseract.image_to_string(img)

    if "Aurora Behavioral Healthcare LLC. Tempe AZ" in e:
        output, output2 = tessertxt(img)
        output.update(output2)
    elif "CPR CARE" in e:
        output = cloudvision()
    elif "ADMISSION RECORD" in e:
        output = tabled(pdf_path)
    elif "Oasis Behavioral Health" in e:
        output = cloudvision()

    #### OUTPUT ANALYSIS

    mapdict = {
        "Patient Name" : ["Patient (first name, middle name, last name)", "Patient's Full Name", "Resident Name"],
        "Gender" : ["Sex"],
        "Age" : ["Age"],
        "Birth Date" : ["Birthdate", "Birth Date", "BirthDaie"],
        "SSN" : ["Social Security #", "Soc Sec No"],
        "Phone Number" : ["Phone Number", "Previous Phone #", "Patient Phone"],
        "Address" : ["Location", "Previous address"],
        "Marital Status" : ["Marital", "Marital Status"],
        "Admission Date" : ["Admission Date", "Admit Date/Time"],
        "Religion" : ["Religion"],
        "Race" : ["Race & Ethnicity", "Race"],
        "Resident #" : ["Resident #"],
        "Language" : ["Primary Lang.", "P. Language"],
        "Primary Insurance" : ["Prim. Insurance"],
        "2nd Primary Insurance" : ["2nd Primary Insurance"],
        "Secondary Insurance" : ["Secondary Insurance"]
    }

    newlabeled = {}
    for line in mapdict.keys():
        newlabeled[line] = ""

    for line in output.keys():
        for row in mapdict.keys():
            if line in mapdict[row]:
                newlabeled[row] = output[line]

    try:
        newlabeled["Patient First Name"] = newlabeled["Patient Name"].split()[0]
    except:
        newlabeled["Patient First Name"] = newlabeled["Patient Name"]
    try:
        newlabeled["Patient Last Name"] = newlabeled["Patient Name"].split()[-1]
    except:
        newlabeled["Patient Last Name"] = ""

    del newlabeled["Patient Name"]

    return newlabeled

out = readmed("data/"+args[0])

print(json.dumps(out))

##Created by Sidharth Rao -- github.com/sidharthmrao
