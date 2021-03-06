# %%
# 
# Purpose: clean up raw NPS .xlsx files prior to training, write cleaned output to new .csv file
# Method: 
#   1) Remove missing values, stop words
#   2) Use textblob to calculate sentiment scores for NPS and Overall comments
#   3) Add sentiment scores for both the NPS Comment, and Overall Comment using 
#
# Author: Eric G. Suchanek, PhD
# (c)2022 BestBuy, all rights reserved
#

# %%
# library imports
import pandas as pd
import re

#TextBlob Features
from textblob import TextBlob

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder

from collections import Counter
import os


# %%
#
# global variables used for pathing relative to the notebook directory
#
DEBUG = False

_txt_extension = ".txt"
_raw_path = "../data/raw/"
_cleaned_path = "../data/clean/"    
_filename_prefix = "export_Main Hierarchy_"
_output_filename_prefix = "NPS_cleaned_"

_prom_words_path = _cleaned_path + "prom/"
_pass_words_path = _cleaned_path + "pass/"
_det_words_path = _cleaned_path + "det/" 

# %%
def cleanup_data(data):
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)

    # Remove elipsis
    data = re.sub("\...",  " ", data)

    # Strip escaped quotes
    data = re.sub('\\"', '', data)
 
    # Strip quotes
    data = re.sub('"', '', data)
        
    return data

# %%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(sent):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sent)
 
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = []
 
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return(filtered_sentence)

# %%
s2 = remove_stopwords(' this uh good service Eric!! blah')
print(s2)

# %%
# Adding sentiment dimensions with textblob
def tb_enrich(ls):
    #Enriches a column of text with TextBlob Sentiment Analysis outputs
    tb_polarity = []
    tb_subject = []

    for comment in ls:
        sentiment = TextBlob(comment).sentiment
        tb_polarity.append(sentiment[0])
        tb_subject.append(sentiment[1])
    return tb_polarity, tb_subject

# %%
# 
# write the list of comments to a file based on the path, district, and prefix
def write_sentences(sentencelist, storename, _path, district, prefix):
    lc = 1
    outfilename = f'{_path}{district}_{storename}_{prefix}_{lc}{_txt_extension}'
    outfile = open(outfilename, 'w')

    for items in sentencelist:
        outfile.writelines(items)
        outfile.write('\n')
        lc += 1
    outfile.close()
    return

# %%
import shutil
import glob, os, os.path

def clean_txt_dirs():
    _cleaned_path = "../data/clean/"    
    
    cwd = os.getcwd()

    _prom_words_path = os.path.join(cwd, "../data/clean/prom/")
    _pass_words_path = os.path.join(cwd, "../data/clean/pass/")
    _det_words_path = os.path.join(cwd, "../data/clean/det/" )
    
    filelist = glob.glob(os.path.join(_prom_words_path, "*.txt"))
    
    for f in filelist:
        os.remove(f)
    
    filelist = glob.glob(os.path.join(_pass_words_path, "*.txt"))
    for f in filelist:
        os.remove(f)

    filelist = glob.glob(os.path.join(_det_words_path, "*.txt"))
    for f in filelist:
        os.remove(f)   
    
    return


# %%
# convert 'promoter' 'passive' 'detractor' to numerical index
def nps_to_code(nps_list):
    codelist = []
    for comment in nps_list:
        if (comment =="Promoter"):
            codelist.append('2')
        elif (comment == "Passive"):
            codelist.append('1')
        elif (comment == "Detractor"):
            codelist.append('0')
        else:
            codelist.append('xxx')
    return(codelist)
        

# %%
# write the comments (nps and overall) to files in specific directories based on whether they are
# promoters, passive or detractors

def write_comments(district=125):
    storelist = district_stores_dict.get(district)
    if (DEBUG):
        print(f'..Writing cleaned comments for District: {district}')

    for storename in storelist:
        if (DEBUG):
            print(f'....Writing comments for store: {storename}')
        input_filename = f"{_cleaned_path}{_output_filename_prefix}{storename}_{district}.csv"
        new_df = pd.read_csv(input_filename)

        pos_comments = new_df.loc[new_df['NPS_Code'] == 2]
        pass_comments = new_df.loc[new_df['NPS_Code'] == 1]
        detr_comments = new_df.loc[new_df['NPS_Code'] == 0]

        pos_stringlist = pos_comments['NPSCommentCleaned']
        pass_stringlist = pass_comments['NPSCommentCleaned']
        detr_stringlist = detr_comments['NPSCommentCleaned']
        
        write_sentences(pos_stringlist, storename, _prom_words_path, district, "NPSComment")
        write_sentences(pass_stringlist, storename, _pass_words_path, district, "NPSComment")
        write_sentences(detr_stringlist, storename, _det_words_path, district, "NPSComment")

        pos_stringlist = pos_comments['OverallCommentCleaned']
        pass_stringlist = pass_comments['OverallCommentCleaned']
        detr_stringlist = detr_comments['OverallCommentCleaned']

        write_sentences(pos_stringlist, storename, _prom_words_path, district, "OverallComment")
        write_sentences(pass_stringlist, storename, _pass_words_path, district, "OverallComment")
        write_sentences(detr_stringlist, storename, _det_words_path, district, "OverallComment")

    return



# %%
# NPS spreadsheet cleanup and reformatting for natural language processing
# data paths and filename patterns
# Assumes following directory structure
# main directory/
#  - notebook/
#  - data/
#  - raw/
#   -clean/

# a list containing district numbers
all_districts = [125,]

# Store numbers for our District - we use text fields to keep the formatting right
# 
district_stores_125 = [ "00154", "00266", "00274", "00617", "00494", "00790", "01252", "01474"]
district_stores_dict = dict([(125, district_stores_125)])

our_district = 125
our_store = "00494"

# Given the district, read the raw .xlsx file, fill null values, and perform the following cleanup:
#  - remove URL, emails, extraneous punctuation
#  - calculate sentiment scores from the blob package
# write a new .csv file with NPS comment, overall comment, confirm number, location, workforce, NPS rating
#

def NPS_cleanup(district):   

    storelist = district_stores_dict.get(district)
    
    # filename for aggregated final file
    cleaned_filename = f"{_cleaned_path}NPS_District_{district}.csv"
    output_filename_list = []

    print(f'Processing District: {district}')
    
    for storename in storelist:
        all_promotors_list = []
        all_passive_list = []
        all_detractor_list = []

        new_df = pd.DataFrame()
        input_filename = f"{_raw_path}{_filename_prefix}{storename}.xlsx"
        output_filename = f"{_cleaned_path}{_output_filename_prefix}{storename}_{district}.csv"
        output_filename_list.append(output_filename)
        storename_string = f'_{storename}'

        # read the .xlsx file, skipping the first 4 lines since they are not the
        # actual column headers
        print(f'..Processing Store: {storename}')
        all_df = pd.read_excel(input_filename, header=3)
        
        # Fill null values.
        all_df['NPS?? Comment'].fillna("NONE", inplace = True)
        all_df['Overall Comment'].fillna("NONE", inplace = True)
        all_df['ConfirmationNumber'].fillna("0000", inplace = True)
        all_df['Service Order ID'].fillna("0000-0000", inplace = True)

        all_df['Location'] = storename_string

        # transform the text into an integer value. 0 = detractor, 1 = passive, 2 = promoter
        NPS_code = nps_to_code(all_df['NPS?? Breakdown'].values.tolist())
        
        new_df = all_df[['Location','Workforce','NPS?? Breakdown']].copy()
        new_df['NPS_Code'] = NPS_code

        #Splitting pd.Series to list

        temp = []
        data_to_list = all_df['NPS?? Comment']
        
        for i in range(len(data_to_list)):
            temp.append(cleanup_data(data_to_list[i]))
        new_df['NPSCommentCleaned'] = temp

        new_df['NPSCommentPolarity'], new_df['NPSCommentSubjectivity'] = tb_enrich(temp)
        
        temp = []
        data_to_list = all_df['Overall Comment'].values.tolist()
        for i in range(len(data_to_list)):
            temp.append(cleanup_data(data_to_list[i]))
        new_df['OverallCommentCleaned'] = temp
        
        OverallCommentPolarity, OverallCommentSubjectivity = tb_enrich(temp)
        new_df['OverallCommentPolarity'], new_df['OverallCommentSubjectivity'] = OverallCommentPolarity, OverallCommentSubjectivity
        
        # write the store file
        new_df.to_csv(output_filename, index=False)

    # now concatenate all of the resulting files into a single district .csv file
    total_df = pd.DataFrame()
    for infile in output_filename_list:
        df = pd.read_csv(infile)
        total_df = pd.concat([total_df, df], axis=0)
        #os.remove(infile)

    print(f'..Writing {cleaned_filename} with a total of {total_df.shape[0]} comments')
    total_df.to_csv(cleaned_filename, index=False)
    
    write_comments(district)
    
    print('Done')
    return

# %%
NPS_cleanup(125)      

# %%



