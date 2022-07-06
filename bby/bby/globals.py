#
# globals.py - global definitions for BestBuy ML analysis
#
# Author: Eric g. Suchanek,PhD for Best Buy
# (c) 2022 Best Buy Inc., All Rights Reserved
# This is a restricted document and may not be shared.
# 
 
DEBUG = False

# Globals for our store
import py_compile


Our_Territory = 3
Our_Market = 21
Our_District = 125
Our_Store = "00494"

Territory_list = [1, 2, 3, 4, 14]

# List of all markets within territory 3
market_list_3 = [4, 5, 21, 24, 78, 79, 94]
market_list_locale_dict = {4:"Detroit/Indy", 5:"Boston", 21:"Ohio/Pittsburgh", 24:"Philly/Upstate NY", 78:"MKT 78 Philadelphia", 
                    79:"mkt 79 New York City", 93:"T33 Pooled Labor"}

# a list containing district numbers for each market within territory 3
district_list_4 = [16, 17, 49]
district_list_5 = [37, 39, 107]
district_list_21 = [14, 23, 52, 93, 125, 127]
district_list_24 = [43, 54, 87, 126]
district_list_78 = [235, 525, 530, 532, 533, 537, 538, 539, 540, 6541, 548]
district_list_79 = [215, 216, 217, 218, 219, 221, 222, 223, 224, 25, 226, 227, 228, 229, 231, 232, 233, 234, 252]
district_list_94 = [214, 220, 230, 479, 480, 484, 492, 498]

# Store numbers for each District within market 21 - we use text fields to keep the formatting right
district_stores_14 = [ "00228", "00229", "00368", "00371", "00490", "00491", "01025", "01094"]
district_stores_23 = [ "00162", "00168", "00271", "00278", "00279", "00285", "00286", "00758", "00879", "00880", "01050", "01099"]
district_stores_52 = [ "00227", "00259", "00333", "00335", "00791", "01010", "01121", "01477"]
district_stores_93 = [ "00145", "00156", "00292", "00295", "00339", "00390", "00570", "00573", "01096", "01266"]
district_stores_125 = [ "00154", "00161", "00266", "00274", "00617", "00494", "00790", "01252", "01474"]
district_stores_127 = [ "00230", "00232", "00327", "00489", "00858", "02501", "02512"]

# dict for all stores within market 21
district_stores_21_dict = {14:district_stores_14, 125:district_stores_23, 
                           52:district_stores_52, 93:district_stores_93, 125:district_stores_125,
                           127:district_stores_127}

district_stores_dict = {21:district_stores_21_dict,}

#
# global variables used for pathing relative to the notebook directory
#
_txt_extension = ".txt"
_raw_path = "../data/raw/"
_cleaned_path = "../data/clean/"    
_filename_prefix = "export_Main Hierarchy_"
_output_filename_prefix = "NPS_cleaned_"
_output_filename_prefix_natl = "NPS_NATL_"

_prom_words_path = _cleaned_path + "prom/"
_pass_words_path = _cleaned_path + "pass/"
_det_words_path = _cleaned_path + "det/" 
